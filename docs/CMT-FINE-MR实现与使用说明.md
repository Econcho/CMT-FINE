# CMT-FINE-MR 实现与使用说明

> 实现基线：本地 fork 的 D-FINE master，实现时 HEAD 为 956d170。本文描述当前代码事实；方案动机和完整推导见 CMTDet-main/docs/核心文档/基于D-FINE融合CMT的三版模型方案与优先级.md。

## 1. 实现范围

当前代码实现方案 A：CMT-FINE-MR（Moment-constrained Refinement）。它保留官方 D-FINE 的 backbone、HybridEncoder、query 选择、FDR、GO-LSD、postprocessor 和 COCO evaluator，只新增一条 CMT 定位证据支路，并在 decoder 的 pred_corners → distance2bbox 之间加入可选的矩约束分布投影。

因此：

- 官方检测输出、COCO AP/AR 统计口径和测试命令保持不变；
- CMTMomentRefiner.mode=baseline 时不计算 CMT，走原 D-FINE 路径；
- projection_strength=0 时分布投影逐位返回原 logits，但仍会计算和训练证据支路；若 `class_residual=True`，分类分数仍会受到矩描述影响；
- 新增训练损失只在 CMT 主配置中启用，官方配置不受影响；
- 尚未修改 ONNX、TensorRT 或其他部署导出路径。

这是一版可训练、可消融的研究实现，不代表 CMT 已经提高检测精度或相位稳定性。

## 2. 数据流

~~~mermaid
flowchart TD
    X["RGB 输入 B×3×H×W"] --> O["官方 HGNetV2 + HybridEncoder"]
    X --> E["EvidenceEncoder<br/>stride 1"]
    E --> M["MomentPyramid<br/>M4/M8/M16/M32"]
    O --> Q["官方 query 初始化与 decoder layer"]
    Q --> U["原 FDR 累积四边 logits"]
    Q --> R["MomentReader"]
    M --> R
    R --> G["多窗口权重 + 可靠性门"]
    R --> C["q + alpha * MLP(d_M)<br/>可选分类残差"]
    U --> P["MomentDistributionProjector<br/>批量固定步 Newton"]
    G --> P
    P --> B["官方 Integral + distance2bbox"]
    C --> S["官方分类头"]
    B --> N["下一 decoder layer / 官方输出"]
~~~

CMT 原始状态为 [m, px, py, Qxx, Qxy, Qyy]。M4 由固定卷积核一次生成，M8/M16/M32 用换原点恒等式合并。原始矩始终使用 FP32；可学习门和 decoder 保持官方 AMP 行为。

当训练命令启用 CUDA AMP 时，`MomentPyramid`、`MomentReader` 的积分矩图和 `MomentDistributionProjector` 会在各自的数值敏感区间局部禁用 autocast。原因是 CUDA autocast 会把固定矩卷积自动降为 FP16，二阶项 `x²m` 在 640 输入坐标范围内容易溢出，进而把矩中心、门控和预测框传播为 NaN，最终在 Hungarian matcher 的 GIoU 检查处触发断言。局部使用 FP32 不会关闭主干、encoder 或 decoder 的 AMP；对应行为由 `test_moment_pyramid_stays_fp32_under_autocast` 回归测试保护。

## 3. 新增和改动文件

| 文件 | 作用 |
|---|---|
| src/zoo/dfine/cmt.py | 证据编码、矩金字塔、积分矩读取、可靠性门和 KL 分布投影 |
| src/zoo/dfine/dfine.py | 可选注入 CMT，并在 backbone 前生成矩状态 |
| src/zoo/dfine/dfine_decoder.py | 在 FDR logits 解码前调用 CMT；训练时返回诊断张量 |
| src/zoo/dfine/dfine_criterion.py | 可选的证据中心、矩中心和 gate 辅助损失 |
| configs/dfine/cmt_fine_mr_hgnetv2_s_coco.yml | HGNetV2-S 完整示例配置 |
| tests/test_cmt_fine_mr.py | 数学、退化等价、投影方向、梯度、消融冻结和 decoder 输出契约测试 |

没有修改 postprocessor、COCO evaluator、数据集类和训练循环。

## 4. CMT 组件

### 4.1 EvidenceEncoder

输入是与 backbone 完全相同的、变换后的 RGB 张量。默认结构为：

~~~text
3×3 Conv → SiLU → depthwise 3×3 → SiLU → 1×1 logit
~~~

默认宽度 16。证据为 softplus(logit)，保证非负并避免 ReLU 死区；可通过 evidence_activation=relu 做原始设定对照。该分支不接受 backbone 特征的乘法门控，避免直接继承有步长语义路径的相位误差。

### 4.2 MomentPyramid

M4 的六个矩由一个固定 6×1×4×4 kernel、stride 4 的卷积同时计算，没有逐像素 Python 循环。更粗层每次合并 2×2 子单元，并显式加入原点位移项。

当前 reader 使用 M4；M8–M32 被保留为后续多尺度读取接口。主要计算成本来自全分辨率 EvidenceEncoder。

### 4.3 MomentReader

为每个 query 按参考框建立三个矩形支持窗口，默认倍率为 [0.75, 1.5, 3.0]，最小半径 4 个输入像素。

实现先把 M4 转成全局原始矩并建立二维积分图，每个窗口只需四次 gather 即可得到质量、一阶和二阶矩，复杂度接近 O(B×Q×S)，没有逐 query 的 grid_sample 或大特征重复转换。

三个窗口的中心用学习到的凸权重融合；全部窗口无有效质量时回退到 D-FINE 参考中心，gate 强制为 0。

矩描述还可通过一个窄 MLP 形成 `q + alpha * MLP(d_M)` 分类输入，默认 `alpha=0.01` 且可学习。它不覆盖 decoder query，也不改变下一层状态，只影响当前层官方分类头；通过 `class_residual=False` 可独立关闭，以区分定位投影和置信度稳定性收益。

### 4.4 MomentDistributionProjector

D-FINE 四边顺序按官方代码使用 left、top、right、bottom。投影保留固定的 ref_points_initial 和官方非均匀 bin support，分别求解水平、垂直单变量对偶问题。默认使用 4 次批量 Newton 更新。

所有 query、边和 bin 一起计算，没有 query 级 Python 循环。求解使用 FP32，结果转换回原 logits dtype。校正后的 logits 继续进入官方 Integral、LQE、FGL 和 GO-LSD。

projection_strength=0 直接返回输入 Tensor，不执行 softmax 或 Newton，因此可做逐位恒等检查。

## 5. 训练损失

主配置在官方 vfl、bbox、giou、fgl、ddf 外增加：

| 损失 | 默认权重 | 含义 |
|---|---:|---|
| loss_cmt_evidence | 0.25 | 在 M4 分辨率上监督双线性中心 target，背景使用较低权重 focal 项 |
| loss_cmt_center | 0.10 | 对最终匹配 query 的矩中心做尺度归一化 Smooth L1；无有效质量不计 |
| loss_cmt_gate | 0.05 | 根据矩中心和语义中心谁更接近 GT 生成 detached 软目标 |

辅助 loss 只作用于最终正常 query；encoder、pre-head、DN 和中间辅助输出仍只使用其官方损失。框匹配仍由官方 Hungarian matcher 完成。

这些权重是保守起点，需要用训练日志观察三个量级后再调。不能因为辅助 loss 数值大就直接增加权重。

## 6. 配置、训练和消融

入口配置：

~~~powershell
python train.py -c configs/dfine/cmt_fine_mr_hgnetv2_s_coco.yml --use-amp --seed 42
~~~

单卡或小数据集训练可用 CLI 覆盖 batch：

~~~powershell
python train.py -c configs/dfine/cmt_fine_mr_hgnetv2_s_coco.yml --use-amp --seed 42 -u train_dataloader.total_batch_size=8 val_dataloader.total_batch_size=8
~~~

恢复训练和加载官方预训练检测器：

~~~powershell
python train.py -c configs/dfine/cmt_fine_mr_hgnetv2_s_coco.yml -r output/run/last.pth
python train.py -c configs/dfine/cmt_fine_mr_hgnetv2_s_coco.yml -t path/to/official_dfine_s.pth
~~~

从官方检测权重 tuning 时，新增 CMT 参数不存在是预期现象；必须查看加载日志，确认旧 D-FINE 参数没有大面积 shape mismatch。

### 6.1 核心消融模式

| 模式 | CMTMomentRefiner.mode | 作用 |
|---|---|---|
| 官方基线 | baseline | 完全跳过 CMT 计算，新增 loss 也不会产生 |
| 证据支路对照 | evidence_only | 训练证据、中心和 gate，但不修改 FDR logits |
| 质量对照 | mass_only | 只用每个 M4 单元的质量和单元中心 |
| 一阶矩 | first_order | 使用质量与一阶矩，不使用二阶可靠性描述 |
| 完整模型 | full | 使用零、一、二阶矩和分布投影 |

示例：

~~~powershell
python train.py -c configs/dfine/cmt_fine_mr_hgnetv2_s_coco.yml --use-amp --seed 42 -u CMTMomentRefiner.mode=first_order output_dir=./output/ablation_first_order
~~~

### 6.2 其他关键消融参数

~~~text
CMTMomentRefiner.refine_layers=last
CMTMomentRefiner.learned_gate=False
CMTMomentRefiner.fixed_gate=0.5
CMTMomentRefiner.class_residual=False
CMTMomentRefiner.class_residual_init=0.01
CMTMomentRefiner.projection_strength=0.0
CMTMomentRefiner.solver_steps=1
CMTMomentRefiner.support_scales=[1.5]
CMTMomentRefiner.evidence_activation=relu
~~~

hidden_dim 必须与 DFINETransformer.hidden_dim 一致，reg_max 也必须一致。代码会在 query 维不一致时直接报错。`projection_strength=0` 只保证四边分布逐位不变；若需要预测路径的完整定位对照，还应同时设置 `class_residual=False`。

构造器会自动冻结被配置关闭的路径：`baseline`/`enabled=False` 冻结整个 CMT，固定 gate 冻结 gate head，关闭分类残差冻结对应 MLP 和尺度参数。因此这些消融可沿用官方 `find_unused_parameters=False`，无需为 DDP 兼容性增加每轮未使用参数扫描。

建议第一轮至少跑 baseline、evidence_only、mass_only、first_order、full、full+零投影、full+仅末层、full+固定 gate。所有组固定数据划分、官方预训练来源、有效 batch、epoch、增强策略和随机种子。

## 7. 测试与评测

评测仍使用官方 D-FINE：

~~~powershell
python train.py -c configs/dfine/cmt_fine_mr_hgnetv2_s_coco.yml -r output/run/best_stg1.pth --test-only
~~~

输出仍由 DFINEPostProcessor 处理，指标仍为官方 COCO AP50:95、AP50、AP75、APsmall、APmedium、APlarge 等。当前实现没有加入自定义 Tiny AP，也没有改变 evaluator 口径。

核心单元测试：

~~~powershell
python -m unittest discover -s tests -p "test_*.py" -v
~~~

测试覆盖分层矩合并、1 px 平移读取、零强度恒等、投影方向以及完整前向/反向有限梯度。

## 8. 效率设计

- M4 六矩由一次固定卷积计算，粗层用 reshape 与广播合并。
- query 窗口通过积分矩图读取，不重复采样整张高分辨率特征。
- Newton 求解完全批量化，复杂度约为 O(B×Q×L×K×I)。
- 仅保存最后层 CMT 诊断用于辅助 loss。
- 原始矩、积分图和 Newton 求解在局部禁用 autocast 并使用 FP32；EvidenceEncoder 与 decoder 仍可使用 AMP。
- baseline 不创建证据图；零投影强度跳过求解。
- HGNetV2-S 示例配置中的 CMT 分支有 56,428 个可训练参数；该数值不含官方 D-FINE 主体，也不代表实际 FLOPs。

训练前仍应在目标 GPU 上分别测 baseline/full 的 iteration 时间、峰值显存和数据加载占比。本地验证是 CPU 合成检查，不能替代服务器实测。

## 9. 已知边界

- 局部权函数是矩形窗口，积分读取在单元级精确，但窗口边界相对 query 是离散的，不提供连续边界梯度。
- 当前只在最终正常 query 上监督矩中心和 gate，中间层通过检测损失与校正分布获得梯度。
- 多目标落入同一窗口时，矩中心可能位于两者之间；学习门只能回退，不能从六矩唯一分离实例。
- EvidenceEncoder 增加全分辨率计算，真实训练成本必须实测。
- 未验证分布投影的 TorchScript、ONNX 或 TensorRT 导出。
- 未完成真实数据训练，AP、PFR、收敛速度和最佳 loss 权重均未知。

## 10. 推荐实施顺序

先用官方配置复现 D-FINE-S；再用 CMT 配置加载同一官方权重，做 100–300 iteration 的速度、显存和 loss smoke。确认没有 NaN、gate 不全零/全一、三个辅助 loss 量级合理后，再做配对训练。

研究结论需要同时检查 COCO 检测指标、候选召回、中心误差和相位敏感性。PFR 降低但 AP/AR 降低，可能只是稳定漏检，不应视为成功。
