# CMTDet：RTX 5060 Ti 16GB 训练瓶颈复评与实测报告

日期：2026-09-14。对象：HGNetV2-B0、640×640、完整 CMTDet。目的：解释每轮约 2 小时，并验证优先优化方向。

## 1. 结论

**当前主要瓶颈是 CMT 候选矩读取的执行和反向传播，不是 HGNetV2-B0 参数量过大，也不是 DUT 数据读取速度。** 其中，PyTorch 2.7.1 的通用 `prod(dim)` 反向会检查零元素并读取 CUDA 标量；这与 CMT 的大量二维求积、紧支撑核中的零值、细粒度 query 分块叠加，造成了明显开销。

连续真实 DUT 数据测试保持 `batch_size=2、accumulation=8`：

| 项目 | 当前配置 | 实验性优化组合 |
|---|---:|---:|
| 测量 micro-batch 数 | 24（另8个预热） | 24（另8个预热） |
| 平均完整 micro-batch | 2.760 s | 0.838 s |
| 中位数 | 2.713 s | 0.833 s |
| DataLoader 等待均值 | 0.389 ms | 0.409 ms |
| 峰值 allocated 显存 | 6.03 GiB | 9.35 GiB |
| 峰值 reserved 显存 | 6.35 GiB | 10.46 GiB |
| AMP 跳过更新次数 | 0 | 0 |
| 5200张训练图线性外推，**不含验证** | 119.6 min/epoch | 36.3 min/epoch |

这组实测吞吐提升为 **3.29 倍**。优化组合包含：CMT 分块32→400、注意力分块1024→4096、二维 `prod` 显式相乘、关闭编码器 checkpoint，以及4个CPU线程。线程数本身无明显收益；单因素证据见后文。

**这里只完成诊断与独立性能原型，没有把实验性替换写入正式 `src/cmtdet`，没有重启正式长训练，也没有训练精度/AP结论。** “约36分钟”是早期短测外推，不是已经跑完一个完整 epoch 的记录，且不包括验证、保存和后期候选分布变化。

## 2. 核实的环境与原训练状态

- 工作目录：`/data/coding/CMTDet-main`。
- GPU：RTX 5060 Ti，16311 MiB；驱动580.142。
- Python：3.11.16；PyTorch：2.7.1+cu128；PyTorch CUDA运行时12.8。`nvidia-smi`显示的13.0是驱动支持版本，不能替代运行时版本。
- 训练/验证/测试图像数：5200 / 2600 / 2200。
- 参数量：15,207,118，约15.21M。
- 骨干预训练：`weights/PPHGNetV2_B0_stage1.pth`；初始化其余检测器/CMT参数，未使用已训练完成的检测器权重。
- 实际配置：hidden256，encoder2层、decoder6层、queries300、denoising100、P2启用、CMT full、AMP启用。
- 本次固定训练 batch 实际包含300个常规query + 100个去噪query，共400个；不是只处理300个。
- 原训练已手动中断，`last_completed_epoch=0`；服务器开始测量时无GPU任务。本次没有停止用户的训练。
- 原始训练日志、配置、环境和中断状态备份在实验目录 `original_run/`。

服务器源码没有 `.git`，但关键文件 `attention.py/moments.py/engine.py` SHA256 与本地已推送版本一致；原训练环境记录中的源码哈希也一致。因此此次慢速不是因为服务器漏用了上一轮优化。

CPU可见96逻辑核，PyTorch默认48线程，但容器CPU配额等效15核。这是合理怀疑对象；实测48→4线程约2.784→2.800秒/batch，无加速，不能把它写成主因。

## 3. 测试设计与解释范围

### 3.1 固定真实 batch 的单因素测量

使用真实 DUT 训练图，固定随机种子、预训练骨干和检测器初始化；3个预热 + 6个测量 micro-batch。每阶段末尾同步CUDA后计时；所有候选方案单独启动进程、顺序运行。正式源码不变。

这组测试为了拆分前向、损失和反向，每次清梯度、重复固定图像，最后单独测量一次优化器更新；**不能单独作为真实训练吞吐结论**。因此又做了3.2节的连续训练短循环。

最初 `actual_t48` 是全随机初始化的探路结果（约3.666秒），随后发现实际使用了仓库内骨干权重，重新测量了 `pretrained_t48`。报告以**匹配实际预训练设置的2.784秒**为固定batch基线，不混用两者。

### 3.2 连续真实数据与实际更新

`real_loop.py` 每个方案处理32个真实训练 micro-batch、共64张图，前8个用于预热，后24个计时。使用同一随机种子、相同图像ID顺序、相同骨干预训练、训练增强、学习率预热、AMP、梯度裁剪与8步累积；全程共4次优化器更新，测量段包含3次。

每个micro-batch的时间包含数据等待、搬运、前向、损失、反向和轮到的优化器更新；阶段拆分屏障已去掉，仅在micro-batch边界同步计时。没有保存或覆盖正式模型参数。原配置与优化组合的图像ID序列已核对一致。

### 3.3 Profiling限制

服务器 CUPTI 报 `CUPTI_ERROR_INVALID_DEVICE`，所以 `torch.profiler` **没有有效CUDA kernel活动时间**。本报告使用：

1. CUDA同步后的实际墙钟时间进行性能对比；
2. CUDA Event测量模块前向时间区间（包含区间内GPU等待CPU发射的空闲，不能解释为纯kernel计算时间）；
3. CPU trace 追溯 `ProdBackward1 → aten::item` 调用来源。

导出JSON中为0的 `device_self_us` 不表示该算子没有GPU开销。独立reader的dispatch计数可能改变框架内部选路，只用于辅助检查，不能代替未经dispatch插桩的时延。checkpoint重计算可能提前结束，模块包装器未记录完成的重计算调用不代表没有重计算。

## 4. 单因素实验结果

单位：时间为毫秒/micro-batch，显存为GiB。此表不包含数据加载和按8步摊销的优化器更新。

| 配置 | 前向 | 损失 | 反向 | 合计 | 峰值 allocated |
|---|---:|---:|---:|---:|---:|
| 实际配置，48线程 | 713.4 | 58.5 | 2011.9 | 2783.8 | 5.27 |
| 仅改4线程 | 716.0 | 58.7 | 2024.8 | 2799.6 | 5.27 |
| 4线程 + CMT chunk=128 | 400.8 | 83.8 | 1684.8 | 2169.4 | 5.27 |
| 4线程 + CMT chunk=400 | 312.8 | 58.2 | 1660.8 | 2031.9 | 5.28 |
| 4线程 + attention chunk=4096 | 674.7 | 58.7 | 1900.2 | 2633.6 | 5.27 |
| 4线程 + CMT 400 + attention 4096 | 267.9 | 57.8 | 1497.9 | 1823.6 | 5.28 |
| 上一项 + 关闭编码器 checkpoint | 278.9 | 62.5 | 1443.1 | 1784.5 | 8.69 |
| 4线程 + 二维 prod 显式相乘 | 760.3 | 61.4 | 1327.2 | 2148.9 | 5.17 |
| 组合分块 + 二维相乘，保留 checkpoint | 279.3 | 62.3 | 554.8 | 896.5 | 5.18 |
| 组合分块 + 二维相乘 + 关闭 checkpoint | 272.1 | 61.5 | 492.8 | 826.5 | 8.59 |
| baseline 诊断模式：关闭 CMT | 173.8 | 43.0 | 487.6 | 704.4 | 2.51 |

解读：

- 当前固定batch基线反向约2012ms，占合计约72%；损失约59ms，仅约2%。
- 增大CMT分块本身就有实际收益，400优于32和128；此配置处理400个训练query，32意味着13个query块。
- 注意力分块增大有收益，但幅度小于解决CMT读取问题。
- **只调整现有参数**，组合分块并关闭checkpoint可从2.784降至1.784秒，约1.56倍；无法复现包含二维求积替换的3.3倍收益。
- 求积替换与大分块组合后约0.896秒；关闭checkpoint进一步至0.826秒，换来约3.4GiB额外allocated显存。
- 保留checkpoint的0.896秒方案也是合理选择，显存更充裕，适合后期候选支持区域变大时验证稳定性。
- `baseline`模式用于定位增量开销，会关闭CMT、改变损失/匹配次数与模型能力，**不是建议通过关闭创新模块来“加速CMTDet”**。

## 5. 具体瓶颈及代码原因

### 5.1 一级优先：二维 prod 的通用反向

源码位置：`src/cmtdet/moments.py` 的 `_candidate_rows`，以及其他二维面积/协方差表达式。典型代码包括核函数两个坐标分量的 `.prod(-1)`、`mean.prod(-1)`、`a.prod(-1)`。

CPU trace 中，当前预训练配置的一次前后向有 **603次 `_local_scalar_dense` 位于 `ProdBackward1` 内部**；相关CPU区间累计约360ms。这是带profiler的定位证据，不能直接当成未经插桩的可节约时间。

对应PyTorch 2.7.1源码 `prod_backward` 会计算零元素数量，再执行：

```cpp
int64_t total_zeros = slice_zero_count.sum().item<int64_t>();
```

存在零元素时还会进入更通用的累计乘积处理。CMT紧支撑核在支持域外产生大量零；因此即使Python层没有写 `.item()`，反向仍可能发生标量同步与额外计算。[PyTorch 2.7.1源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/csrc/autograd/FunctionsManual.cpp#L759)

这里很多求积固定只有两个实数分量，可用：

```python
x, y = v.unbind(-1)
area = x * y
```

其数学形式为 `f(x,y)=xy`，梯度为 `(y,x)`，零元素也自然适用。无需通用求积的零计数分支。正式实现应在具体位置替换，并保留dtype/autocast行为、消融接口；**本次测试的全局Tensor方法替换仅用于独立进程内的归因原型，不应直接放进训练入口**。

### 5.2 一级优先：CMT query分块过细，动态分桶仍然同步

`read_candidates` 遍历query块，再遍历半径桶；每个桶调用 `selected.nonzero()`，随后做索引收集、矩变换、加权归约与多个输出scatter。当前stride4、support_max120对应6个半径桶。

400个query、chunk32需要13块；6个decoder层，共 `13×6×6=468` 次前向动态nonzero调用，CPU trace与此一致。chunk400将结构上的调用数降至36，数量减少13倍。

**GPU上执行分桶不等于CPU与GPU完全异步。** 动态nonzero的输出大小依赖GPU计算，需要主机获知大小；相关机制见[PyTorch CUDA说明](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/)。上一轮实现移除了显式radius `.item()`，但不能据此声称去除了同步。

单次reader隔离实验，从实际模型抓取第一层reader的真实state和400个候选，测量其前后向：

| query_chunk | 求积实现 | 前向ms | 反向ms |
|---|---|---:|---:|
| 32 | 原始 prod | 77.42 | 238.21 |
| 32 | 显式二维相乘 | 83.65 | 127.61 |
| 400 | 显式二维相乘 | 12.48 | 19.45 |

这一隔离结果与整网测试方向一致。后续更深优化可考虑先全局分桶，再按内存预算切块；避免“每个query小块都对全部桶动态筛选”，减少完整输出的反复 `index_put` 和小autograd节点。若采用自定义融合reader，应同时设计反向，避免仅优化前向。

### 5.3 二级优先：P2全位置编码器与PyTorch deformable attention

640输入、P2/P3/P4/P5的空间位置总数为：

```text
160² + 80² + 40² + 20² = 34,000
```

这34,000是编码器token数，和300个检测query不同。两个encoder层都会处理全部位置。注意力chunk1024时每层34块、4个level，单次两层前向为272次grid_sample调用；chunk4096时降为72次。checkpoint还会带来重计算。

上一轮把FP32转换移出循环，解决的是重复转换和保存问题；仍有多次grid_sample及其反向、Python循环、分块归约、拼接等开销。

`attention_backend=cuda` **当前仍只是接口占位和回退逻辑**，不存在已接入的融合CUDA实现。单纯改这个字段不会获得融合注意力加速。

建议先使用已实测的分块调整；在CMT一级瓶颈修复后，再评估真实融合MSDeformAttention的收益。直接删除P2、减少encoder层或检测query会改变模型结构/能力，不应排在当前等价执行优化前面。

### 5.4 不是当前首要矛盾的部分

一次额外前向模块测量如下，仅用于定位，不与前述6次均值混为同一次测量：

| 模块 | 调用数 | CUDA Event区间合计ms |
|---|---:|---:|
| `backbone` | 1 | 8.85 |
| `evidence_and_encode` | 1 | 56.87 |
| `moment_fusion` | 1 | 2.65 |
| `encoder` | 2 | 217.36 |
| `decoder` | 6 | 19.29 |
| `moment_reader` | 6 | 442.72 |
| `moment_refiner` | 6 | 9.91 |

骨干前向约9ms，明显小于候选矩读取；CMT整图证据生成也有开销，但本次应先优化reader。

连续真实数据测得DataLoader等待平均约 `0.389ms`，远小于约2760ms的训练micro-batch；目前盲目增加workers、换SSD或缩小骨干不会解决主瓶颈。

完整模式每次criterion执行13次批级matcher调用（常规6层、encoder1次、temporary6次），batch2且都有GT时对应26次图像级Hungarian求解。损失总耗时约59ms。上一轮“复用匹配”仅在temporary/final共享同一框张量时适用；full模式经过refiner后通常不满足，不能把full模式宣称为已经省掉这些匹配。

## 6. 验证环节的短测

使用相同初始化模型、相同前16张val图、batch1；不做精度比较。full使用FP32并保存匹配详情，light使用当前配置的AMP并省略部分详情输出。

| CMT chunk | 验证模式 | 总墙钟s | 前向s | metric s | serialization s |
|---|---|---:|---:|---:|---:|
| 32 | full | 7.772 | 6.942 | 0.398 | 0.073 |
| 32 | light | 6.637 | 5.828 | 0.368 | 0.075 |
| 400 | full | 3.950 | 3.088 | 0.243 | 0.224 |
| 400 | light | 3.119 | 2.446 | 0.217 | 0.075 |

增大CMT分块在验证前向也有效。full32→full400保持FP32设置，16图总时间约7.77→3.95秒。但只有16张图，包含初始化/加载成本，不能把COCO计算和序列化时间按比例精确外推到2600张。

`full_val_interval=0`表示每次都是full；当前验证确实会增加epoch总时间。不过约2小时的训练主体已经可由 `5200/2×2.76s`解释，不能归因于保存JSON或COCO评估。

正式比较精度应固定验证精度设置。light启用AMP可能导致数值和best checkpoint选择变化，不能将其与FP32 full视为完全相同的实验口径。建议先优化相同验证精度下的reader，再考虑验证频率和详情保存频率。

## 7. 数值检查与研究结论的边界

- 同一真实reader输入、同一chunk32下，显式二维相乘与原prod的stats、descriptor、support、boxes梯度和state梯度，在本次CUDA样本中最大绝对差均为0。
- 同时把chunk32改为400后，descriptor最大绝对差约 `4.77e-7`；boxes梯度约 `7.45e-8`；state梯度约 `2.86e-6`。
- raw stats最大绝对差为0.75，需要结合约3,624,614的整体最大幅值理解；`max(abs(diff))/max(abs(reference))≈2.07e-7`。这是整体归一化误差，不等于每个近零元素的相对误差。
- 分块变化可能改变浮点归约/GEMM顺序；**不能承诺bitwise相同或后续训练轨迹完全一致**。
- 连续短训的两组loss轨迹在更新后存在差异。相同数学目标和局部梯度检查不等于已经证明最终AP保持不变；本次没有这项实验。
- 正式实现还应覆盖全零、单零、边界支持、负值、不同dtype/autocast、不同query数和support范围的输出/梯度回归；统一种子做更长的训练与val对照。
- 16GB上本次峰值是早期权重、64张真实图的实测。后期预测支持范围、GT数量和分配器状态可能变化，需要长一点的显存压力验证。

## 8. 建议的实施顺序

| 优先级 | 改动 | 性质 | 验收依据 |
|---|---|---|---|
| P0 | 在CMT二维核/统计量中显式相乘替代prod | 数学表达等价，需保持dtype与梯度行为 | 输出/梯度测试；同步次数；完整训练micro-batch |
| P0 | CMT query_chunk提高到400，保留可配置 | 执行粒度变化 | 400query实测；大支持范围显存；更多真实图 |
| P1 | attention_chunk提高到4096 | 执行粒度变化 | FP32/AMP数值检查；完整前后向 |
| P1 | 显存允许时关闭encoder checkpoint | 以显存换重计算时间 | 峰值显存与长一点的连续训练 |
| P2 | reader按全query分桶后再切块，减少动态筛选/scatter | 需修改实现 | 保持参考reader；覆盖forward/backward及边界 |
| P2 | 真正接入融合deformable attention | 新执行后端 | forward/backward正确性与GPU端实测 |
| P3 | 批量化匹配/target生成、减少每步同步、优化器细节 | 次要执行开销 | 重新profiling后决定 |

在正式代码修改前，现有参数可先验证：

```bash
source /data/coding/setup/env_cmtdet.sh
cd /data/coding/CMTDet-main
python -m cmtdet train \
  --config configs/cmtdet_hgnetv2_b0.yaml \
  --data data/dut_coco/data.yaml \
  --output runs/dut_hgnetv2_b0_chunk400_trial \
  --device cuda \
  --pretrained-backbone weights/PPHGNetV2_B0_stage1.pth \
  --set model.input_size=640 \
  --set train.batch_size=2 \
  --set train.accumulation=8 \
  --set cmt.query_chunk=400 \
  --set model.attention_chunk=4096 \
  --set model.checkpoint_encoder=false \
  --max-batches 32 \
  --stop-after-epochs 1
```

该命令是短试验，输出目录需为空；如果已经有结果请换新目录。它**不包含二维prod替换**，不能期待仅凭此命令获得0.84秒/batch。去掉限制参数进行正式训练前，应实施并验证P0代码优化。本报告没有启动这条训练命令。

如果显存余量较小，保留 `model.checkpoint_encoder=true`。当前不建议优先牺牲输入尺寸、P2、CMT支持域、query数或decoder层数；这些属于能力/结构取舍，应在执行瓶颈解决后单独消融。

## 9. 文件与复现

- 远程结果目录：`/data/coding/CMTDet-main/runs/bottleneck_audit_20260914/`。
- 本地结果目录：`F:/antiuav/TSCRNet/CMTDet-main/runs/bottleneck_audit_20260914/server/`。
- `original_run/`：原训练配置/日志/环境。
- `pretrained_t48.json` 等：固定batch分阶段原始结果及源码SHA256。
- `real_loop_original.json`、`real_loop_optimized.json`：逐micro-batch真实数据、优化器更新与AMP记录。
- `sync_parents.json`：CPU trace中标量同步的父调用链。
- `isolated_reader_pair.json`：reader隔离性能和输出/梯度差异。
- `validation_audit.json`、`val_*/`：16张val图的有界测量和输出。
- `audit_pretrained.py`、`audit_pair.py`、`real_loop.py`、`isolate_pair.py`：实际运行脚本。`audit.py`是最初随机骨干探路脚本，以对应JSON权重字段区分。
- trace压缩包另存，GPU kernel活动缺失的限制见3.3节。

复现单项测量示例（只运行独立诊断脚本）：

```bash
source /data/coding/setup/env_cmtdet.sh
cd /data/coding/CMTDet-main
python runs/bottleneck_audit_20260914/audit_pair.py \
  --name repeat_pair_combined --threads 4 --chunk 400 \
  --attention-chunk 4096 --pair-product
```

数据与源码未被替换；正式训练结果目录未被覆盖。本次短循环的优化器更新仅发生在独立诊断进程内，未保存为训练checkpoint。
