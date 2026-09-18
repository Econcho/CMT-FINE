# CMTDet-I实现验证记录

日期：2026-09-15。对应`miattn`分支；验证范围是正确性、训练/评估闭环和有限性能短测，没有完成模型收敛实验。

## 1. 测试环境

- Windows、PowerShell 7。
- Python 3.11.16、PyTorch 2.7.1+cu128。
- 本地GPU：NVIDIA GeForce RTX 3060 Laptop GPU。**不是5060 Ti**，耗时不能直接外推到服务器。
- 使用项目已有`.venv`，它复用bcrnet的PyTorch，并提供SciPy/COCO等依赖；未修改系统Python。

## 2. 自动化检查

全套测试命令：

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests -q --basetemp=.test-tmp-miattn-release
```

结果：**84 passed**。包括旧架构回归，以及：

- CMT同权积分与独立逐像素展开的质量、坐标矩、外观均值一致，并比较梯度。
- bucket/full-grid在stride1/2/4/8及同权/独立权下输出、反向梯度一致。
- 六个forward mode均可计算检测损失并反向传播。
- 空证据、负样本、padding边缘、单元一阶退化、checkpoint开关和CUDA AMP。
- 24种消融配方逐一构建、前向、损失与反向，检查有限梯度。
- CPU两轮连续训练与一轮后恢复的最终权重逐项完全一致。
- 完整checkpoint独立评估、早停与val_interval旧功能回归。
- 输入面积<100的包含/排除边界、上采样、crowd及高分误报处理。

容差比较不表示浮点结果逐bit相同；CPU恢复测试在workers=0、固定种子条件下使用严格一致检查。

另外从Git暂存区导出一个只含待提交文件的副本（没有本地pyproject.toml、数据和权重），清空PYTHONPATH后执行CLI帮助和全套测试，结果同样为**84 passed**。它使用已安装依赖的解释器，验证远程源码不依赖本地未跟踪的安装入口；不是对所有Linux/CUDA版本的兼容性承诺。

## 3. DUT真实图片闭环

使用`F:\antiuav\dataset\DUT`，每个train/val/test各取原排序前4张，转换到本地`data/dut_miattn_smoke`。没有修改原始图像或重划分数据。

执行R50主模型、640输入、随机初始化、BN不冻结、batch1、AMP，先训一轮保存，再从last恢复第二轮，最后加载best对4张test图像独立测试。

最终运行目录：

- `runs/miattn_validation/dut_r50_fp32e`：8个训练batch、两轮验证、last/best、恢复事件、每batch与每epoch日志。
- `runs/miattn_validation/dut_r50_test`：独立测试报告、预测、COCO匹配/曲线和输入尺度tiny数据。

| 检查项 | 结果 |
|---|---|
| 两轮完成状态 | all_epochs_completed，epoch=2 |
| 实际AMP跳步 | 每轮0，共8步完成更新 |
| 每batch梯度范数/日志 | 有限，完整记录 |
| best/last保存与重载 | 通过 |
| 标准COCO与输入tiny报告 | 均生成 |
| 本次val/test子集输入tiny GT数 | 0，tiny AP/AR正确返回−1 |

这组4张的tiny数量为0，不能检验真实tiny性能；非空tiny与100面积边界由构造数据的COCO评估测试验证。随机初始化两轮的AP无研究意义，本记录不把它解释为检测能力。

## 4. 修复了实际暴露的AMP问题

初次短测让证据网络跟随AMP，仅把矩编码/读出转FP32；出现连续跳步，最初4个训练batch中3次更新被GradScaler跳过。近空证据的归一化梯度可能很大，在回传到FP16证据头时溢出，仅在末端做FP32求和不足以保护该链。

改为默认整条证据body、证据head和外观head使用FP32，保持其他语义路径AMP。修复后的同一DUT小集完成8步，scale保持128，没有跳步。回归测试还检查AMP环境下证据输出dtype为FP32。

原始失败运行保留在本地`runs/miattn_validation/dut_r50`用于追溯，不作为最终通过结果。上述对照支持该数值修复，但不承诺所有后续数据/参数组合都不会触发GradScaler调整。

## 5. 最终版本的性能短测

同一随机种子、R50、640输入、batch1，300普通query、训练DN预算100；合成单GT图像。warmup2步，测量3步。不是完整数据加载/训练epoch基准。

| 读出checkpoint | 单步均值 | forward | backward | 峰值allocated | 峰值reserved | AMP跳步 |
|---|---:|---:|---:|---:|---:|---:|
| 关闭，正式配方默认 | 582.22 ms | 206.45 ms | 327.65 ms | 3360.13 MiB（3.28 GiB） | 3438 MiB | 0 |
| 开启，低显存选项 | 764.48 ms | 271.92 ms | 446.05 ms | 1792.11 MiB（1.75 GiB） | 2012 MiB | 0 |

单步包括数据传输、forward、loss、backward和optimizer；表中未逐列列出全部项目。两种配置均保留FP32证据、证据checkpoint和encoder checkpoint。关闭读出checkpoint约减少24%的单步时间，但显著增加激活显存；不同batch、GT数、窗口分布和设备会改变结论。

前向计数：**35,040,809参数；约108.95 GFLOPs**。这是推理路径、batch1，乘加计2；Torch FlopCounterMode没有计入部分elementwise、grid_sample、gather、reduction，不能视为完整算力成本。

随机初始化推理的两次model-only测量为137.51/149.55ms，不包含图像解码和COCO评估，也不代表优化后的实际部署速度。

可复查的原始记录：

- [默认训练步](assets/miattn/final_r50_640.json)
- [开启读出checkpoint](assets/miattn/final_r50_640_checkpoint.json)
- [前向算子与FLOPs记录](assets/miattn/final_inference_cost.json)

这些最终记录替代实现中途的FP16证据试验数据。数值运算代码在测量后只补充了架构入口校验/公开导出；记录中的源代码哈希对应测量时版本。

## 6. 尚需完成的研究验证

完整DUT/DetFly收敛、多随机种子、强基线、独立消融、相位敏感性实验，以及目标5060 Ti上的正式batch/吞吐基准。当前产物证明了模型可运行及声明的数值接口得到检查，没有证明CMT的相位收益或SOTA。
