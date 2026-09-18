# CMTDet训练性能优化：第二阶段实现说明

日期：2026-09-14。将[5060 Ti瓶颈复评](CMTDet_5060Ti训练瓶颈复评与实测报告.md)中已验证的方向落实为正式源码和配置。输入640×640，保留HGNetV2-B0、P2、完整CMT、300个检测query、两层encoder和六层decoder。

## 1. 实现范围

| 文件 | 修改 |
|---|---|
| `src/cmtdet/products.py` | 新增局部 `pair_product`，支持显式相乘和原生prod两种后端 |
| `src/cmtdet/moments.py` | 核函数、协方差交叉项、支持域面积的二维求积接入统一函数；两个reader均支持选择 |
| `src/cmtdet/geometry.py` | IoU/GIoU面积计算接入统一函数，原有两参数调用保持兼容 |
| `src/cmtdet/losses.py` | 匹配、GIoU、忽略区域面积使用配置指定的求积方式 |
| `src/cmtdet/config.py` | 增加 `cmt.pair_product_backend`，验证合法值，兼容历史checkpoint |
| `configs/cmtdet_hgnetv2_b0.yaml` | 更新分块与checkpoint设置，显式选择优化求积路径 |

没有增加或删除模型参数，state_dict键和形状不变。优化器、梯度累积、日志、评价指标、数据划分及forward模式沿用现有流程。

## 2. 数学与精度行为

固定二维求积为 `f(x,y)=xy`，梯度为 `(y,x)`。显式实现使用：

```python
x, y = value.unbind(-1)
return x * y
```

一项为零、两项为零或输入为负数时仍成立，不使用“乘积除以输入”求导，也不detach梯度。这样避免了通用prod反向的零计数和累计乘积路径。

**正式实现不修改 `torch.Tensor.prod` 或其他全局PyTorch方法。** 全局替换仅是之前独立进程里的归因原型。

函数只接受最后一维长度为2的实数浮点张量。CPU/CUDA开启autocast时，FP16/BF16输入按照原生prod策略提升到FP32；关闭autocast时保持输入dtype。其他尚未验证的设备回退原生prod。

## 3. 推荐配置

正式配置仍是 `configs/cmtdet_hgnetv2_b0.yaml`，关键变更：

| 参数 | 原值 | 新值 |
|---|---|---|
| `model.attention_chunk` | 1024 | 4096 |
| `model.checkpoint_encoder` | true | false |
| `cmt.query_chunk` | 32 | 400 |
| `cmt.pair_product_backend` | 原先使用原生prod | explicit |

batch=2、累积8步、640输入及完整模型结构保持原配置。此前5060 Ti连续真实数据原型的allocated峰值约9.35GiB、reserved约10.46GiB。关闭checkpoint是以显存换时间；如果后期支持范围变大导致显存吃紧，可用 `--set model.checkpoint_encoder=true`。

没有强制修改CPU线程数，因为此前48→4线程无明显收益。没有启用占位的CUDA注意力后端，没有更改验证频率或验证精度。400仍是可调整参数，不是所有数据集的硬性要求。

## 4. Linux运行方式

假设DUT已按原协议转换为 `data/dut_coco/data.yaml`，预训练权重已存在。从仓库根目录执行，先独立短测：

```bash
source /data/coding/setup/env_cmtdet.sh
cd /data/coding/CMTDet-main
python -m cmtdet train \
  --config configs/cmtdet_hgnetv2_b0.yaml \
  --data data/dut_coco/data.yaml \
  --pretrained-backbone weights/PPHGNetV2_B0_stage1.pth \
  --output runs/dut_hgnetv2_b0_perf2_smoke \
  --device cuda \
  --max-batches 32 \
  --stop-after-epochs 1
```

正式训练使用另一个空目录，去掉短测限制：

```bash
python -m cmtdet train \
  --config configs/cmtdet_hgnetv2_b0.yaml \
  --data data/dut_coco/data.yaml \
  --pretrained-backbone weights/PPHGNetV2_B0_stage1.pth \
  --output runs/dut_hgnetv2_b0_perf2_seed42 \
  --device cuda
```

本次仅修改本地项目，没有更新服务器源码、启动长训练或推送远程。将新版源码同步到服务器后再执行上述命令。

## 5. 恢复训练与兼容性

- 新YAML默认使用explicit，解析后的设置会写入日志和checkpoint。
- 历史checkpoint缺少 `cmt.pair_product_backend` 时，`from_dict`解释为prod，保留历史求积路径。旧分块和checkpoint开关同样来自保存配置，不会被新版YAML覆盖。
- 新checkpoint含此字段，恢复时保持保存的选择。
- 模型权重键和形状兼容，不代表浮点轨迹逐位一致。分块变化可能改变归约顺序。

恢复新版训练的命令：

```bash
python -m cmtdet train \
  --resume runs/dut_hgnetv2_b0_perf2_seed42/last.pt \
  --data data/dut_coco/data.yaml \
  --output runs/dut_hgnetv2_b0_perf2_seed42 \
  --device cuda
```

当前CLI禁止resume与config/set/pretrained-backbone混用。本次没有新增checkpoint迁移功能，恢复旧checkpoint不会自动获得整套新优化。此前服务器任务在首轮完成前中断，可以用新配置从原骨干预训练重新开始。

## 6. 消融与回退

| 目的 | 参数 |
|---|---|
| 只关闭显式二维相乘 | `--set cmt.pair_product_backend=prod` |
| 切换半径读取参考后端 | `--set cmt.reader_backend=reference` |
| 恢复旧的主要执行设置 | prod、query_chunk=32、attention_chunk=1024、checkpoint_encoder=true |
| 节省显存 | 优先开启encoder checkpoint，保留显式相乘 |

reader_backend决定半径读取/分桶方法，pair_product_backend决定二维求积方法，二者独立。性能A/B应固定权重、图像、输入、batch、累积、AMP及优化器状态，记录完整micro-batch、前后向和显存。关闭CMT的baseline模式仅供定位开销，不代表完整CMT优化。

## 7. 验证与限制

本地执行：

```text
.venv/Scripts/python.exe -m pytest -q --basetemp runs/pytest_perf_v2
60 passed in 14.63s
```

新增22项检查，加上原有38项，覆盖：

- CPU/CUDA、FP16/BF16/FP32/FP64、autocast开关的输出dtype、数值和梯度。
- 零值、负值、小数、一阶及二阶梯度检查。
- 两个reader、不同分块、空证据、边界及出画候选的输出和梯度。
- 完整模型两种求积方式的loss/参数梯度对照，包含忽略框。
- 历史配置恢复、strict权重加载、新字段验证。
- 原有多forward模式、训练与恢复一致性、CUDA AMP更新、COCO评价等回归。

此前报告中的 **2.760→0.838秒/batch、3.29倍加速** 来自服务器的独立原型。本次正式源码还没有重新部署到5060 Ti进行同条件速度复测，不能把原型数字视为新版源码再次测量。没有长训AP或相位稳健性结论；正式实验仍需统一评价口径验证精度，并观察后期速度与显存。
