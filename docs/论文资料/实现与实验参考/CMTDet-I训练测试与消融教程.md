# CMTDet-I训练、测试与消融教程

本教程对应`miattn`分支，主配方为`configs/cmtdet_i_r50_640.yaml`，输入640×640。结构见[实现说明](miattn分支CMTDet-I结构与实现.md)。`cmtdet_i_smoke.yaml`只用于64输入的快速工程检查，不是论文配方。

## 1. 环境与入口

Linux服务器已有PyTorch时，先激活对应环境，确认安装的CUDA构建支持显卡：

```bash
cd /data/coding/CMTDet-main
git fetch origin
git switch miattn
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pip install -r scripts/requirements.txt
python scripts/run_cmtdet.py --help
```

PyTorch要求至少2.4，Python至少3.10。GPU对应的PyTorch构建应自行先安装；不要因本教程覆盖一个已能正常使用显卡的版本。本实现的ResNet不导入torchvision，避免不相关的torchvision二进制算子兼容问题。

仓库采用文件跟踪白名单，根目录`pyproject.toml`可能没有随Git下载。因此统一使用`python scripts/run_cmtdet.py`入口，它自动加载`src`，不要求`pip install -e .`。若希望`python -m cmtdet`，Linux先执行`export PYTHONPATH="$PWD/src"`；PowerShell 7使用`$env:PYTHONPATH = "$PWD/src"`。

Windows请在PowerShell 7（`pwsh.exe`）中操作：

```powershell
Set-Location F:\antiuav\TSCRNet\CMTDet-main
python scripts/run_cmtdet.py --help
```

## 2. ResNet-50预训练

正式配方冻结骨干BN，必须提供预训练骨干。支持Torchvision标准ResNet-50 V1.5格式，严格检查参数键和形状，排除分类fc。官方说明见[ResNet-50文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)。

可用官方V2权重；以下命令只下载到本仓库weights，不运行训练：

```bash
python -c "import torch; torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-11ad3fa6.pth', model_dir='weights', check_hash=True)"
```

后续传入`--pretrained-backbone weights/resnet50-11ad3fa6.pth`。若只是检查随机初始化的流水线，允许`--set model.resnet_freeze_norm=false`并省略预训练；不要把这种短测当成精度实验。

## 3. COCO数据与DUT转换

COCO是默认接口。单类或多类均可，COCO category id允许不连续，模型内部按排序映射；`model.classes`必须与categories数量一致。图像包括负样本，crowd/ignore不会被当作普通正样本。

```yaml
root: /data/coding
train:
  images: dataset/DUT/train/img
  annotations: CMTDet-main/data/dut_coco/train.json
val:
  images: dataset/DUT/val/img
  annotations: CMTDet-main/data/dut_coco/val.json
test:
  images: dataset/DUT/test/img
  annotations: CMTDet-main/data/dut_coco/test.json
```

root为相对路径时，相对于data.yaml所在目录解析；images/annotations也可使用绝对路径。迁移机器后检查路径，不复用旧Windows绝对路径。

DUT原始结构需有`train/val/test`，每个split下为`img`与`xml`：

```bash
python scripts/run_cmtdet.py convert-dut \
  --source /data/coding/dataset/DUT \
  --output data/dut_coco \
  --coordinates inclusive
```

Windows等价命令：

```powershell
python scripts/run_cmtdet.py convert-dut --source F:\antiuav\dataset\DUT --output data/dut_coco --coordinates inclusive
```

只生成COCO JSON和data.yaml，不复制/修改原始图片，不重划分原有split。`inclusive`沿用本项目的零起点闭区间DUT解释；若手中的标注来自另一转换，核对`half_open`或`voc1_inclusive`，不要混用口径。转换设置写入`conversion.json`。

输出目录不得已有data.yaml。`--limit-images 4`可建立独立小型检查集；它会被标记为subset，不能用于完整结果。

## 4. 先做短测

在正式数据配置上，只读少量图像进行640输入的短测：

```bash
python scripts/run_cmtdet.py train \
  --config configs/cmtdet_i_r50_640.yaml \
  --data data/dut_coco/data.yaml \
  --pretrained-backbone weights/resnet50-11ad3fa6.pth \
  --output runs/dut_i_smoke \
  --device cuda \
  --limit-images 4 --max-batches 2 --stop-after-epochs 1 \
  --set train.workers=0 --set train.batch_size=1 \
  --set eval.batch_size=1 --set train.full_val_interval=0
```

检查进度条、loss有限、AMP是否持续跳步、last.pt/best.pt和val结果是否保存。首次warmup较慢；不要用第一步耗时估算总训练时间。

这是部分训练，不能通过恢复它来变成完整训练：恢复时样本限制、batch限制和数据哈希必须一致。正式训练用新输出目录。

## 5. 正式训练

```bash
python scripts/run_cmtdet.py train \
  --config configs/cmtdet_i_r50_640.yaml \
  --data data/dut_coco/data.yaml \
  --pretrained-backbone weights/resnet50-11ad3fa6.pth \
  --output runs/dut_i_r50_seed42 \
  --device cuda \
  --set train.seed=42
```

主配方起点：

| 参数 | 默认 | 说明 |
|---|---:|---|
| 输入 | 640 | 等比例resize，右侧/底部补114，RGB/ImageNet归一化 |
| encoder/decoder | 1/4 | decoder每层有矩积分 |
| 普通query | 300 | 150语义+150矩 |
| epochs | 100 | 初始研究预算，不保证是最优收敛轮数 |
| batch/accumulation | 2/8 | 单设备有效batch16；不要只改batch而忘记累积 |
| LR / backbone LR | 2e-4 / 2e-5 | AdamW，weight decay=1e-4 |
| warmup | 1轮 | epoch级调度，之后cosine |
| AMP | 开 | 关键证据和矩链保持FP32 |
| workers | 4 | Windows启动慢可先用0定位；prefetch默认2 |
| val_interval | 1 | 每轮val |
| full_val_interval | 5 | 中间轮省去大型查询/匹配日志；最后一轮完整val |
| patience | 20 | 连续20次有效验证未提升早停；0关闭 |
| monitor | AP50_95 | 可设为输入tiny AP，但无tiny GT时不累计patience |
| logging.interval | 1 | 保存每batch日志；设置10/50可减少日志同步开销 |

light val仍评估**整个val集和同一套指标**，只是减少大型产物；默认light/full都启用AMP，保持数值设置一致。`full_val_interval=0`表示每次均完整保存。

自定义参数通过重复`--set section.key=value`；未知键会报错。例如：

```bash
python scripts/run_cmtdet.py train \
  --config configs/cmtdet_i_r50_640.yaml --data data/dut_coco/data.yaml \
  --pretrained-backbone weights/resnet50-11ad3fa6.pth \
  --output runs/dut_i_r50_batch4 --device cuda \
  --set train.batch_size=4 --set train.accumulation=4 \
  --set train.val_interval=2 --set train.patience=10 \
  --set logging.interval=10
```

显存能否容纳batch4必须在目标显卡上短测，不依据本机batch1直接保证。冻结BN与梯度累积也不能让所有随机性与大物理batch完全等价。

## 6. 断点恢复

```bash
python scripts/run_cmtdet.py train \
  --resume runs/dut_i_r50_seed42/last.pt \
  --data data/dut_coco/data.yaml \
  --output runs/dut_i_r50_seed42 --device cuda
```

恢复会读取模型、optimizer、scheduler、GradScaler、完成轮数、早停状态和随机状态；不要再传config、set或pretrained。完整训练日志继续追加。支持从最后一个完整epoch恢复；没有实现任意batch精确恢复。

CPU、workers=0的确定性测试检查了“连续两轮”和“训一轮再恢复”权重逐项完全一致。多worker预取和非确定性CUDA核下，不承诺逐bit恢复相同的数据增强轨迹。

改变结构、训练轮数/配方或数据集属于新实验，应建立新配置和新输出目录，不修改checkpoint冒充原实验延续。

## 7. 独立测试

```bash
python scripts/run_cmtdet.py test \
  --weights runs/dut_i_r50_seed42/best.pt \
  --data data/dut_coco/data.yaml --split test \
  --output runs/dut_i_r50_seed42_test --device cuda \
  --set eval.batch_size=2 --set train.workers=4
```

`--split val`可独立复算验证集。测试从checkpoint恢复结构，类别映射严格检查。测试输出必须是新目录。只有排查时使用`--limit-images`/`--max-batches`；报告会标识partial。

指标：

- 标准COCO AP/AP50/AP75、APsmall/medium/large、AR1/10/100及尺度AR。
- `AP50_95_input_bbox_area_lt100`：预处理后框面积<100的AP。
- `AR50_95_input_bbox_area_lt100_maxDets100`：同一组的平均召回。
- `GT_input_bbox_area_lt100`：该组非ignore实例数量。

它们以0–1保存，无有效GT为−1。指标不是PFR；本实现不自动运行独立采样敏感性实验。完成训练后，可把checkpoint/模型适配器接入原采样协议做研究验证。

## 8. 日志与产物

| 文件 | 内容 |
|---|---|
| config.json | 合并默认项及覆盖后的完整配置 |
| environment.json | PyTorch、设备、参数量、数据/源代码哈希、类别映射 |
| batches.jsonl | 采样间隔内每batch损失、梯度范数、LR、AMP scale、优化步数 |
| epochs.jsonl / history.csv | 每轮损失、val、早停计数、分阶段耗时；CSV的tiny列使用输入尺度 |
| train.log / status.json | 运行消息、异常栈、结束原因 |
| last.pt / best.pt | 完整训练状态；best按monitor提升选择 |
| resume_events.jsonl | 每次恢复的来源和环境 |
| val/epoch_NNNN | 分轮验证结果 |
| metrics.json / report.md | 数值和中文评估报告 |
| predictions.json | 原图坐标预测，用于标准COCO评估 |
| tiny_input_annotations.json / tiny_input_predictions.json | 实际输入尺度的附加评估数据 |
| images.json / metadata.json | 每图缩放/有效尺寸、完整配置和统计口径 |
| *_curves.npz / *_matches.jsonl | PR/召回张量、逐图匹配；light省去匹配文件 |
| raw_queries.jsonl | 全部query框与类别logit；完整评估可保存 |
| checkpoint.json | 独立测试使用的权重哈希、epoch和split |

异常时last.pt仍指最后一个保存完成的epoch；不是失败batch的即时状态。checkpoint采用临时文件再替换。

## 9. 消融实验

一次生成多个变体及随机种子配置，默认**只生成、不训练**：

```bash
python scripts/run_miattn_ablations.py \
  --config configs/cmtdet_i_r50_640.yaml --data data/dut_coco/data.yaml \
  --pretrained-backbone weights/resnet50-11ad3fa6.pth \
  --output runs/dut_i_ablation_plan \
  --variants baseline evidence_aux feature_only readout_only full \
  --seeds 42 43 44
```

指定新的输出目录并加`--execute`会顺序训练所有变体，耗时可能很长。完整参数与命令数组保存到生成目录的configs/和commands.json。脚本没有默默派发后台训练。

建议分组：

| 问题 | 变体 |
|---|---|
| 局部内容/几何是否有效 | baseline、evidence_aux、feature_only、readout_only、full |
| 格内位置和二阶是否有效 | mass_only、first_order、full |
| 同权读取是否有效 | no_content_weights、independent_weights、full |
| 几何进入内容的作用 | no_geometry_content、full |
| 候选召回贡献 | semantic_queries、moment_queries、full；no_proposal_loss |
| 多轮闭环/多支持 | last_layer、single_support、decoder2、decoder6、full |
| 监督和门控 | no_heat_loss、no_dn、no_gate、no_residual |
| 压缩粒度 | stride1、stride2、full(stride4)、stride8 |

先让路径对照统一`--set mia.moment_queries=0.0`以隔离积分算子，再测候选贡献。结构消融必须重新训练；只在test切换`cmt.mode`是冻结模型依赖诊断，不能替代公平消融。

所有变体使用相同数据划分、输入、预训练、训练预算和验证规则。stride1/2增加信息与计算，不应当作不增加代价的等价模型。参数、FLOPs与耗时要逐配置记录。

## 10. 训练前做性能检查

```bash
python scripts/profile_train_step.py \
  --config configs/cmtdet_i_r50_640.yaml \
  --pretrained-backbone weights/resnet50-11ad3fa6.pth \
  --output runs/profile_i_batch2.json \
  --device cuda --batch 2 --warmup 5 --steps 20 --amp
```

记录forward、loss、backward、optimizer、峰值显存和AMP跳步。它是合成训练步，不含真实数据读取，也不能保证后期窗口变大时成本不变。可加`--data data/dut_coco/data.yaml --val-batches 2`做限定验证检查。

显存不足先开启`--set mia.checkpoint_readout=true`，再降低物理batch并调整accumulation。显存充足时优先维持关闭checkpoint，并试增大query_chunk/cell_budget减少小块启动；这些都是资源参数，先检查数值与峰值。

证据链默认`mia.evidence_fp32=true`。不要仅为速度关闭它：归一化读出对近空证据的导数可能很大，FP16证据head可能溢出。若AMP持续跳步，先检查日志和梯度，而不是把跳步当作有效优化。

```bash
python -m pytest tests -q --basetemp=.test-tmp-miattn
```

测试入口也能在没有根pyproject.toml的干净克隆中加载src。具体已验证范围见[验证记录](CMTDet-I实现验证记录.md)。
