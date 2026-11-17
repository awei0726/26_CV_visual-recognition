# GRID Student 训练准备

本项目已生成 Student 的灰度嘴部序列和 Teacher 的逐时间步 CTC logits。三份实验清单按说话人划分，共有 13,887 条已配对样本。

| 划分 | 说话人 | 样本数 | 数据包 |
| --- | --- | ---: | --- |
| 训练 | `s1`–`s8` | 7,887 | `grid-train.tar.gz` |
| 验证 | `s9`–`s10` | 2,000 | `grid-val.tar.gz` |
| 测试 | `s11`–`s14` | 4,000 | `grid-test.tar.gz` |

## 取用数据

安装 Git LFS 后克隆仓库，在仓库根目录执行以下命令。三个压缩包包含各自清单引用的 ROI 和 Teacher logits，解压后保留清单中的相对路径。

```sh
git lfs pull
tar -xzf training-data/grid-train.tar.gz
tar -xzf training-data/grid-val.tar.gz
tar -xzf training-data/grid-test.tar.gz
```

```text
data/student_grid/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── roi/<speaker>/<stem>.npy
output/<speaker>/<stem>.pt
benchmarks/GRID/models/GRID_V_WER4.8/model.json
```

每行 JSONL 包含 `id`、`roi`、`text`、`teacher` 和 `length`。从仓库根目录解析 `roi` 和 `teacher` 路径。`roi` 是 `uint8` 灰度数组 `[T, 64, 64]`，读取后增加通道维并转成浮点张量。Teacher `.pt` 包含 CPU 上的 FP16 `logits` `[T, 41]`、`length` 和 `blank_id`。`length` 同时是 ROI 的帧数和 logits 的有效时间步数。

## 接入训练

1. 以 `benchmarks/GRID/models/GRID_V_WER4.8/model.json` 中的 `char_list` 为词表，沿用全部 41 类的顺序。空格映射为 `<space>`（ID `13`），CTC blank 为 `0`。转录目标只编码 `text` 中的字符。
2. 在 `Dataset` 中加载 ROI 和转录，蒸馏训练再读 Teacher 缓存。组批时沿时间轴补齐，保留 `input_lengths` 和 `target_lengths`。CNN + TCN 的时间维保持不变。
3. 在同一训练清单上先训练纯 CTC 基线，再训练 CTC + KD。蒸馏时将 Teacher logits 转成 FP32，根据有效帧掩码计算带温度的 KL 散度。用验证集选择配置，最终在测试集记录 WER 和 CER。

当前代码提供数据准备和评测模块。Student 模型、`Dataset`、训练入口及量化仍待实现。

## 重新生成输入

已有压缩包可直接用于训练数据加载。若要从原始 GRID 视频重新生成，按 `data/GRID/<speaker>/<stem>.mpg` 放置视频，并将 `GRID_V_WER4.8` 教师权重 `model.pth` 放到 `model.json` 同级。数据准备曾使用 Python 3.10、PyTorch 2.8、TorchVision 0.23、TorchAudio 2.8、PyAV 17.1 和 MediaPipe 0.10.14。重新裁剪时沿用 Teacher 的 `torchvision.io.read_video` 读帧路径，避免由于解码差异改变帧序列。

先将 `batch_extract_teacher_logits.py` 中的 `ROOT` 指向实际仓库根目录，再在根目录运行以下命令。已有的 Teacher 缓存和有效 ROI 会被复用。

```sh
python batch_extract_teacher_logits.py
python batch_prepare_student_data.py
```

数据来自 Martin Cooke、Jon Barker、Stuart Cunningham 和 Xu Shao 发布的 GRID 音视频语料库。ROI 经对齐、灰度化、中心裁剪与缩放处理。原始视频和 Teacher 权重可通过原项目获取，源码许可见 `LICENSE`。[GRID 语料库](https://zenodo.org/records/3625687) · [原项目与模型下载](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
