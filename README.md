# GRID Student 训练准备

本项目已生成 Student 的灰度嘴部序列和 Teacher 的逐时间步 CTC logits。三份实验清单按说话人划分，共有 13,887 条已配对样本。

| 划分 | 说话人 | 样本数 | 数据包 |
| --- | --- | ---: | --- |
| 训练 | `s1`–`s8` | 7,887 | `grid-train.tar.gz` |
| 验证 | `s9`–`s10` | 2,000 | `grid-val.tar.gz` |
| 测试 | `s11`–`s14` | 4,000 | `grid-test.tar.gz` |

## 取用数据

安装 Git LFS 后，在仓库根目录下载并解压中期所需的训练和验证包。压缩包包含各自清单引用的 ROI 和 Teacher logits，解压后保留清单中的相对路径。

```sh
git lfs pull --include="training-data/grid-train.tar.gz,training-data/grid-val.tar.gz" --exclude=""
tar -xzf training-data/grid-train.tar.gz
tar -xzf training-data/grid-val.tar.gz
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
3. 在同一训练清单上先训练纯 CTC 基线，再训练 CTC + KD。蒸馏时将 Teacher logits 转成 FP32，根据有效帧掩码计算带温度的 KL 散度。中期在相同验证集上比较 WER 和 CER。

### 运行中期对照

在 Python 3.11 环境中安装依赖。GPU 训练时先按显卡选择对应的 PyTorch 安装包，再安装 `requirements-student.txt`。

```sh
python -m pip install -r requirements-student.txt
```

在仓库根目录运行以下命令。两组使用相同模型、数据、随机种子和训练参数，训练轮数可按小组预算统一调整。标准 KD 组额外使用预计算的 Teacher logits，温度为 2，蒸馏权重为 0.5。

```sh
python train_student.py --mode ctc --epochs 60 --batch-size 8 --seed 42 --output-dir output/student_midterm/ctc
python train_student.py --mode kd --epochs 60 --batch-size 8 --seed 42 --temperature 2 --kd-weight 0.5 --output-dir output/student_midterm/kd
```

每轮自动验证，并记录损失、WER、CER 到 `history.csv`。`best.pt` 依次按验证 WER、CER 和 CTC 损失选择，`last.pt` 用于续训。续训时沿用该组的原命令，追加 `--resume <该组输出目录>/last.pt`，`--epochs` 表示目标总轮数。

换电脑续训时复制完整的训练目录，并准备同一版清单、ROI 和 Teacher 缓存。清单内容相同时，Windows 和 WSL 的 CRLF/LF 换行差异可自动兼容。

### 复核验证结果

评估时保留各组输出目录中的 `best.pt`、`config.json`、`history.csv`、`summary.json` 和 `status.json`。为每条评估命令使用新的输出目录。

```sh
python evaluate_student.py --checkpoint output/student_midterm/ctc/best.pt --manifest data/student_grid/val.jsonl --output-dir output/student_midterm/eval_ctc
python evaluate_student.py --checkpoint output/student_midterm/kd/best.pt --manifest data/student_grid/val.jsonl --output-dir output/student_midterm/eval_kd
```

`metrics.json` 中的 WER 和 CER 为比率，`predictions.jsonl` 保存逐样本结果。找到与 checkpoint 指纹匹配的训练清单后，评估会记录与该清单重叠的样本数，存在重叠时结果仅作诊断。报告结果时说明实验基于当前 Student 划分和给定 Teacher。仓库所附的 Teacher 训练参考清单包含 Student 验证集全部 2,000 条样本和测试集全部 4,000 条样本。

## 后续蒸馏实验

在相同的 Student 模型、数据划分、随机种子和训练预算下比较四组。各组从同一初始化规则开始，独立保存 checkpoint。

| 组别 | 模式 | 训练目标 |
| --- | --- | --- |
| E1 | `ctc` | CTC 监督 |
| E2 | `kd` | CTC 加均匀的有效帧蒸馏 |
| E3 | `kd-entropy` | CTC 加按 Teacher 预测熵调整权重的蒸馏 |
| E4 | `kd-blank` | CTC 加同时考虑预测熵和 Blank 概率的蒸馏 |

两种自适应模式根据未加温的 Teacher 概率计算帧权重，`--kd-epsilon` 默认 0.1。只对有效帧归一化，使平均权重为 1。蒸馏项仍使用相同温度和温度平方缩放。

自适应 KD 会检查完整的 Teacher 词表顺序，支持仓库的词表或内容相同的副本。比较 E2–E4 时，核对独立评估记录里的 `training_data_fingerprint`，它包含训练 ROI 和 Teacher 缓存内容。与纯 CTC 对照时，核对 `training_roi_fingerprint`。

```sh
python train_student.py --mode kd-entropy --epochs 60 --batch-size 8 --seed 42 --temperature 2 --kd-weight 0.5 --kd-epsilon 0.1 --output-dir output/student_final/entropy
python train_student.py --mode kd-blank --epochs 60 --batch-size 8 --seed 42 --temperature 2 --kd-weight 0.5 --kd-epsilon 0.1 --output-dir output/student_final/blank
```

数据规模消融给两条对照命令同时加 `--train-size 1000 --subset-seed 42`，或同时改用 4,000 条。采样基于样本 ID 和相同的子集种子，小子集包含在大子集里。现有全量训练清单有 7,887 条，跑全量时省略 `--train-size`。每种规模使用独立输出目录。`--limit-train` 和 `--limit-val` 用于流程试跑。

## ONNX、INT8 与 CPU 对照

在 Student 环境内安装部署依赖。导出脚本接受已有 Student checkpoint，运行时只需要 Student 模型。默认使用训练清单中固定种子选出的 128 个样本做静态 QDQ 校准。

```sh
python -m pip install -r requirements-deployment.txt
python export_student.py --checkpoint output/student_final/blank/best.pt --output-dir output/student_final/blank_onnx
```

导出目录包含 `student_fp32.onnx`、`student_int8.onnx` 和 `export_report.json`。报告记录 checkpoint、校准样本、量化节点和导出自检。给 `export_student.py` 加 `--fp32-only` 可单独导出 FP32。

用同一份评估清单比较两种精度。验证集用于调试和选择配置，定稿后用测试清单做最终比较。

CPU 对照和预裁剪 ROI 推理只需要 NumPy、ONNX Runtime 和测量依赖，可在单独的 Python 3.11 环境中安装 `requirements-runtime.txt`。

```sh
python benchmark_student.py --fp32 output/student_final/blank_onnx/student_fp32.onnx --int8 output/student_final/blank_onnx/student_int8.onnx --manifest data/student_grid/val.jsonl --output-dir output/student_final/blank_benchmark --threads 1
```

`metrics.json` 记录 WER/CER、模型大小、CPU 延迟和峰值内存。`predictions.jsonl` 按样本保存两种精度的预测。CPU 延迟统计单视频的 ONNX Runtime 调用，预处理和解码在计时之外。峰值常驻内存来自各自的 NumPy/ONNX Runtime 独立进程，并记录加载模型前的基线。比较时使用同一台机器、同一版本、相同线程数和清单。

CPU 对照会检查清单是否包含 GRID 训练样本，并在推理前后校验清单和 ROI 文件。完整的仓库验证集或测试集会标为对应划分，自定义清单和受限样本试跑的结果标为诊断。

正式的 CPU 对照还会核验 ONNX 记录的 checkpoint 训练清单指纹。旧导出和自定义训练仍支持 CPU 试跑，会标为诊断。旧 ONNX 如需用于正式对照，使用对应 checkpoint 重新导出一对模型。

最终测试时从仓库根目录取得测试包，再把评估命令的 `--manifest` 指向 `data/student_grid/test.jsonl`，使用新的输出目录。

```sh
git lfs pull --include="training-data/grid-test.tar.gz" --exclude=""
tar -xzf training-data/grid-test.tar.gz
```

## 单视频演示

准备好的灰度 ROI 可以直接输入 ONNX Student。

```sh
python predict_student.py --model output/student_final/blank_onnx/student_int8.onnx --roi data/student_grid/roi/s9/bbae4n.npy
```

处理原始视频时，在独立的 Python 3.11 环境安装 `requirements-video.txt`。演示沿用项目的 MediaPipe 人脸对齐和灰度口型裁剪，适用于单人、正脸、接近 GRID 采集条件的短片。

```sh
python -m pip install -r requirements-video.txt
python predict_student.py --model output/student_final/blank_onnx/student_int8.onnx --video path/to/clip.mpg
```

## 重新生成输入

已有压缩包可直接用于训练数据加载。若要从原始 GRID 视频重新生成，按 `data/GRID/<speaker>/<stem>.mpg` 放置视频，并将 `GRID_V_WER4.8` 教师权重 `model.pth` 放到 `model.json` 同级。数据准备曾使用 Python 3.10、PyTorch 2.8、TorchVision 0.23、TorchAudio 2.8、PyAV 17.1 和 MediaPipe 0.10.14。重新裁剪时沿用 Teacher 的 `torchvision.io.read_video` 读帧路径，避免由于解码差异改变帧序列。

先将 `batch_extract_teacher_logits.py` 中的 `ROOT` 指向实际仓库根目录，再在根目录运行以下命令。已有的 Teacher 缓存和有效 ROI 会被复用。

```sh
python batch_extract_teacher_logits.py
python batch_prepare_student_data.py
```

数据来自 Martin Cooke、Jon Barker、Stuart Cunningham 和 Xu Shao 发布的 GRID 音视频语料库。ROI 经对齐、灰度化、中心裁剪与缩放处理。原始视频和 Teacher 权重可通过原项目获取，源码许可见 `LICENSE`。[GRID 语料库](https://zenodo.org/records/3625687) · [原项目与模型下载](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
