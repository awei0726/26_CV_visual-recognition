# GRID 同说话人划分

对现有 ROI 中的 `s1–s14` 分别以 70%／10%／20% 随机分配训练、验证和测试样本。使用种子 `42`，每位说话人的整数样本数按最大余数法分配。三份清单共覆盖 13,887 条样本，样本 ID 和完整转录均无跨集合重叠。

| 集合 | 清单 | 样本数 |
| --- | --- | ---: |
| 训练 | `train.jsonl` | 9,720 |
| 验证 | `val.jsonl` | 1,389 |
| 测试 | `test.jsonl` | 2,778 |

## 准备文件

从仓库根目录下载并解压原有三个 Git LFS 数据包。新清单仍从仓库根目录解析 ROI 和 Teacher logits 路径，因此每个新集合都可能引用三个包里的文件。

```sh
git lfs pull --include="training-data/*.tar.gz" --exclude=""
tar -xzf training-data/grid-train.tar.gz
tar -xzf training-data/grid-val.tar.gz
tar -xzf training-data/grid-test.tar.gz
```

从根目录运行 `python split_grid_within_speaker.py --seed 42` 可以重建清单。

## 训练与评估

使用 Python 3.11 或更新版本，并安装仓库根目录的 `requirements-student.txt`。`--split-protocol within-speaker` 自动读取本目录下的训练和验证清单，并检查它们与本目录测试集的样本是否独立。以下命令从头训练，分别保存纯 CTC 和标准 KD 的验证集最佳检查点。

```sh
python train_student.py --split-protocol within-speaker --mode ctc --epochs 30 --batch-size 16 --lr 0.0003 --seed 42 --output-dir output/student_within/ctc
python train_student.py --split-protocol within-speaker --mode kd --epochs 30 --batch-size 16 --lr 0.0003 --seed 42 --temperature 2 --kd-weight 0.5 --output-dir output/student_within/kd
python evaluate_student.py --checkpoint output/student_within/ctc/best.pt --manifest data/student_grid/within_speaker/test.jsonl --output-dir output/student_within/eval_ctc
python evaluate_student.py --checkpoint output/student_within/kd/best.pt --manifest data/student_grid/within_speaker/test.jsonl --output-dir output/student_within/eval_kd
```

与原跨说话人结果比较时，两种划分的训练样本数和说话人数也不同。若要单独检验同说话人带来的影响，还需控制训练数据量和模型设置。
