
import os
import time
import torch

from pipelines.pipeline import InferencePipeline


# ==========================================
# 1. 路径与参数配置
# ==========================================

ROOT = r"D:\Code\Visual_Speech_Recognition_for_Multiple_Languages"

INPUT_ROOT = os.path.join(ROOT, "data", "GRID")
OUTPUT_ROOT = os.path.join(ROOT, "output")

CONFIG = os.path.join(
    ROOT, "configs", "GRID_V_WER4.8.ini"
)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("Device:", DEVICE)


# ==========================================
# 2. 加载 Teacher（只加载一次）
# ==========================================

# 官方配置使用相对路径，因此设置工作目录
os.chdir(ROOT)

pipeline = InferencePipeline(
    config_filename=CONFIG,
    detector="mediapipe",
    face_track=True,
    device=DEVICE,
)

pipeline.eval()

net = pipeline.model.model
net.eval()

for param in net.parameters():
    param.requires_grad_(False)

print("Teacher loaded successfully!")
print("Vocabulary size:", pipeline.model.odim)


# ==========================================
# 3. 逐视频提取 Logits
# ==========================================

@torch.inference_mode()
def extract_logits(video_path):

    # 获取人脸关键点
    landmarks = pipeline.process_landmarks(
        video_path,
        None
    )

    # 官方视频预处理
    video = pipeline.dataloader.load_data(
        video_path,
        landmarks
    )

    if video.numel() == 0:
        raise ValueError("Empty video")

    # 移动到 GPU
    video = video.to(DEVICE)

    # Teacher Encoder
    features = net.encode(video)
    # [T, 256]

    # Teacher CTC Head
    logits = net.ctc.ctc_lo(
        features.unsqueeze(0)
    )
    # [1, T, 41]

    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain NaN or Inf")

    # 去掉 Batch 维度，转换为 FP16 并移回 CPU
    return logits.squeeze(0).half().cpu()


# ==========================================
# 4. 遍历 s1-s35，跳过 s21
# ==========================================

total_success = 0
total_failed = 0
total_skipped = 0

for speaker_id in range(1, 36):

    if speaker_id == 21:
        continue

    speaker = f"s{speaker_id}"

    input_dir = os.path.join(INPUT_ROOT, speaker)
    output_dir = os.path.join(OUTPUT_ROOT, speaker)

    if not os.path.isdir(input_dir):
        print(f"[SKIP] Folder not found: {input_dir}")
        continue

    os.makedirs(output_dir, exist_ok=True)

    # 获取当前 speaker 的所有视频
    video_files = sorted([
        filename
        for filename in os.listdir(input_dir)
        if filename.lower().endswith((".mpg", ".mpeg", ".mp4"))
        and os.path.isfile(os.path.join(input_dir, filename))
    ])

    print(f"\n===== {speaker}: {len(video_files)} videos =====")

    speaker_success = 0
    speaker_failed = 0
    speaker_skipped = 0

    for idx, filename in enumerate(video_files, start=1):

        video_path = os.path.join(input_dir, filename)

        video_name = os.path.splitext(filename)[0]

        save_path = os.path.join(
            output_dir,
            video_name + ".pt"
        )

        # 已处理文件直接跳过
        if os.path.exists(save_path):
            speaker_skipped += 1
            total_skipped += 1
            continue

        start = time.perf_counter()

        try:
            logits = extract_logits(video_path)

            torch.save(
                {
                    "logits": logits,
                    "length": logits.shape[0],
                    "blank_id": int(net.blank),
                },
                save_path
            )

            speaker_success += 1
            total_success += 1

            print(
                f"[{speaker}] "
                f"{idx}/{len(video_files)} "
                f"{filename} "
                f"-> {tuple(logits.shape)} "
                f"({time.perf_counter() - start:.2f}s)"
            )

        except Exception as e:
            speaker_failed += 1
            total_failed += 1

            print(
                f"[ERROR] {speaker}/{filename}: {e}"
            )

    print(
        f"{speaker} finished | "
        f"Success={speaker_success}, "
        f"Skipped={speaker_skipped}, "
        f"Failed={speaker_failed}"
    )


# ==========================================
# 5. 最终统计
# ==========================================

print("\n========== ALL FINISHED ==========")
print("Total success:", total_success)
print("Total skipped:", total_skipped)
print("Total failed :", total_failed)
print("Output path  :", OUTPUT_ROOT)
