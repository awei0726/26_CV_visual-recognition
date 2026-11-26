import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPLITS = {"train": 70, "val": 10, "test": 20}


def split_counts(total):
    counts = {split: total * percentage // 100 for split, percentage in SPLITS.items()}
    remainders = sorted(SPLITS, key=lambda split: -(total * SPLITS[split] % 100))
    for split in remainders[:total - sum(counts.values())]:
        counts[split] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Split each GRID speaker into 70/10/20 sets.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = ROOT / "data" / "student_grid"
    by_speaker = defaultdict(list)
    seen = set()
    for split in SPLITS:
        with (source / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record["id"]
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample ID: {sample_id}")
                seen.add(sample_id)
                speaker, _ = sample_id.split("/", maxsplit=1)
                by_speaker[speaker].append(record)

    randomizer = random.Random(args.seed)
    output = {split: [] for split in SPLITS}
    for speaker in sorted(by_speaker, key=lambda value: int(value[1:])):
        records = sorted(by_speaker[speaker], key=lambda row: row["id"])
        randomizer.shuffle(records)
        counts = split_counts(len(records))
        start = 0
        for split, count in counts.items():
            output[split].extend(records[start:start + count])
            start += count
        print(speaker, counts)

    destination = source / "within_speaker"
    destination.mkdir(exist_ok=True)
    for split, records in output.items():
        target = destination / f"{split}.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(target)
        print(f"{split}: {len(records)} samples")


if __name__ == "__main__":
    main()
