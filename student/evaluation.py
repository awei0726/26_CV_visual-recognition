"""CTC greedy decoding and corpus-level evaluation for the lightweight Student."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile

import torch

from .losses import ctc_loss


def edit_distance(hypothesis, reference):
    """Return Levenshtein edits for either character or word sequences."""
    previous = list(range(len(reference) + 1))
    for index, token in enumerate(hypothesis, start=1):
        current = [index]
        for offset, expected in enumerate(reference, start=1):
            current.append(min(
                previous[offset] + 1,
                current[offset - 1] + 1,
                previous[offset - 1] + (token != expected),
            ))
        previous = current
    return previous[-1]


def greedy_decode(logits, input_lengths, vocabulary):
    """Collapse adjacent repeats, then remove blank, using valid frames only.

    A blank separates repeated characters: ``A, blank, A`` decodes to ``AA``.
    This order is essential for a correct CTC decoder.
    """
    if (logits.ndim != 3 or logits.shape[-1] != vocabulary.size
            or logits.shape[0] == 0 or logits.shape[1] == 0 or not logits.is_floating_point()):
        raise ValueError("Expected logits [B, T, vocabulary.size].")
    lengths = torch.as_tensor(input_lengths, device="cpu")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_lengths must contain integers; fractional lengths are invalid.")
    if lengths.ndim != 1 or lengths.numel() != logits.shape[0]:
        raise ValueError("input_lengths must contain one length per sample.")
    if (lengths <= 0).any() or (lengths > logits.shape[1]).any():
        raise ValueError("input_lengths must lie between 1 and the logits time dimension.")
    valid = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths.to(logits.device)[:, None]
    if not torch.isfinite(logits[valid]).all():
        raise ValueError("Student logits contain NaN or Inf.")

    paths = logits.detach().argmax(dim=-1).cpu()
    predictions = []
    for path, length in zip(paths, lengths.tolist()):
        collapsed = torch.unique_consecutive(path[:length]).tolist()
        tokens = [token for token in collapsed if token != vocabulary.blank_id]
        predictions.append(vocabulary.decode(tokens))
    return predictions


@torch.inference_mode()
def evaluate(model, loader, device, vocabulary, predictions_path=None):
    """Evaluate with corpus WER/CER and a sample-weighted mean CTC loss.

    WER and CER are ratios (0.0 to potentially above 1.0), not percentages.
    CER excludes literal spaces, matching the repository's existing convention.
    Teacher caches are unnecessary during validation and final evaluation.
    """
    device = torch.device(device)
    prediction_file = None
    if predictions_path is not None:
        prediction_file = Path(predictions_path)
        prediction_file.parent.mkdir(parents=True, exist_ok=True)

    totals = dict(loss=0.0, word_errors=0, word_count=0,
                  char_errors=0, char_count=0, samples=0)
    was_training = model.training
    model.eval()
    temporary_file = None
    try:
        # Open inside try so even an unwritable destination restores model mode.
        # Publish only complete predictions, preserving an existing file on error.
        if prediction_file is not None:
            stream_context = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="\n", dir=prediction_file.parent,
                prefix=f".{prediction_file.name}.", suffix=".tmp", delete=False)
            temporary_file = Path(stream_context.name)
        else:
            stream_context = nullcontext(None)
        with stream_context as stream:
            for batch in loader:
                frames = batch["frames"].to(device, non_blocking=True)
                lengths = batch["input_lengths"]
                logits = model(frames, lengths)
                loss = ctc_loss(
                    logits, batch["targets"].to(device, non_blocking=True),
                    lengths, batch["target_lengths"], blank_id=vocabulary.blank_id,
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite validation CTC loss; check targets and lengths.")
                predictions = greedy_decode(logits, lengths, vocabulary)
                references, sample_ids = batch["texts"], batch["ids"]
                if len(predictions) != len(references) or len(predictions) != len(sample_ids):
                    raise ValueError("Batch ids/texts do not match the number of model outputs.")
                totals["loss"] += float(loss) * len(predictions)
                totals["samples"] += len(predictions)

                for sample_id, reference, prediction in zip(sample_ids, references, predictions):
                    reference_words = reference.split()
                    reference_chars = reference.replace(" ", "")
                    word_errors = edit_distance(prediction.split(), reference_words)
                    char_errors = edit_distance(prediction.replace(" ", ""), reference_chars)
                    totals["word_errors"] += word_errors
                    totals["word_count"] += len(reference_words)
                    totals["char_errors"] += char_errors
                    totals["char_count"] += len(reference_chars)
                    if stream is not None:
                        stream.write(json.dumps({
                            "id": sample_id, "reference": reference,
                            "prediction": prediction, "word_errors": word_errors,
                            "word_count": len(reference_words),
                            "char_errors": char_errors, "char_count": len(reference_chars),
                        }, ensure_ascii=False) + "\n")
        if totals["samples"] == 0:
            raise ValueError("Evaluation loader is empty.")
        if totals["word_count"] == 0 or totals["char_count"] == 0:
            raise ValueError("References must contain words and non-space characters.")
        totals["loss"] /= totals["samples"]
        totals["wer"] = totals["word_errors"] / totals["word_count"]
        totals["cer"] = totals["char_errors"] / totals["char_count"]
        if temporary_file is not None:
            os.replace(temporary_file, prediction_file)
            temporary_file = None
        return totals
    finally:
        model.train(was_training)
        if temporary_file is not None:
            try:
                temporary_file.unlink(missing_ok=True)
            except OSError:
                # Preserve the original export/loader error if cleanup also
                # fails, for example because the filesystem became read-only.
                pass
