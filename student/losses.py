"""CTC supervision and valid-frame standard and adaptive distillation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_lengths(logits: torch.Tensor, input_lengths: torch.Tensor) -> None:
    if logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("Student logits must be a floating-point [B,T,V] tensor")
    if logits.shape[0] < 1 or logits.shape[1] < 1 or logits.shape[2] < 2:
        raise ValueError("Student logits require B >= 1, T >= 1 and V >= 2")
    if input_lengths.ndim != 1 or len(input_lengths) != logits.shape[0]:
        raise ValueError("input_lengths must have one value per batch item")
    if input_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_lengths must contain integers")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > logits.shape[1]):
        raise ValueError("input_lengths must be positive and no greater than T")


def ctc_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    """Mean CTC loss in FP32; impossible alignments are not silently zeroed."""
    _validate_lengths(student_logits, input_lengths)
    if target_lengths.ndim != 1 or len(target_lengths) != student_logits.shape[0]:
        raise ValueError("target_lengths must have one value per batch item")
    if target_lengths.dtype not in (torch.int32, torch.int64) or torch.any(target_lengths <= 0):
        raise ValueError("target_lengths must contain positive integers")
    if targets.ndim != 1 or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must be concatenated integer token IDs")
    if targets.numel() != int(target_lengths.sum()):
        raise ValueError("Concatenated targets do not match target_lengths")
    if not 0 <= blank_id < student_logits.shape[-1]:
        raise ValueError("blank_id outside vocabulary")
    if torch.any(targets < 0) or torch.any(targets >= student_logits.shape[-1]) or torch.any(targets == blank_id):
        raise ValueError("CTC targets must be valid nonblank token IDs")
    start = 0
    for length, available in zip(target_lengths.tolist(), input_lengths.tolist()):
        tokens = targets[start : start + length]
        required = length + int((tokens[1:] == tokens[:-1]).sum())
        if required > available:
            raise ValueError(f"CTC alignment needs {required} frames but only {available} are valid")
        start += length
    mask = torch.arange(student_logits.shape[1], device=student_logits.device)[None, :] < input_lengths.to(student_logits.device)[:, None]
    logits = student_logits.float().masked_fill(~mask[:, :, None], 0)
    if not torch.isfinite(logits).all():
        raise ValueError("Valid Student logits contain NaN or Inf")
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    # PyTorch CUDA CTC backward is not deterministic. Preserve the strict flag
    # with a differentiable CPU loss instead of silently weakening the guarantee.
    loss_device = torch.device("cpu") if (student_logits.device.type == "cuda" and
                                          torch.are_deterministic_algorithms_enabled()) else student_logits.device
    loss = F.ctc_loss(
        log_probs.to(loss_device),
        targets.to(loss_device),
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=blank_id,
        reduction="mean",
        zero_infinity=False,
    )
    return loss.to(student_logits.device)


def _distillation_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_lengths: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_lengths(student_logits, input_lengths)
    if teacher_logits.shape != student_logits.shape or not teacher_logits.is_floating_point():
        raise ValueError("Teacher logits must be floating point and match student [B,T,V]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    positions = torch.arange(student_logits.shape[1], device=student_logits.device)
    mask = positions.unsqueeze(0) < input_lengths.to(student_logits.device).unsqueeze(1)
    student = student_logits.float()[mask] / temperature
    teacher = teacher_logits.detach().to(device=student_logits.device, dtype=torch.float32)[mask]
    teacher_tempered = teacher / temperature
    if not torch.isfinite(student).all() or not torch.isfinite(teacher_tempered).all():
        raise ValueError("Valid distillation logits contain NaN or Inf")
    return student, teacher_tempered, teacher


def standard_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_lengths: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """tau² * mean_valid_frame KL(teacher_tau || student_tau)."""
    student, teacher, _ = _distillation_logits(student_logits, teacher_logits, input_lengths, temperature)
    loss = F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    )
    return loss * (temperature * temperature)


def adaptive_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_lengths: torch.Tensor,
    temperature: float = 2.0,
    *,
    epsilon: float = 0.1,
    blank_id: int | None = None,
) -> torch.Tensor:
    """Weight tempered KL by teacher confidence and optionally nonblank probability.

    Compute weights from the ordinary teacher distribution. Normalize them to
    mean one over the valid frames of the batch so the KD scale stays comparable.
    """
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("epsilon must be finite and in (0, 1]")
    student, teacher_tempered, teacher = _distillation_logits(
        student_logits, teacher_logits, input_lengths, temperature
    )
    vocab_size = teacher.shape[-1]
    if blank_id is not None and (type(blank_id) is not int or not 0 <= blank_id < vocab_size):
        raise ValueError("blank_id outside vocabulary")
    probabilities = teacher.softmax(dim=-1)
    entropy = torch.special.entr(probabilities).sum(dim=-1)
    confidence = (1 - entropy / math.log(vocab_size)).clamp(0, 1)
    if blank_id is not None:
        confidence = confidence * (1 - probabilities[:, blank_id])
    # Keep even very small positive floors representable until normalization.
    weights = epsilon + (1 - epsilon) * confidence.double()
    frame_kl = F.kl_div(
        student.log_softmax(dim=-1), teacher_tempered.softmax(dim=-1), reduction="none"
    ).sum(dim=-1)
    weights = (weights / weights.sum()).to(frame_kl.dtype)
    return (weights * frame_kl).sum() * (temperature * temperature)
