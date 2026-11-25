"""Verify that Student runs differ only in their intended training treatment."""

from __future__ import annotations

from collections.abc import Collection
import json
import math
import re


_ADAPTIVE_MODES = {"kd-entropy", "kd-blank"}
_MODES = {"ctc", "kd", *_ADAPTIVE_MODES}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MISSING = object()
_OPTIONAL_INTS = {"limit_train", "limit_val"}
_SHARED_FIELDS = {
    "train_roi_fingerprint": str,
    "val_data_fingerprint": str,
    "train_samples": int,
    "val_samples": int,
    "vocab_tokens": list,
    "model_config": dict,
    "parameters": int,
    "preprocessing_version": str,
    "decoder": str,
    "metric_unit": str,
    "seed": int,
    "batch_size": int,
    "epochs": int,
    "lr": float,
    "weight_decay": float,
    "grad_clip": float,
    "lr_patience": int,
    "lr_monitor": str,
    "amp": bool,
    "deterministic": bool,
    "smoke_test": bool,
    "limit_train": int,
    "limit_val": int,
}


def _read_field(config, role, key, kind, differences):
    value = config.get(key, _MISSING)
    if value is _MISSING:
        differences.append(f"{role} config is missing {key}")
        return _MISSING
    if value is None and key in _OPTIONAL_INTS:
        return value
    if kind is float:
        valid = type(value) is int or (type(value) is float and math.isfinite(value))
    elif key.endswith("_fingerprint") or key == "implementation_sha256":
        valid = type(value) is str and _SHA256.fullmatch(value) is not None
    else:
        valid = type(value) is kind
    if not valid:
        differences.append(f"{role} config has invalid {key}")
        return _MISSING
    return value


def check_reference_config(
    current: dict,
    reference: dict,
    *,
    compatible_reference_implementations: Collection[str] = (),
) -> None:
    """Raise ValueError if two trainer-generated configurations cannot be compared.

    The caller approves equivalent source revisions for the reference mode. Raw
    manifest hashes, paths and hardware details may differ between machines. The
    content hashes cover selected sample IDs, labels, lengths and file bytes.
    """
    if not isinstance(current, dict) or not isinstance(reference, dict):
        raise ValueError("Current and reference configs must be JSON objects")

    differences = []

    def read_pair(key, kind):
        return (
            _read_field(current, "current", key, kind, differences),
            _read_field(reference, "reference", key, kind, differences),
        )

    modes = read_pair("mode", str)
    for role, mode in zip(("current", "reference"), modes):
        if mode is not _MISSING and mode not in _MODES:
            differences.append(f"{role} config has unknown mode {mode!r}")

    current_sha, reference_sha = read_pair("implementation_sha256", str)
    if (current_sha is not _MISSING and reference_sha is not _MISSING
            and current_sha != reference_sha and reference_sha not in compatible_reference_implementations):
        differences.append(f"implementation_sha256 is unapproved for the reference ({reference_sha})")

    fields = dict(_SHARED_FIELDS)
    if current.get("amp") is True and reference.get("amp") is True:
        fields["amp_init_scale"] = float
    if all(mode in _MODES and mode != "ctc" for mode in modes):
        fields.update(train_data_fingerprint=str, temperature=float, kd_weight=float)
    if all(mode in _ADAPTIVE_MODES for mode in modes):
        fields["kd_epsilon"] = float

    for key, kind in fields.items():
        current_value, reference_value = read_pair(key, kind)
        if current_value is _MISSING or reference_value is _MISSING:
            continue
        if kind is dict:
            keys = current_value.keys() | reference_value.keys()
            changed = [name for name in sorted(keys)
                       if json.dumps(current_value.get(name), sort_keys=True)
                       != json.dumps(reference_value.get(name), sort_keys=True)
                       or (name in current_value) != (name in reference_value)]
            if changed:
                differences.append(f"{key} differs for {', '.join(changed)}")
        elif current_value != reference_value:
            detail = ("content or order" if kind is list
                      else f"current {current_value!r}, reference {reference_value!r}")
            differences.append(f"{key} differs ({detail})")

    if differences:
        raise ValueError("Reference config is incompatible: " + "; ".join(differences))
