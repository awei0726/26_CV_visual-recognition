"""Atomic artifacts, stable model fingerprints, and an exclusive run lock.

Locks use the operating system, so a terminated process cannot leave a stale
held lock. The small lock file intentionally remains after release: deleting it
would let two processes lock different files at the same pathname.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


class RunLockedError(RuntimeError):
    """Another process owns the requested output directory."""


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate short Windows sharing/antivirus locks without hiding failures.

    Other OS/errors are raised immediately. Windows retries are limited to
    access/sharing/lock violations and at most 1.27 seconds of backoff.
    """
    delays = (.01, .02, .04, .08, .16, .32, .64)
    deadline = time.monotonic() + 1.5
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if (os.name != "nt" or getattr(error, "winerror", None) not in {5, 32, 33}
                    or attempt == len(delays)):
                raise
            delay = delays[attempt]
            if time.monotonic() + delay > deadline:
                raise
            time.sleep(delay)


@contextmanager
def run_lock(output: str | Path):
    """Acquire a nonblocking cross-process exclusive lock for a run directory.

    The directory is created if needed. Callers checking whether a run is empty
    should ignore ``.run.lock``. The lock is held throughout the context and is
    released by the OS even if the process crashes.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".run.lock"
    stream = lock_path.open("a+b")
    acquired = False
    try:
        # msvcrt locks a byte range. Ensure byte zero exists even for a new run.
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RunLockedError(
                    f"Another process is using output directory {output}; choose a different --output-dir"
                ) from error
            raise
        acquired = True
        yield lock_path
    finally:
        try:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


@contextmanager
def _atomic_stream(path: str | Path, *, binary: bool):
    """Write and sync a unique same-directory temporary file, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    options = {} if binary else {"encoding": "utf-8", "newline": "\n"}
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb" if binary else "w", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False, **options,
        ) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        # On Windows the temporary file must be closed before replacing.
        _replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Do not hide the original write/replace error if cleanup also
                # fails (for example, because the filesystem became read-only).
                pass


def atomic_json(value, path: str | Path) -> None:
    """Atomically save UTF-8 JSON; reject NaN/Infinity instead of emitting them."""
    with _atomic_stream(path, binary=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_text(value: str, path: str | Path) -> None:
    """Atomically save a UTF-8 text artifact such as training history."""
    with _atomic_stream(path, binary=False) as stream:
        stream.write(value)


def atomic_save(value, path: str | Path) -> None:
    """Atomically save a PyTorch artifact without changing an existing good file."""
    import torch
    with _atomic_stream(path, binary=True) as stream:
        torch.save(value, stream)


def model_state_sha256(state) -> str:
    """Hash dense model tensors independently of dict order/device/strides.

    The fingerprint includes each key, dtype, shape, and exact CPU tensor bytes.
    Sparse, meta, and quantized tensors are rejected explicitly: the Student's
    training state is dense, and these representations need extra metadata.
    No float conversion is performed, so FP16/BF16/FP32 remain distinguishable.
    """
    import torch
    if any(not isinstance(key, str) for key in state):
        raise TypeError("Model state keys must be strings")
    digest = hashlib.sha256(b"student-state-sha256-v1\0")

    def add(payload):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Model state value {key!r} is not a Tensor")
        if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
            raise TypeError(f"Model state tensor {key!r} must be a dense, materialized, nonquantized Tensor")
        add(key.encode("utf-8"))
        add(str(value.dtype).encode("ascii"))
        add(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        cpu = value.detach().to(device="cpu").resolve_conj().resolve_neg().contiguous()
        # Viewing bytes supports BF16 too, for which numpy has no native dtype.
        add(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
