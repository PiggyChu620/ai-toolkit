"""Checkpoint saving that does not stop training.

save() used to write the LoRA, optimizer.pt and the data-loader state on the training thread,
about a minute per save on the USB drive the job folders live on. Now the training thread only
takes private CPU copies (a second or two) and one background thread hashes and writes them.

- safetensors' save_file holds the GIL for its whole disk write (measured: a 21 s freeze of every
  other thread per 0.9 GB on that drive), so LoRA files are written here by a small streaming
  writer instead: the same bytes, written in 4 MB slices (larger writes made any other file
  operation on that drive wait for seconds), with the GIL released during each write.
- torch.save releases the GIL while it writes tensor storages, so optimizer.pt uses it as is.
- Every file is written under a dot-prefixed temporary name, which the resume and clean-up globs
  ("<job name>*") and the UI's file list never match, fsynced, and only then renamed into place:
  optimizer.pt and the data-loader state first, the LoRA (what resume starts from) last. A stop,
  crash or BSOD mid-write leaves the previous checkpoint intact instead of a truncated newest one.
"""
import functools
import hashlib
import itertools
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, List, Optional, Tuple

import torch

TEMP_PREFIX = ".aitk_saving_"
_WRITE_CHUNK = 4 * 1024 * 1024
_REPLACE_TIMEOUT_S = 10.0

_ST_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
    torch.uint8: "U8", torch.bool: "BOOL",
}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2"),
                     ("uint16", "U16"), ("uint32", "U32"), ("uint64", "U64")):
    if hasattr(torch, _name):
        _ST_DTYPES[getattr(torch, _name)] = _code


# -- safetensors, streamed ------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _dtype_rank():
    """safetensors lays tensors out by dtype (its own enum order, descending), then by name. Ask
    the installed version once instead of hard-coding an order that has changed between versions."""
    import safetensors.torch
    probes = {}
    for dtype, code in _ST_DTYPES.items():
        try:
            probes[code] = torch.zeros(1, dtype=dtype)
        except Exception:
            pass
    data = safetensors.torch.save(probes)
    length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + length])
    order = sorted((info["data_offsets"][0], code) for code, info in header.items() if code != "__metadata__")
    return {code: index for index, (_, code) in enumerate(order)}


def _st_dtype(tensor):
    try:
        return _ST_DTYPES[tensor.dtype]
    except KeyError:
        raise TypeError(f"safetensors cannot store dtype {tensor.dtype}") from None


def _tensor_bytes(tensor):
    """A view of the tensor's bytes, row-major as safetensors stores them; no copy if contiguous."""
    if tensor.device.type != "cpu":
        raise ValueError("checkpoint tensors must already be on the CPU")
    if tensor.numel() == 0:
        return memoryview(b"")
    return memoryview(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy())


def _layout(tensors):
    rank = _dtype_rank()
    names = sorted(tensors, key=lambda name: (rank[_st_dtype(tensors[name])], name))
    offsets, position = {}, 0
    for name in names:
        tensor = tensors[name]
        size = tensor.numel() * tensor.element_size()
        offsets[name] = (position, position + size)
        position += size
    return names, offsets, position


def _header(tensors, names, offsets, metadata):
    """The exact header safetensors writes: 8-byte length, compact JSON padded with spaces to a
    multiple of 8, "__metadata__" first (when given), then the tensors in data order."""
    header = {}
    if metadata is not None:
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("safetensors metadata must map str to str")
        header["__metadata__"] = metadata
    for name in names:
        tensor = tensors[name]
        header[name] = {"dtype": _st_dtype(tensor), "shape": list(tensor.shape),
                        "data_offsets": list(offsets[name])}
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return len(raw).to_bytes(8, "little") + raw


def _feed(hasher, view, start=0, end=None):
    end = len(view) if end is None else end
    for position in range(start, end, _WRITE_CHUNK):
        hasher.update(view[position:min(position + _WRITE_CHUNK, end)])


def addnet_hashes(tensors, metadata):
    """The two hashes toolkit.metadata.add_model_hash_to_meta stores, from the tensors themselves.

    That function serializes the whole state dict once more in memory (about 3x its size at the
    peak) only to hash it. Same results here without the copy:
    - sshs_model_hash: sha256 of the data section, which is the same whatever the metadata;
    - sshs_legacy_hash: sha256 of the 64 KiB at file offset 1 MiB of the file as serialized with
      only the ss_* metadata entries.
    """
    names, offsets, data_size = _layout(tensors)
    model = hashlib.sha256()
    for name in names:
        _feed(model, _tensor_bytes(tensors[name]))
    ss_metadata = {key: value for key, value in (metadata or {}).items() if key.startswith("ss_")}
    header = _header(tensors, names, offsets, ss_metadata)
    window_start, window_end = 0x100000, 0x110000
    legacy = hashlib.sha256()
    if window_start < len(header):
        legacy.update(header[window_start:window_end])
    data_start = max(window_start - len(header), 0)
    data_end = min(window_end - len(header), data_size)
    for name in names:
        begin, end = offsets[name]
        if end <= data_start or begin >= data_end:
            continue
        _feed(legacy, _tensor_bytes(tensors[name]), max(data_start - begin, 0), min(data_end, end) - begin)
    return {"sshs_model_hash": model.hexdigest(), "sshs_legacy_hash": legacy.hexdigest()[0:8]}


def write_safetensors(path, tensors, metadata=None, consume=False):
    """safetensors' save_file, streamed: the GIL is released while each slice is written, and with
    consume=True every tensor is dropped from `tensors` as soon as it is on disk."""
    names, offsets, _ = _layout(tensors)
    header = _header(tensors, names, offsets, metadata)
    with open(path, "wb") as f:
        f.write(header)
        for name in names:
            view = _tensor_bytes(tensors[name])
            for position in range(0, len(view), _WRITE_CHUNK):
                f.write(view[position:position + _WRITE_CHUNK])
            del view
            if consume:
                tensors[name] = None
        f.flush()
        os.fsync(f.fileno())


# -- private CPU copies ---------------------------------------------------------------------------

class UnsupportedState(TypeError):
    """The state holds an object that cannot be copied safely; save it the old way instead."""


def snapshot_to_cpu(state):
    """A private CPU copy of a state_dict that the training thread can keep changing.

    Optimizers hand back live tensors (bitsandbytes' 8-bit states, AdamW's moments and its CPU
    "step") that the next step updates in place, so every tensor is copied. Note `.to('cpu')`
    alone would alias a tensor that is already on the CPU. Containers are rebuilt, tensors shared
    between entries (bitsandbytes' qmap) stay shared, and anything that is not a plain value raises
    UnsupportedState rather than being saved by reference.
    """
    copies = {}

    def copy(value):
        if isinstance(value, torch.Tensor):
            key = id(value)
            if key not in copies:
                copies[key] = (value, value.detach().to("cpu", copy=True))
            return copies[key][1]
        if value is None or isinstance(value, (str, bytes, bool, Number, torch.dtype, torch.device, torch.Size)):
            return value
        if type(value) is dict:
            return {key: copy(item) for key, item in value.items()}
        if type(value) is OrderedDict:
            return OrderedDict((key, copy(item)) for key, item in value.items())
        if type(value) is list:
            return [copy(item) for item in value]
        if type(value) is tuple:
            return tuple(copy(item) for item in value)
        raise UnsupportedState(f"cannot copy a {type(value).__module__}.{type(value).__name__} out of the optimizer state")

    return copy(state)


# -- the writer -----------------------------------------------------------------------------------

@dataclass
class CheckpointJob:
    step: Optional[int]
    lora_path: str
    lora_tensors: dict
    lora_metadata: dict
    # torch.save'd alongside the LoRA, e.g. optimizer.pt and the data-loader state; a failure here
    # is reported but does not stop the LoRA from being saved (as before)
    companions: List[Tuple[str, Any, str]] = field(default_factory=list)  # (path, object, label)
    notes: List[str] = field(default_factory=list)  # what could not be included, reported with the result
    committed_early: List[str] = field(default_factory=list)  # files the training thread already replaced


@dataclass
class CheckpointResult:
    step: Optional[int]
    lora_path: str
    seconds: float = 0.0
    error: Optional[BaseException] = None  # the LoRA could not be saved
    notes: List[str] = field(default_factory=list)
    committed: List[str] = field(default_factory=list)  # files already replaced, even if the LoRA failed


def _fsync_path(path):
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace(source, target):
    """os.replace with retries: on Windows it fails while anything (a virus scanner, the UI) has
    the target open, whatever its share mode."""
    deadline = time.monotonic() + _REPLACE_TIMEOUT_S
    delay = 0.02
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


_temp_counter = itertools.count()


def temp_path_for(final_path):
    # short on purpose: a temp name longer than the final one could cross Windows' MAX_PATH
    # (260) where the final path still fits
    folder = os.path.dirname(final_path)
    return os.path.join(folder, f"{TEMP_PREFIX}{os.getpid()}_{next(_temp_counter)}.tmp")


def remove_stale_temps(folder, printer=print):
    """Temporary files of a save that was cut off (stop, crash, BSOD); the checkpoint before it is intact."""
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in names:
        if name.startswith(TEMP_PREFIX):
            try:
                os.remove(os.path.join(folder, name))
                printer(f"Removed the leftovers of an interrupted save: {name}")
            except OSError:
                pass


class BackgroundCheckpointWriter:
    """Writes one checkpoint at a time on a background thread.

    The training thread calls submit() with private copies, then collects the outcome with
    poll() (non-blocking) or wait(). A new submit() first waits for the previous checkpoint, so
    at most one set of copies is ever held.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._job: Optional[CheckpointJob] = None
        self._result: Optional[CheckpointResult] = None

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def pending_step(self):
        job = self._job
        return job.step if job is not None and self.busy else None

    def submit(self, job: CheckpointJob):
        if self._thread is not None:
            raise RuntimeError("collect the previous checkpoint with wait() before submitting another")
        self._job = job
        self._result = None
        self._thread = threading.Thread(target=self._run, args=(job,), name="aitk-checkpoint-writer", daemon=True)
        self._thread.start()

    def join(self, timeout=None) -> bool:
        """Wait for the write to end without collecting its result (safe from another thread).
        True if nothing is being written any more."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def poll(self) -> Optional[CheckpointResult]:
        """The finished checkpoint's result, once; None while it is still being written or if none."""
        if self._thread is None or self._thread.is_alive():
            return None
        return self._collect()

    def wait(self, timeout=None) -> Optional[CheckpointResult]:
        """Block until the checkpoint being written is done (or the timeout passes: then None)."""
        thread = self._thread
        if thread is None:
            return None
        thread.join(timeout)
        if thread.is_alive():
            return None
        return self._collect()

    def _collect(self):
        result = self._result
        self._thread = None
        self._job = None
        self._result = None
        return result

    def _run(self, job: CheckpointJob):
        started = time.monotonic()
        result = CheckpointResult(step=job.step, lora_path=job.lora_path, notes=list(job.notes),
                                  committed=list(job.committed_early))
        staged = []  # (temp, final) in commit order
        try:
            # the optimizer first: torch.save adds no memory, and its copy is gone before the LoRA
            # is hashed, which keeps the peak at about one copy
            for path, obj, label in job.companions:
                temp = temp_path_for(path)
                try:
                    torch.save(obj, temp)
                    _fsync_path(temp)
                    staged.append((temp, path))
                except Exception as e:
                    result.notes.append(f"Could not save {label}: {e}")
                    _remove_quietly(temp)
            job.companions = []
            obj = None

            tensors = job.lora_tensors
            job.lora_tensors = None
            metadata = OrderedDict(job.lora_metadata or {})
            metadata.update(addnet_hashes(tensors, metadata))
            temp = temp_path_for(job.lora_path)
            try:
                write_safetensors(temp, tensors, metadata, consume=True)
            except BaseException:
                _remove_quietly(temp)
                raise
            staged.append((temp, job.lora_path))
            tensors = None

            # commit: the companions, then the LoRA, which is what resume starts from (if the LoRA
            # itself failed above, nothing at all was committed)
            for temp, final in staged[:-1]:
                try:
                    _replace(temp, final)
                    result.committed.append(os.path.basename(final))
                except Exception as e:
                    result.notes.append(f"Could not save {os.path.basename(final)}: {e}")
            _replace(*staged[-1])
            result.committed.append(os.path.basename(staged[-1][1]))
        except BaseException as e:
            result.error = e
        finally:
            for temp, _ in staged:
                _remove_quietly(temp)
            result.seconds = time.monotonic() - started
            self._result = result


def _remove_quietly(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
