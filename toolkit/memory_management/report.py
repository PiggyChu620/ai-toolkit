"""Host (CPU) memory breakdown of a training run.

The trainer once held ~60 GB of RAM where its parts should add up to ~20-25 GB,
and nothing in the log said where it went. This prints, at the start of training
and at every checkpoint save, how the process's private memory splits up:

  - pinned memory (page-locked, held by PyTorch's pinned allocator / the arena),
  - model tensors that live in ordinary CPU memory, per component,
  - everything else ("not attributed": Python, CUDA runtime, dataset, caches...).

Comparing successive reports shows whether memory grows during a run. The
report only reads metadata (sizes, devices, pointers); it never copies tensors
and must never stop training.
"""

import ctypes
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch

GiB = 1024 ** 3


def _gib(x: Optional[float]) -> str:
    return "?" if x is None else f"{x / GiB:.2f} GiB"


def _system_commit() -> Tuple[Optional[int], Optional[int]]:
    """(committed bytes, commit limit) for the whole machine, Windows only."""
    if os.name != "nt":
        return None, None
    try:
        class PERFORMANCE_INFORMATION(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t),
                        ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                        ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                        ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
                        ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                        ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
                        ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]
        info = PERFORMANCE_INFORMATION()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None, None
        return info.CommitTotal * info.PageSize, info.CommitLimit * info.PageSize
    except Exception:
        return None, None


def _leaves(t) -> Iterable[torch.Tensor]:
    """The real storage-backed tensors behind a (possibly subclass) tensor."""
    if hasattr(t, "__tensor_flatten__") and type(t) is not torch.Tensor and type(t) is not torch.nn.Parameter:
        try:
            names, _ = t.__tensor_flatten__()
        except Exception:
            names = []
        for name in names:
            inner = getattr(t, name, None)
            if isinstance(inner, torch.Tensor):
                yield from _leaves(inner)
        return
    yield t


def _components(sd) -> List[Tuple[str, torch.nn.Module]]:
    out, seen = [], set()

    def add(name, m):
        if isinstance(m, torch.nn.Module) and id(m) not in seen:
            seen.add(id(m))
            out.append((name, m))

    add("transformer/unet", getattr(sd, "model", None) or getattr(sd, "unet", None))
    te = getattr(sd, "text_encoder", None)
    if isinstance(te, (list, tuple)):
        for i, m in enumerate(te):
            add(f"text_encoder[{i}]", m)
    else:
        add("text_encoder", te)
    add("vae", getattr(sd, "vae", None))
    return out


def log_memory_breakdown(sd=None, network=None, label: str = "", printer=print) -> Optional[Dict]:
    try:
        rows = []
        proc_private = proc_ws = None
        try:
            import psutil
            mi = psutil.Process().memory_info()
            proc_private = getattr(mi, "private", None) or getattr(mi, "vms", None)
            proc_ws = mi.rss
        except Exception:
            pass
        commit, commit_limit = _system_commit()

        pinned_reserved = pinned_allocated = None
        if torch.cuda.is_available():
            try:
                hs = torch.cuda.host_memory_stats()
                pinned_reserved = hs.get("reserved_bytes.current")
                pinned_allocated = hs.get("allocated_bytes.current")
            except Exception:
                pass
        try:
            from .manager_modules import pinned_arena_stats
            arena = pinned_arena_stats()
        except Exception:
            arena = None

        # CPU-resident model tensors per component. Each leaf is counted once, by its
        # own byte size (views into the arena are counted as what they hold).
        seen_ptrs = set()
        biggest = []
        comps = _components(sd) if sd is not None else []
        if isinstance(network, torch.nn.Module) and all(m is not network for _, m in comps):
            comps.append(("network (LoRA)", network))
        cpu_pinned_total = cpu_pageable_total = 0
        for name, module in comps:
            pinned = pageable = 0
            tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
            for p in tensors:
                for leaf in _leaves(p):
                    try:
                        if leaf.device.type != "cpu":
                            continue
                        key = (leaf.data_ptr(), leaf.numel(), leaf.dtype)
                        if key in seen_ptrs:
                            continue
                        seen_ptrs.add(key)
                        n = leaf.numel() * leaf.element_size()
                        if leaf.is_pinned():
                            pinned += n
                        else:
                            pageable += n
                            biggest.append((n, name, tuple(leaf.shape), str(leaf.dtype).replace("torch.", "")))
                    except Exception:
                        continue
            cpu_pinned_total += pinned
            cpu_pageable_total += pageable
            if pinned or pageable:
                rows.append((name, pinned, pageable))

        accounted = (pinned_reserved or 0) + cpu_pageable_total
        unattributed = None if proc_private is None else proc_private - accounted

        printer("")
        printer(f"=== Host memory {label} ===".rstrip())
        if commit is not None:
            pct = 100.0 * commit / commit_limit if commit_limit else 0.0
            printer(f"  system commit: {_gib(commit)} of {_gib(commit_limit)} ({pct:.0f}%)"
                    + ("   <-- nearly full: other programs or this run may crash" if pct >= 92 else ""))
        printer(f"  this process: private {_gib(proc_private)} | working set {_gib(proc_ws)}")
        printer(f"  pinned (page-locked): {_gib(pinned_reserved)} reserved, {_gib(pinned_allocated)} in use")
        if arena is not None:
            if arena.get("enabled") or arena.get("slabs"):
                printer(f"    arena: {arena['tensors']} weights, {_gib(arena['payload_bytes'])} packed into "
                        f"{arena['slabs']} slab(s) totalling {_gib(arena['slab_bytes'])} "
                        f"({_gib(arena['unused_bytes'])} unused)")
            if arena.get("disabled_reason"):
                printer(f"    arena disabled: {arena['disabled_reason']}")
        for name, pinned, pageable in rows:
            printer(f"  {name:20s} on CPU: {_gib(pinned)} pinned + {_gib(pageable)} pageable")
        printer(f"  not attributed (Python, CUDA runtime, dataset, caches, ...): {_gib(unattributed)}")
        biggest.sort(reverse=True)
        if biggest and biggest[0][0] >= 256 * 1024 ** 2:
            top = ", ".join(f"{name} {shape} {dt} {_gib(n)}" for n, name, shape, dt in biggest[:3])
            printer(f"  largest pageable model tensors: {top}")
        printer("")
        return {
            "private": proc_private, "working_set": proc_ws, "commit": commit, "commit_limit": commit_limit,
            "pinned_reserved": pinned_reserved, "pinned_allocated": pinned_allocated, "arena": arena,
            "components": rows, "cpu_pinned": cpu_pinned_total, "cpu_pageable": cpu_pageable_total,
            "unattributed": unattributed,
        }
    except Exception as e:
        try:
            printer(f"(memory breakdown unavailable: {type(e).__name__}: {e})")
        except Exception:
            pass
        return None
