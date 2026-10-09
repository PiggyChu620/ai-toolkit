"""Memory clean-up at the start of a job.

Two separate things, both safe to run at any job start:

kill_orphaned_workers()
    DataLoader worker processes of THIS install whose trainer is gone. Before the
    Windows job object (toolkit/windows_job.py) a hard-killed trainer left its
    workers running, each holding gigabytes of RAM until someone killed them by
    hand, so every new job started with less memory. Only workers whose parent
    process no longer exists (or whose parent PID was reused by a newer process)
    are killed; workers of a trainer that is still running are never touched.

release_cached_memory()
    Right before the training loop. Loading and quantizing the model leaves the
    CUDA caching allocator holding freed blocks of whatever sizes loading used,
    plus pinned staging buffers in the pinned-host cache. Training reuses those
    only where sizes happen to fit, so they fragment its memory. Releasing them
    lets training start from a consolidated state. (Physical RAM itself needs no
    defragmenting: memory a process releases is simply free again.)
"""

import gc
import os
import sys

import torch

GiB = 1024 ** 3


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except Exception:
        return False


def kill_orphaned_workers(printer=print) -> int:
    """Kill orphaned multiprocessing workers of this Python install. Returns how many."""
    try:
        import psutil
    except Exception:
        return 0
    me = os.getpid()
    killed, freed = [], 0
    try:
        for proc in psutil.process_iter(["pid", "ppid", "exe", "cmdline", "create_time"]):
            info = proc.info
            try:
                if info["pid"] == me or not info.get("exe") or not _same_file(info["exe"], sys.executable):
                    continue
                cmd = " ".join(info.get("cmdline") or [])
                if "multiprocessing.spawn" not in cmd and "--multiprocessing-fork" not in cmd:
                    continue
                orphan = False
                try:
                    parent = psutil.Process(info["ppid"])
                    # a parent created after the child is a recycled PID, not the real parent
                    orphan = parent.create_time() > info["create_time"]
                except psutil.NoSuchProcess:
                    orphan = True
                if not orphan:
                    continue
                mem = proc.memory_info()
                size = getattr(mem, "private", None) or mem.rss
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
                killed.append(info["pid"])
                freed += size
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        printer(f"(orphaned-worker check failed: {type(e).__name__}: {e})")
        return len(killed)
    if killed:
        printer(f"Killed {len(killed)} orphaned data-loader worker(s) left by earlier jobs "
                f"(PIDs {', '.join(map(str, killed))}), freeing {freed / GiB:.2f} GiB of RAM.")
    else:
        printer("No orphaned data-loader workers from earlier jobs.")
    return len(killed)


def release_cached_memory(printer=print) -> dict:
    """Return cached-but-unused GPU and pinned-host memory before training starts."""
    out = {}
    try:
        try:
            import psutil
            proc = psutil.Process()
            ram0 = getattr(proc.memory_info(), "private", None)
        except Exception:
            proc, ram0 = None, None
        gc.collect()
        if not torch.cuda.is_available():
            return out
        try:
            from .manager_modules import release_offload_buffers
            release_offload_buffers()  # synchronizes before dropping the rings
        except Exception:
            pass
        torch.cuda.synchronize()
        gpu0 = torch.cuda.memory_reserved()
        host0 = torch.cuda.host_memory_stats().get("reserved_bytes.current")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if callable(host_empty):
            host_empty()  # frees only pinned blocks nothing uses; the arena slabs stay
        gpu1 = torch.cuda.memory_reserved()
        host1 = torch.cuda.host_memory_stats().get("reserved_bytes.current")
        ram1 = getattr(proc.memory_info(), "private", None) if proc else None
        out = {"gpu_reserved": (gpu0, gpu1), "pinned_reserved": (host0, host1), "ram_private": (ram0, ram1)}

        def pair(a, b):
            return "?" if a is None or b is None else f"{a / GiB:.2f} -> {b / GiB:.2f} GiB"

        printer(f"Released cached memory before training: GPU cache {pair(gpu0, gpu1)} | "
                f"pinned cache {pair(host0, host1)}"
                + ("" if callable(host_empty) else " (pinned cache release not available in this torch)")
                + f" | process RAM {pair(ram0, ram1)}")
    except Exception as e:
        try:
            printer(f"(memory release before training skipped: {type(e).__name__}: {e})")
        except Exception:
            pass
    return out
