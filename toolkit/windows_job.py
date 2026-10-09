"""Tie every child process (DataLoader workers, ...) to the trainer's lifetime on Windows.

Windows does not kill child processes when their parent dies, and there are paths
where the trainer dies without shutting its DataLoader workers down:
DiffusionTrainer's stop watcher ends the process with
os.kill(os.getpid(), SIGINT), which on Windows is TerminateProcess (an instant hard
kill, exit code 2, no clean-up), and so does killing the trainer in Task Manager or
a crash. Because the trainer is then already gone, the UI's stop route never runs
its `taskkill /T` tree kill either, and orphaned workers kept holding gigabytes of
RAM across jobs.

A Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE fixes this at the OS level:
the trainer puts itself in the job, every process it spawns afterwards joins the
job automatically, and when the trainer ends -- for ANY reason -- the kernel closes
the last handle to the job and terminates everything still in it. The handle is
deliberately never closed and is not inheritable, so only the trainer holds it.
"""

import ctypes
import os

_JOB_HANDLE = None


def kill_children_with_this_process() -> str:
    """Returns a short status string; never raises."""
    global _JOB_HANDLE
    if os.name != "nt":
        return "not needed (not Windows)"
    if _JOB_HANDLE:
        return "active"
    try:
        from ctypes import wintypes as wt

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC_LIMITS(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD),
                        ("SchedulingClass", wt.DWORD)]

        class EXTENDED_LIMITS(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC_LIMITS), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wt.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
        k32.SetInformationJobObject.restype = wt.BOOL
        k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        k32.AssignProcessToJobObject.restype = wt.BOOL
        k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        k32.GetCurrentProcess.restype = wt.HANDLE
        k32.CloseHandle.argtypes = [wt.HANDLE]

        job = k32.CreateJobObjectW(None, None)  # NULL attributes: handle is not inheritable
        if not job:
            return f"unavailable (CreateJobObject error {ctypes.get_last_error()})"
        info = EXTENDED_LIMITS()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            k32.CloseHandle(job)
            return f"unavailable (SetInformationJobObject error {err})"
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            err = ctypes.get_last_error()
            k32.CloseHandle(job)
            return f"unavailable (AssignProcessToJobObject error {err})"
        _JOB_HANDLE = job
        return "active"
    except Exception as e:
        return f"unavailable ({type(e).__name__}: {e})"
