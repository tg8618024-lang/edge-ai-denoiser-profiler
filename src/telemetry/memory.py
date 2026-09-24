"""Native zero-dependency host process memory telemetry.

Queries OS memory counters directly using Win32 API (ctypes) on Windows
and resource / /proc/self/status on POSIX without external dependencies.
"""

from __future__ import annotations
import sys
import os
from typing import NamedTuple


class MemorySnapshot(NamedTuple):
    """Snapshot of process memory usage in megabytes."""

    rss_mb: float
    private_mb: float = 0.0
    peak_rss_mb: float = 0.0


def get_process_memory() -> MemorySnapshot:
    """Retrieve current process Resident Set Size (RSS) without external dependencies.

    Returns
    -------
    MemorySnapshot
        MemorySnapshot containing rss_mb, private_mb, and peak_rss_mb.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes as wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            handle = kernel32.GetCurrentProcess()

            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                rss = counters.WorkingSetSize / (1024.0 * 1024.0)
                private = counters.PrivateUsage / (1024.0 * 1024.0)
                peak = counters.PeakWorkingSetSize / (1024.0 * 1024.0)
                return MemorySnapshot(
                    rss_mb=round(rss, 3),
                    private_mb=round(private, 3),
                    peak_rss_mb=round(peak, 3),
                )
        except Exception:
            pass

        # Fallback to dynamic Python heap telemetry via tracemalloc
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            current, peak = tracemalloc.get_traced_memory()
            mem_mb = max(1.0, round(current / (1024.0 * 1024.0), 3))
            peak_mb = max(mem_mb, round(peak / (1024.0 * 1024.0), 3))
            return MemorySnapshot(rss_mb=mem_mb, private_mb=mem_mb, peak_rss_mb=peak_mb)
        except Exception:
            return MemorySnapshot(rss_mb=1.0, private_mb=1.0, peak_rss_mb=1.0)

    else:
        # POSIX (Linux / macOS)
        rss_mb = 0.0
        try:
            import resource

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS ru_maxrss is in bytes; Linux ru_maxrss is in kilobytes
            scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
            rss_mb = rusage.ru_maxrss / scale

            # On Linux, try reading immediate VmRSS from /proc/self/status
            if os.path.exists("/proc/self/status"):
                with open("/proc/self/status", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = float(line.split()[1])
                            rss_mb = kb / 1024.0
                            break
        except Exception:
            pass

        if rss_mb <= 0.0:
            try:
                import tracemalloc
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                current, peak = tracemalloc.get_traced_memory()
                rss_mb = max(1.0, round(current / (1024.0 * 1024.0), 3))
            except Exception:
                rss_mb = 1.0

        return MemorySnapshot(
            rss_mb=round(rss_mb, 3),
            private_mb=round(rss_mb, 3),
            peak_rss_mb=round(rss_mb, 3),
        )
