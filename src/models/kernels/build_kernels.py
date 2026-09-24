"""Build and compile native C SIMD INT8 kernel into dynamic library."""

from __future__ import annotations
import os
import sys
import shutil
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("edge_simd_builder")


def get_library_name() -> str:
    """Return platform-specific dynamic library filename."""
    if sys.platform == "win32":
        return "edge_simd_int8.dll"
    elif sys.platform == "darwin":
        return "edge_simd_int8.dylib"
    else:
        return "edge_simd_int8.so"


def find_compiler() -> Optional[str]:
    """Look for available C compilers in system PATH."""
    for cc in ["clang", "gcc", "cl"]:
        found = shutil.which(cc)
        if found:
            return found
    return None


def build_native_dll(force: bool = False) -> Optional[str]:
    """Compile edge_simd_int8.c into a shared library.

    Returns the absolute path to the compiled shared library, or None if no
    compatible compiler is installed.
    """
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    c_source = os.path.join(kernel_dir, "edge_simd_int8.c")
    dll_path = os.path.join(kernel_dir, get_library_name())

    if not os.path.isfile(c_source):
        return None

    if os.path.isfile(dll_path) and not force:
        return dll_path

    compiler = find_compiler()
    if not compiler:
        logger.debug("No external C compiler found (clang/gcc/cl) for native DLL build.")
        return None

    compiler_base = os.path.basename(compiler).lower()

    try:
        if "cl" in compiler_base and not ("clang" in compiler_base or "gcc" in compiler_base):
            # MSVC compiler
            cmd = [
                compiler,
                "/O2",
                "/arch:AVX2",
                "/LD",
                f"/Fe:{dll_path}",
                c_source,
            ]
        else:
            # GCC or Clang
            cmd = [
                compiler,
                "-O3",
                "-mavx2",
                "-shared",
                "-fPIC",
                "-o",
                dll_path,
                c_source,
            ]

        res = subprocess.run(cmd, cwd=kernel_dir, capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and os.path.isfile(dll_path):
            logger.info("Successfully compiled native SIMD kernel: %s", dll_path)
            return dll_path
        else:
            logger.debug("Compiler returned non-zero code %d: %s", res.returncode, res.stderr)
            return None
    except Exception as exc:
        logger.debug("Failed to compile native kernel: %s", exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = build_native_dll(force=True)
    if path:
        print(f"Native kernel compiled: {path}")
    else:
        print("No external compiler found. Falling back to Tier 2 Numba LLVM JIT.")
