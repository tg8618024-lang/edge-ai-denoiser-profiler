"""Tiered SIMD Hardware Acceleration Dispatcher for INT8 Inference.

Provides a robust 3-tier dispatch architecture:
- Tier 1: Native C shared library (edge_simd_int8.dll/.so) via ctypes releasing GIL
- Tier 2: Numba LLVM JIT vectorization kernel (@njit(fastmath=True, nogil=True))
- Tier 3: Zero-allocation cached NumPy fallback with pre-transposed weights
"""

from __future__ import annotations
import os
import sys
import ctypes
import threading
import logging
from typing import Optional, Tuple
import numpy as np

from .build_kernels import build_native_dll, get_library_name

logger = logging.getLogger("edge_simd_dispatch")


class SIMDDispatcher:
    """Multi-tier hardware SIMD dispatcher for INT8 integer-domain GEMV."""

    TIER_1_NATIVE = 1
    TIER_2_NUMBA = 2
    TIER_3_NUMPY = 3

    def __init__(self, preferred_tier: Optional[int] = None) -> None:
        self._tier = self.TIER_3_NUMPY
        self._tier_name = "Tier 3: Zero-Allocation Cached NumPy Fallback"
        self._native_lib: Optional[ctypes.CDLL] = None
        self._has_numba = False

        # Scratch buffer for Tier 3 zero-allocation upcast with thread safety
        self._scratch_x32 = np.zeros(1024, dtype=np.int32)
        self._tier3_lock = threading.Lock()

        # 1. Probe Tier 1: Native C DLL
        self._init_tier1()

        # 2. Probe Tier 2: Numba LLVM JIT
        self._init_tier2()

        # 3. Select active tier based on availability and preference
        self.set_tier(preferred_tier)

    def _init_tier1(self) -> None:
        """Attempt to locate and load native compiled C SIMD library."""
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        dll_name = get_library_name()
        dll_path = os.path.join(kernel_dir, dll_name)

        if not os.path.isfile(dll_path):
            # Attempt to build if compiler is present
            dll_path = build_native_dll() or ""

        if os.path.isfile(dll_path):
            try:
                lib = ctypes.CDLL(dll_path)
                # Configure function signatures
                lib.int8_simd_detect_capability.restype = ctypes.c_int
                lib.int8_gemv_auto.argtypes = [
                    ctypes.c_void_p,  # const uint8_t* x_u8
                    ctypes.c_void_p,  # const int8_t*  W_transposed
                    ctypes.c_void_p,  # const int32_t* bias
                    ctypes.c_void_p,  # int32_t*       out
                    ctypes.c_int32,   # int32_t        M
                    ctypes.c_int32,   # int32_t        K
                ]
                lib.int8_gemv_auto.restype = None

                lib.int8_gemv_scalar_signed.argtypes = [
                    ctypes.c_void_p,  # const int8_t*  x
                    ctypes.c_void_p,  # const int8_t*  W_transposed
                    ctypes.c_void_p,  # const int32_t* bias
                    ctypes.c_void_p,  # int32_t*       out
                    ctypes.c_int32,   # int32_t        M
                    ctypes.c_int32,   # int32_t        K
                ]
                lib.int8_gemv_scalar_signed.restype = None

                self._native_lib = lib
                cap = lib.int8_simd_detect_capability()
                cap_names = {0: "Scalar", 1: "SSE4.1", 2: "AVX2", 3: "AVX-VNNI"}
                logger.info("Loaded Tier 1 native SIMD library (%s detected)", cap_names.get(cap, "Unknown"))
            except Exception as exc:
                logger.debug("Tier 1 ctypes load failed: %s", exc)
                self._native_lib = None

    def _init_tier2(self) -> None:
        """Attempt to initialize Numba LLVM JIT vectorizer."""
        try:
            import numba
            from numba import njit

            @njit(fastmath=True, nogil=True)
            def _jit_gemv_u8(x_u8, w_t, bias, out):
                M, K = w_t.shape
                for j in range(M):
                    acc = np.int32(bias[j]) if bias is not None else np.int32(0)
                    for i in range(K):
                        acc += np.int32(x_u8[i]) * np.int32(w_t[j, i])
                    out[j] = acc

            @njit(fastmath=True, nogil=True)
            def _jit_gemv_s8(x_s8, w_t, bias, out):
                M, K = w_t.shape
                for j in range(M):
                    acc = np.int32(bias[j]) if bias is not None else np.int32(0)
                    for i in range(K):
                        acc += np.int32(x_s8[i]) * np.int32(w_t[j, i])
                    out[j] = acc

            # Warmup compile JIT functions on dummy data
            dummy_x_u8 = np.zeros(16, dtype=np.uint8)
            dummy_x_s8 = np.zeros(16, dtype=np.int8)
            dummy_w_t = np.zeros((8, 16), dtype=np.int8)
            dummy_b = np.zeros(8, dtype=np.int32)
            dummy_out = np.zeros(8, dtype=np.int32)

            _jit_gemv_u8(dummy_x_u8, dummy_w_t, dummy_b, dummy_out)
            _jit_gemv_u8(dummy_x_u8, dummy_w_t, None, dummy_out)
            _jit_gemv_s8(dummy_x_s8, dummy_w_t, dummy_b, dummy_out)
            _jit_gemv_s8(dummy_x_s8, dummy_w_t, None, dummy_out)

            self._jit_gemv_u8 = _jit_gemv_u8
            self._jit_gemv_s8 = _jit_gemv_s8
            self._has_numba = True
            logger.info("Tier 2 Numba LLVM JIT vectorizer initialized with nogil=True")
        except Exception as exc:
            logger.debug("Tier 2 Numba initialization unavailable: %s", exc)
            self._has_numba = False

    def set_tier(self, tier: Optional[int] = None) -> None:
        """Configure active backend tier (1, 2, or 3)."""
        if tier == self.TIER_1_NATIVE and self._native_lib is not None:
            self._tier = self.TIER_1_NATIVE
            cap = self._native_lib.int8_simd_detect_capability()
            cap_names = {0: "Scalar", 1: "SSE4.1", 2: "AVX2", 3: "AVX-VNNI"}
            self._tier_name = f"Tier 1: Native C SIMD ({cap_names.get(cap, 'x86_64')}, GIL-released)"
        elif (tier == self.TIER_2_NUMBA or tier is None) and self._has_numba:
            self._tier = self.TIER_2_NUMBA
            self._tier_name = "Tier 2: Numba LLVM JIT (AVX2/AVX-VNNI, GIL-released)"
        elif tier == self.TIER_1_NATIVE and self._native_lib is None:
            if self._has_numba:
                self._tier = self.TIER_2_NUMBA
                self._tier_name = "Tier 2: Numba LLVM JIT (AVX2/AVX-VNNI, GIL-released) [Tier 1 unavailable]"
            else:
                self._tier = self.TIER_3_NUMPY
                self._tier_name = "Tier 3: Zero-Allocation Cached NumPy Fallback [Tier 1 unavailable]"
        else:
            self._tier = self.TIER_3_NUMPY
            self._tier_name = "Tier 3: Zero-Allocation Cached NumPy Fallback"

    def get_tier(self) -> Tuple[int, str]:
        """Return active tier index and descriptive name."""
        return self._tier, self._tier_name

    def gemv(
        self,
        x: np.ndarray,
        w_transposed: np.ndarray,
        bias: Optional[np.ndarray],
        out: np.ndarray,
        is_unsigned_folded: bool = False,
    ) -> None:
        """Perform integer matrix-vector multiplication directly into out buffer.

        out = W_transposed @ x + bias

        Parameters
        ----------
        x : np.ndarray
            Activation vector of shape (K,), dtype int8 (if is_unsigned_folded=False)
            or uint8 (if is_unsigned_folded=True).
        w_transposed : np.ndarray
            Weight matrix transposed of shape (M, K), dtype int8, row-major contiguous.
        bias : Optional[np.ndarray]
            Folded or integer bias vector of shape (M,), dtype int32, or None.
        out : np.ndarray
            Destination output vector of shape (M,), dtype int32.
        is_unsigned_folded : bool
            Whether x is unsigned (x_u8 = x + 128) with precomputed folded bias.
        """
        M, K = w_transposed.shape

        if self._tier == self.TIER_1_NATIVE and self._native_lib is not None:
            # Foreign function call via ctypes releases the Python GIL
            if is_unsigned_folded:
                self._native_lib.int8_gemv_auto(
                    x.ctypes.data,
                    w_transposed.ctypes.data,
                    bias.ctypes.data if bias is not None else None,
                    out.ctypes.data,
                    ctypes.c_int32(M),
                    ctypes.c_int32(K),
                )
            else:
                self._native_lib.int8_gemv_scalar_signed(
                    x.ctypes.data,
                    w_transposed.ctypes.data,
                    bias.ctypes.data if bias is not None else None,
                    out.ctypes.data,
                    ctypes.c_int32(M),
                    ctypes.c_int32(K),
                )

        elif self._tier == self.TIER_2_NUMBA and self._has_numba:
            # Compiled LLVM function with nogil=True releases GIL
            if is_unsigned_folded:
                self._jit_gemv_u8(x, w_transposed, bias, out)
            else:
                self._jit_gemv_s8(x, w_transposed, bias, out)

        else:
            # Tier 3: Zero-allocation NumPy fallback (thread-safe)
            with self._tier3_lock:
                if K > len(self._scratch_x32):
                    self._scratch_x32 = np.zeros(K * 2, dtype=np.int32)
                np.copyto(self._scratch_x32[:K], x)
                np.dot(w_transposed, self._scratch_x32[:K], out=out)
                if bias is not None:
                    np.add(out, bias, out=out)


# Global singleton dispatcher instance
_GLOBAL_DISPATCHER: Optional[SIMDDispatcher] = None


def get_simd_dispatcher() -> SIMDDispatcher:
    """Retrieve or initialize the global SIMD dispatcher singleton."""
    global _GLOBAL_DISPATCHER
    if _GLOBAL_DISPATCHER is None:
        _GLOBAL_DISPATCHER = SIMDDispatcher()
    return _GLOBAL_DISPATCHER


def int8_gemv_dispatch(
    x: np.ndarray,
    w_transposed: np.ndarray,
    bias: Optional[np.ndarray],
    out: np.ndarray,
    is_unsigned_folded: bool = False,
) -> None:
    """Module-level dispatch helper for integer GEMV."""
    get_simd_dispatcher().gemv(x, w_transposed, bias, out, is_unsigned_folded)
