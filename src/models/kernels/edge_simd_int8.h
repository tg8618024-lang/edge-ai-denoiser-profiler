/**
 * Real-Time Edge AI Audio Denoiser & Profiler
 * Native SIMD Integer-Domain Matrix-Vector (GEMV) Kernel Header
 *
 * Supported ISAs:
 * - Intel AVX-VNNI / AVX-512 VNNI (vpdpbusd / _mm256_dpbusd_epi32)
 * - Intel AVX2 (vpmaddubsw / vpmaddwd / _mm256_maddubs_epi16 / _mm256_madd_epi16)
 * - Intel SSE4.1 (pmaddubsw / pmaddwd / _mm_maddubs_epi16 / _mm_madd_epi16)
 * - Portable Scalar Fallback (unrolled 8x)
 */

#ifndef EDGE_SIMD_INT8_H
#define EDGE_SIMD_INT8_H

#include <stdint.h>
#include <stddef.h>

#if defined(_MSC_VER)
  #define SIMD_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
  #define SIMD_EXPORT __attribute__((visibility("default")))
#else
  #define SIMD_EXPORT
#endif

// Capability identifiers returned by int8_simd_detect_capability
#define SIMD_CAP_SCALAR  0
#define SIMD_CAP_SSE41   1
#define SIMD_CAP_AVX2    2
#define SIMD_CAP_VNNI    3

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Detect runtime host CPU SIMD capability via CPUID instruction.
 * Returns one of SIMD_CAP_*.
 */
SIMD_EXPORT int int8_simd_detect_capability(void);

/**
 * AVX-VNNI (VEX-encoded vpdpbusd) GEMV Kernel:
 * 32 INT8 MACs per cycle directly accumulating into 32-bit integer destinations.
 *
 * @param x_u8         Pointer to uint8 activations (length K)
 * @param W_transposed Pointer to int8 transposed weights (shape M x K, row-major)
 * @param bias         Pointer to int32 folded bias vector (length M, or NULL)
 * @param out          Pointer to int32 accumulator destination (length M)
 * @param M            Number of output rows
 * @param K            Number of input columns
 */
SIMD_EXPORT void int8_gemv_vnni(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
);

/**
 * AVX2 (vpmaddubsw + vpmaddwd) GEMV Kernel:
 * 32 INT8 MACs per cycle using 256-bit YMM registers.
 */
SIMD_EXPORT void int8_gemv_avx2(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
);

/**
 * SSE4.1 (pmaddubsw + pmaddwd) GEMV Kernel:
 * 16 INT8 MACs per cycle using 128-bit XMM registers.
 */
SIMD_EXPORT void int8_gemv_sse41(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
);

/**
 * Portable 8x unrolled scalar fallback.
 */
SIMD_EXPORT void int8_gemv_scalar(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
);

/**
 * Signed-input scalar GEMV kernel.
 */
SIMD_EXPORT void int8_gemv_scalar_signed(
    const int8_t*  x,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
);

/**
 * Primary auto-dispatching entrypoint.
 * Automatically interrogates CPUID on initial invocation and branches
 * to the highest performing SIMD instruction set supported by the silicon.
 */
SIMD_EXPORT void int8_gemv_auto(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
);

/**
 * Standard alias matching dispatch specification.
 */
SIMD_EXPORT void int8_gemv(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
);

#ifdef __cplusplus
}
#endif

#endif // EDGE_SIMD_INT8_H
