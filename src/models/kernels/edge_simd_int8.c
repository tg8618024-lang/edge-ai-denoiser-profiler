/**
 * Real-Time Edge AI Audio Denoiser & Profiler
 * Native SIMD Integer-Domain Matrix-Vector (GEMV) Implementation
 *
 * Implements:
 * - Runtime CPUID capability detection (AVX-VNNI, AVX2, SSE4.1, Scalar)
 * - VEX-encoded vpdpbusd (AVX-VNNI)
 * - vpmaddubsw + vpmaddwd (AVX2)
 * - pmaddubsw + pmaddwd (SSE4.1)
 * - Unrolled 8-way scalar fallback
 */

#include "edge_simd_int8.h"

#if defined(_MSC_VER)
  #include <intrin.h>
  #include <immintrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <x86intrin.h>
  #include <cpuid.h>
  #include <immintrin.h>
#endif

/**
 * Detect runtime host CPU SIMD capability via CPUID
 */
SIMD_EXPORT int int8_simd_detect_capability(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    int info[4];

    // CPUID Leaf 1: Features
    #if defined(_MSC_VER)
        __cpuid(info, 1);
    #else
        __cpuid(1, info[0], info[1], info[2], info[3]);
    #endif

    int has_sse41   = (info[2] & (1 << 19)) != 0;
    int has_osxsave = (info[2] & (1 << 27)) != 0;
    int has_avx     = (info[2] & (1 << 28)) != 0;

    if (!has_sse41) {
        return SIMD_CAP_SCALAR;
    }

    if (!has_osxsave || !has_avx) {
        return SIMD_CAP_SSE41;
    }

    // CPUID Leaf 7 Subleaf 0: Extended Features
    #if defined(_MSC_VER)
        __cpuidex(info, 7, 0);
    #else
        __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
    #endif

    int has_avx2        = (info[1] & (1 << 5)) != 0;
    int has_avx512_vnni = (info[2] & (1 << 11)) != 0;

    if (!has_avx2) {
        return SIMD_CAP_SSE41;
    }

    // CPUID Leaf 7 Subleaf 1: AVX-VNNI (EAX bit 4)
    int has_avx_vnni = 0;
    #if defined(_MSC_VER)
        __cpuidex(info, 7, 1);
        has_avx_vnni = (info[0] & (1 << 4)) != 0;
    #elif defined(__GNUC__) || defined(__clang__)
        __cpuid_count(7, 1, info[0], info[1], info[2], info[3]);
        has_avx_vnni = (info[0] & (1 << 4)) != 0;
    #endif

    if (has_avx512_vnni || has_avx_vnni) {
        return SIMD_CAP_VNNI;
    }
    return SIMD_CAP_AVX2;
#else
    return SIMD_CAP_SCALAR;
#endif
}

/**
 * AVX-VNNI Kernel: 32 INT8 MACs per cycle using _mm256_dpbusd_epi32
 */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__AVX_VNNI__)
__attribute__((target("avx2,avxvnni")))
#endif
SIMD_EXPORT void int8_gemv_vnni(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
) {
#if (defined(__AVX2__) || defined(__AVX_VNNI__) || defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
    for (int j = 0; j < M; ++j) {
        const int8_t* w_row = W_transposed + (size_t)j * (size_t)K;
        __m256i acc = _mm256_setzero_si256();

        int i = 0;
        for (; i + 32 <= K; i += 32) {
            __m256i vx = _mm256_loadu_si256((const __m256i*)(const void*)(x_u8 + i));
            __m256i vw = _mm256_loadu_si256((const __m256i*)(const void*)(w_row + i));
            #if defined(__AVX_VNNI__) || defined(_MSC_VER) || (defined(__GNUC__) && defined(__AVX_VNNI__))
                acc = _mm256_dpbusd_epi32(acc, vx, vw);
            #else
                // Emulate dpbusd if compiler built without AVX-VNNI flag
                __m256i prod16 = _mm256_maddubs_epi16(vx, vw);
                __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
                acc = _mm256_add_epi32(acc, prod32);
            #endif
        }

        // Horizontal sum of the eight 32-bit integer lanes in acc
        __m128i low  = _mm256_castsi256_si128(acc);
        __m128i high = _mm256_extracti128_si256(acc, 1);
        __m128i sum4 = _mm_add_epi32(low, high);
        sum4 = _mm_hadd_epi32(sum4, sum4);
        sum4 = _mm_hadd_epi32(sum4, sum4);
        int32_t dot_sum = _mm_cvtsi128_si32(sum4);

        // Remainder loop
        for (; i < K; ++i) {
            dot_sum += (int32_t)x_u8[i] * (int32_t)w_row[i];
        }

        if (bias) {
            dot_sum += bias[j];
        }
        out[j] = dot_sum;
    }
#else
    int8_gemv_scalar(x_u8, W_transposed, bias, out, M, K);
#endif
}

/**
 * AVX2 Kernel: 32 INT8 MACs per cycle using _mm256_maddubs_epi16 and _mm256_madd_epi16
 */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__AVX2__)
__attribute__((target("avx2")))
#endif
SIMD_EXPORT void int8_gemv_avx2(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
) {
#if (defined(__AVX2__) || defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
    const __m256i ones = _mm256_set1_epi16(1);

    for (int j = 0; j < M; ++j) {
        const int8_t* w_row = W_transposed + (size_t)j * (size_t)K;
        __m256i acc = _mm256_setzero_si256();

        int i = 0;
        for (; i + 32 <= K; i += 32) {
            __m256i vx = _mm256_loadu_si256((const __m256i*)(const void*)(x_u8 + i));
            __m256i vw = _mm256_loadu_si256((const __m256i*)(const void*)(w_row + i));

            // Multiply unsigned byte (vx) by signed byte (vw) -> intermediate saturated 16-bit
            __m256i prod16 = _mm256_maddubs_epi16(vx, vw);
            // Sum adjacent pairs of 16-bit into 32-bit integers
            __m256i prod32 = _mm256_madd_epi16(prod16, ones);
            acc = _mm256_add_epi32(acc, prod32);
        }

        // Horizontal sum of the eight 32-bit accumulators
        __m128i low  = _mm256_castsi256_si128(acc);
        __m128i high = _mm256_extracti128_si256(acc, 1);
        __m128i sum4 = _mm_add_epi32(low, high);
        sum4 = _mm_hadd_epi32(sum4, sum4);
        sum4 = _mm_hadd_epi32(sum4, sum4);
        int32_t dot_sum = _mm_cvtsi128_si32(sum4);

        for (; i < K; ++i) {
            dot_sum += (int32_t)x_u8[i] * (int32_t)w_row[i];
        }

        if (bias) {
            dot_sum += bias[j];
        }
        out[j] = dot_sum;
    }
#else
    int8_gemv_scalar(x_u8, W_transposed, bias, out, M, K);
#endif
}

/**
 * SSE4.1 Kernel: 16 INT8 MACs per cycle using 128-bit XMM registers
 */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__SSE4_1__)
__attribute__((target("sse4.1")))
#endif
SIMD_EXPORT void int8_gemv_sse41(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
) {
#if (defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX2__) || defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    const __m128i ones = _mm_set1_epi16(1);

    for (int j = 0; j < M; ++j) {
        const int8_t* w_row = W_transposed + (size_t)j * (size_t)K;
        __m128i acc = _mm_setzero_si128();

        int i = 0;
        for (; i + 16 <= K; i += 16) {
            __m128i vx = _mm_loadu_si128((const __m128i*)(const void*)(x_u8 + i));
            __m128i vw = _mm_loadu_si128((const __m128i*)(const void*)(w_row + i));

            __m128i prod16 = _mm_maddubs_epi16(vx, vw);
            __m128i prod32 = _mm_madd_epi16(prod16, ones);
            acc = _mm_add_epi32(acc, prod32);
        }

        acc = _mm_hadd_epi32(acc, acc);
        acc = _mm_hadd_epi32(acc, acc);
        int32_t dot_sum = _mm_cvtsi128_si32(acc);

        for (; i < K; ++i) {
            dot_sum += (int32_t)x_u8[i] * (int32_t)w_row[i];
        }

        if (bias) {
            dot_sum += bias[j];
        }
        out[j] = dot_sum;
    }
#else
    int8_gemv_scalar(x_u8, W_transposed, bias, out, M, K);
#endif
}

/**
 * Portable 8x Unrolled Scalar Fallback
 */
SIMD_EXPORT void int8_gemv_scalar(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
) {
    for (int j = 0; j < M; ++j) {
        const int8_t* w_row = W_transposed + (size_t)j * (size_t)K;
        int32_t dot_sum = bias ? bias[j] : 0;

        int i = 0;
        for (; i + 8 <= K; i += 8) {
            dot_sum += (int32_t)x_u8[i + 0] * (int32_t)w_row[i + 0]
                     + (int32_t)x_u8[i + 1] * (int32_t)w_row[i + 1]
                     + (int32_t)x_u8[i + 2] * (int32_t)w_row[i + 2]
                     + (int32_t)x_u8[i + 3] * (int32_t)w_row[i + 3]
                     + (int32_t)x_u8[i + 4] * (int32_t)w_row[i + 4]
                     + (int32_t)x_u8[i + 5] * (int32_t)w_row[i + 5]
                     + (int32_t)x_u8[i + 6] * (int32_t)w_row[i + 6]
                     + (int32_t)x_u8[i + 7] * (int32_t)w_row[i + 7];
        }
        for (; i < K; ++i) {
            dot_sum += (int32_t)x_u8[i] * (int32_t)w_row[i];
        }
        out[j] = dot_sum;
    }
}

/**
 * Signed-input scalar GEMV kernel
 */
SIMD_EXPORT void int8_gemv_scalar_signed(
    const int8_t*  x,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
) {
    for (int j = 0; j < M; ++j) {
        const int8_t* w_row = W_transposed + (size_t)j * (size_t)K;
        int32_t dot_sum = bias ? bias[j] : 0;

        int i = 0;
        for (; i + 8 <= K; i += 8) {
            dot_sum += (int32_t)x[i + 0] * (int32_t)w_row[i + 0]
                     + (int32_t)x[i + 1] * (int32_t)w_row[i + 1]
                     + (int32_t)x[i + 2] * (int32_t)w_row[i + 2]
                     + (int32_t)x[i + 3] * (int32_t)w_row[i + 3]
                     + (int32_t)x[i + 4] * (int32_t)w_row[i + 4]
                     + (int32_t)x[i + 5] * (int32_t)w_row[i + 5]
                     + (int32_t)x[i + 6] * (int32_t)w_row[i + 6]
                     + (int32_t)x[i + 7] * (int32_t)w_row[i + 7];
        }
        for (; i < K; ++i) {
            dot_sum += (int32_t)x[i] * (int32_t)w_row[i];
        }
        out[j] = dot_sum;
    }
}

/**
 * Auto-dispatching entrypoint selecting the fastest available SIMD ISA
 */
SIMD_EXPORT void int8_gemv_auto(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
) {
    static int cached_cap = -1;
    if (cached_cap < 0) {
        cached_cap = int8_simd_detect_capability();
    }

    if (cached_cap >= SIMD_CAP_VNNI) {
        int8_gemv_vnni(x_u8, W_transposed, bias, out, M, K);
    } else if (cached_cap >= SIMD_CAP_AVX2) {
        int8_gemv_avx2(x_u8, W_transposed, bias, out, M, K);
    } else if (cached_cap >= SIMD_CAP_SSE41) {
        int8_gemv_sse41(x_u8, W_transposed, bias, out, M, K);
    } else {
        int8_gemv_scalar(x_u8, W_transposed, bias, out, M, K);
    }
}

/**
 * Standard alias matching dispatch specification
 */
SIMD_EXPORT void int8_gemv(
    const uint8_t* x_u8,
    const int8_t*  W_transposed,
    const int32_t* bias,
    int32_t*       out,
    int32_t        M,
    int32_t        K
) {
    int8_gemv_auto(x_u8, W_transposed, bias, out, M, K);
}
