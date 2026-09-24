/**
 * @file vst_bridge_c_api.h
 * @brief C ABI Interface for Edge AI Audio Denoiser VST3 / CLAP / AU Plugins.
 *
 * Designed for linking with JUCE C++ audio plugins and Rust nih-plug crates.
 */

#ifndef EDGE_AI_DENOISER_VST_BRIDGE_C_API_H
#define EDGE_AI_DENOISER_VST_BRIDGE_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Parameter Indices */
typedef enum {
    EDGE_PARAM_BYPASS         = 0,  /* 0.0 = active, 1.0 = bypassed */
    EDGE_PARAM_DENOISE_AMOUNT = 1,  /* 0.0 = dry, 1.0 = full denoise */
    EDGE_PARAM_MODEL_SELECT   = 2,  /* 0 = Neural, 1 = Wiener DSP, 2 = Dual Crossfade */
    EDGE_PARAM_CROSSFADE      = 3,  /* 0.0 = Model A, 1.0 = Model B */
    EDGE_PARAM_PRECISION      = 4   /* 0 = FP32, 1 = FP16, 2 = INT8 */
} EdgePluginParamId;

typedef int32_t EdgePluginHandle;

/**
 * @brief Create a new plugin engine instance.
 * @param sample_rate Audio sample rate in Hz (e.g. 16000, 44100, 48000).
 * @return Positive handle on success, negative error code on failure.
 */
EdgePluginHandle edge_vst_create(int32_t sample_rate);

/**
 * @brief Destroy a plugin engine instance and free resources.
 * @param handle Plugin instance handle.
 * @return 0 on success, -1 on invalid handle.
 */
int32_t edge_vst_destroy(EdgePluginHandle handle);

/**
 * @brief Process an audio block of arbitrary block size.
 * @param handle Plugin instance handle.
 * @param in_buffer Pointer to input float32 array (mono).
 * @param out_buffer Pointer to output float32 array (mono).
 * @param num_samples Number of samples in block (e.g. 32, 64, 128, 256, 512, 1024).
 * @return 0 on success, -1 on failure.
 */
int32_t edge_vst_process(
    EdgePluginHandle handle,
    const float* in_buffer,
    float* out_buffer,
    uint32_t num_samples
);

/**
 * @brief Set an automated parameter value.
 * @param handle Plugin instance handle.
 * @param param Parameter identifier.
 * @param value Floating-point parameter value.
 * @return 0 on success, -1 on invalid parameter.
 */
int32_t edge_vst_set_param(
    EdgePluginHandle handle,
    EdgePluginParamId param,
    float value
);

/**
 * @brief Retrieve current parameter value.
 * @param handle Plugin instance handle.
 * @param param Parameter identifier.
 * @return Parameter value as float, or -1.0f on error.
 */
float edge_vst_get_param(
    EdgePluginHandle handle,
    EdgePluginParamId param
);

/**
 * @brief Query plugin latency in samples for DAW Plugin Delay Compensation (PDC).
 * @param handle Plugin instance handle.
 * @return Number of samples of latency (e.g. 256 samples).
 */
int32_t edge_vst_get_latency_samples(EdgePluginHandle handle);

/**
 * @brief Reset plugin internal delay lines and filters.
 * @param handle Plugin instance handle.
 * @return 0 on success, -1 on error.
 */
int32_t edge_vst_reset(EdgePluginHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* EDGE_AI_DENOISER_VST_BRIDGE_C_API_H */
