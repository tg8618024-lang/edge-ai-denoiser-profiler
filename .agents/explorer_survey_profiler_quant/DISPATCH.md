# Dispatch: Explorer 2 (Profiler & Multi-Precision Quantization)

## Context
Project: Edge AI Audio Denoiser & Profiler
Target Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Your Working Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_profiler_quant
Authoritative Request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md

## Mission
Survey and design the NVIDIA-Style Latency Profiler & Multi-Precision Comparison Engine (Requirements R2, R3 and acceptance criteria):
1. Stage Telemetry: Sub-millisecond timing across exactly 3 stages: Pre-processing (STFT, framing, windowing), Tensor Compute (neural network / DSP filter inference), and Output Synthesis (iSTFT, overlap-add). Must use high-resolution hardware timers (e.g. `perf_counter_ns` / CUDA events if applicable).
2. Rolling Statistics: Circular buffer tracking frame metrics to compute median, P95, and P99 latency against real-time budget (e.g., <= 20ms). Headroom calculations.
3. Multi-Precision Formats: Design implementation for FP32, FP16, and INT8 inference modes. Detail how weights and computations are quantized (e.g., PyTorch dynamic quantization `torch.ao.quantization` or ONNX / NumPy quantized matrix ops), memory footprint measurement (RSS / model weight size in MB/KB), latency speedups, and SNR impact.
4. Telemetry Schema: JSON/dataclass schema for publishing per-frame and aggregated stats to profiler logs and dashboard.
5. Deliverables: Comprehensive survey report at `.agents/explorer_survey_profiler_quant/report.md` detailing profiling instrumentation, rolling statistics algorithms, quantization strategies, and data contracts.

## 2026-09-06T17:46:14Z
<USER_REQUEST>
You are Explorer 2 (Profiler & Multi-Precision Specialist).
Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_profiler_quant
Authoritative request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
Your dispatch instructions: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_profiler_quant\DISPATCH.md

Inspect ORIGINAL_REQUEST.md and design the NVIDIA-Style Latency Profiler & Multi-Precision Comparison Engine:
1. Stage Telemetry: Sub-millisecond timing across exactly 3 stages: Pre-processing (STFT, framing, windowing), Tensor Compute (neural network or DSP filtering inference), and Output Synthesis (iSTFT, overlap-add). Must use high-resolution timers (`time.perf_counter_ns()`).
2. Rolling Statistics: Ring buffer tracking frame metrics to compute median, P95, and P99 latency against real-time budget (<= 20ms) and calculate headroom.
3. Multi-Precision Formats: Design implementation for FP32, FP16, and INT8 inference modes. Detail how weights and computations are quantized (PyTorch dynamic quantization `torch.ao.quantization` or simulated/native quantized int8 matrix multiply), memory footprint measurement (RSS / model weight size in MB/KB), latency speedups, and SNR impact.
4. Telemetry Schema: JSON/dataclass schema for publishing per-frame and aggregated stats to profiler logs and dashboard.
5. Write your findings and architecture report to C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_profiler_quant\report.md and a handoff.md. Send a completion message back to parent.
</USER_REQUEST>
