# WebSocket Concurrency & Load-Testing Benchmark Report

Automated concurrency evaluation for the Real-Time Edge AI Audio Pipeline (FastAPI / WebSocket backend).
Generated at: 2026-09-20 18:53:09 UTC

---

## 1. Executive Summary

| Concurrent Clients | Aggregate Throughput | Frame P50 Latency | Frame P95 Latency | Frame P99 Latency | RTT P50 | Jitter | Memory RSS | Budget Headroom |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 client(s)** | 142.3 FPS | 0.892 ms | 1.686 ms | 2.457 ms | 4.838 ms | 0.315 ms | 121.5 MB (+2.94 MB) | 95.5% |
| **5 client(s)** | 142.9 FPS | 11.453 ms | 21.214 ms | 32.738 ms | 32.447 ms | 5.225 ms | 124.75 MB (+3.25 MB) | 42.7% |
| **20 client(s)** | 136.6 FPS | 41.957 ms | 94.433 ms | 171.552 ms | 128.127 ms | 25.306 ms | 137.54 MB (+12.8 MB) | 0.0% |

---

## 2. Key Architecture Findings & Scaling Analysis

### Isolated Per-Client Audio Pipelines
- **Zero Thread-Lock Bottleneck**: Each incoming WebSocket connection receives an isolated `AudioDenoisingPipeline`, `StageProfiler`, and `RollingMetricsBuffer`. Shared server locks are only held for global setting mutations.
- **Zero Buffer Interleaving**: FFT shift buffers, iSTFT overlap-add history, and GRUMaskNet recurrent hidden states are strictly thread-local.

### Concurrency Scaling Breakdown
#### Tier: 1 Concurrent Client(s)
- **Total Frames Processed**: 50 frames in 0.35s (142.3 FPS)
- **Latency Percentiles**: P50 = 0.892 ms, P95 = 1.686 ms, P99 = 2.457 ms
- **Round-Trip Time (RTT)**: Median = 4.838 ms, Jitter = 0.315 ms
- **Real-Time Factor (RTF)**: 0.045 (Budget: 20.0 ms)
- **Memory Footprint**: 121.5 MB total (~3010.6 KB / client)

#### Tier: 5 Concurrent Client(s)
- **Total Frames Processed**: 250 frames in 1.75s (142.9 FPS)
- **Latency Percentiles**: P50 = 11.453 ms, P95 = 21.214 ms, P99 = 32.738 ms
- **Round-Trip Time (RTT)**: Median = 32.447 ms, Jitter = 5.225 ms
- **Real-Time Factor (RTF)**: 0.573 (Budget: 20.0 ms)
- **Memory Footprint**: 124.75 MB total (~665.6 KB / client)

#### Tier: 20 Concurrent Client(s)
- **Total Frames Processed**: 1000 frames in 7.32s (136.6 FPS)
- **Latency Percentiles**: P50 = 41.957 ms, P95 = 94.433 ms, P99 = 171.552 ms
- **Round-Trip Time (RTT)**: Median = 128.127 ms, Jitter = 25.306 ms
- **Real-Time Factor (RTF)**: 2.098 (Budget: 20.0 ms)
- **Memory Footprint**: 137.54 MB total (~655.4 KB / client)

---

## 3. Prometheus Metric Telemetry

All metrics collected during concurrency load tests were streamed directly into Prometheus histograms and gauges:
- `audio_pipeline_frame_latency_seconds`: Multi-stage latency breakdown across pre-processing, tensor compute, and output synthesis.
- `audio_pipeline_active_clients`: Successfully verified dynamic gauge increment and decrement on connection lifecycle.
- `audio_pipeline_processed_frames_total`: Verified frame accounting partitioned across FP32 and INT8 quantization modes.
- `audio_pipeline_process_memory_rss_bytes`: Monitored resident physical memory.
