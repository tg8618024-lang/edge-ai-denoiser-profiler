# Explorer M2 Profiler Task

## Objective
Investigate the technical implementation requirements for the 3-stage hardware-accurate latency profiler and telemetry subsystem:
1. `src/telemetry/profiler.py`: `StageProfiler` isolating pre-processing, tensor compute, output synthesis with `perf_counter_ns`. Must support both `start_stage`/`end_stage` and `start_frame`/`mark_preprocessing_done`/`mark_tensor_done`/`mark_synthesis_done` as expected by `tests/conftest.py` and `tests/e2e/test_evaluation.py`.
2. `src/telemetry/ring_buffer.py`: `RollingMetricsBuffer` with fixed circular preallocated ring buffer, computing P50, P95, P99 percentiles, jitter, overrun counts, and real-time headroom against <= 20.0 ms frame budget.
3. `src/telemetry/memory.py`: Native zero-dependency process memory telemetry using `ctypes` on Windows (`psapi.GetProcessMemoryInfo` returning `rss_mb`) and `resource.getrusage` on POSIX.
4. `src/telemetry/schema.py`: Dataclass and serialization for `TelemetryPayload` matching M2 <-> M3 contract.
5. Unit tests: `tests/unit/test_profiler.py`.
6. Output detailed implementation design, exact formulas, class interfaces, and verification commands in `handoff.md`.
