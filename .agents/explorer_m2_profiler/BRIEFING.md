# BRIEFING — 2026-09-07T01:04:30Z

## Mission
Investigate and specify the design, interface contracts, math, and unit tests for the 3-stage hardware latency profiler and telemetry subsystem (Milestone 2).

## 🔒 My Identity
- Archetype: explorer
- Roles: explorer, telemetry specialist
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_profiler
- Original parent: 781c31a8-414c-4676-94ba-d0072e691cd9
- Milestone: M2 - 3-stage hardware latency profiler & telemetry subsystem

## 🔒 Key Constraints
- Read-only investigation — do NOT implement source files directly
- Must support both start_stage/end_stage and start_frame/mark_* API signatures
- Pure zero-dependency memory retrieval (Win32 psapi + POSIX resource)
- Real-time headroom against <= 20.0 ms budget

## Current Parent
- Conversation ID: 781c31a8-414c-4676-94ba-d0072e691cd9
- Updated: 2026-09-07T01:04:30Z

## Investigation State
- **Explored paths**: tests/conftest.py (MockStageProfiler), tests/e2e/test_evaluation.py (profiler, ring buffer, memory tests), src/audio/pipeline.py, PROJECT.md, TEST_INFRA.md, .agents/explorer_survey_profiler_quant/report.md, .agents/explorer_survey_ui_e2e/report.md
- **Key findings**:
  1. `StageProfiler`: Requires dual API. Pipeline calls `start_stage`/`end_stage`, while evaluation tests call `start_frame`/`mark_*`. `end_stage("output_synthesis")` must auto-finalize frame metrics. `FrameMetrics` must support dict and attribute access.
  2. `RollingMetricsBuffer`: Preallocated NumPy (C, 4) ring buffer with zero append allocation. Computes P50/P95/P99, jitter (mean absolute delta), overrun counts, headroom, and RTF. Empty buffer must yield safe defaults without division by zero.
  3. `get_process_memory`: Zero-dependency native query using Win32 `psapi.GetProcessMemoryInfo` (64-bit `ctypes.c_size_t` fields in `PROCESS_MEMORY_COUNTERS_EX`) and POSIX `resource.getrusage`. Returns `MemorySnapshot(rss_mb, ...)`.
  4. `TelemetryPayload`: Supports both flat dictionary (UI WebSocket) and nested dictionary (`PROJECT.md` M2<->M3 schema).
  5. `tests/unit/test_profiler.py`: Full Tier 1 and Tier 2 test suite planned covering all boundary conditions.
- **Unexplored areas**: None. All telemetry contracts and formulas are mapped and verified.

## Key Decisions Made
- Dual-API StageProfiler: Support both `start_stage`/`end_stage` and `start_frame`/`mark_*`.
- Hybrid dict/dataclass `FrameMetrics`: Inherit from `dict` with `@property` accessors for zero-overhead, dual `metrics["k"]` and `metrics.k` compatibility.
- Zero-dependency ctypes Win32 memory reading strictly configured for 64-bit Windows pointers and types.
- Complete self-contained implementation plan drafted in `handoff.md`.

## Artifact Index
- handoff.md — Comprehensive M2 profiler and telemetry architecture and implementation handoff report
- progress.md — Heartbeat and task tracking
