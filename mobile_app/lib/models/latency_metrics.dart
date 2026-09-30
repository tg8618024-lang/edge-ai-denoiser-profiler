/// Hardware Latency Metrics Model for 3-Stage Pipeline Profiler.
class LatencyMetrics {
  final double tPreMs;
  final double tTensorMs;
  final double tSynthMs;
  final double budgetMs;
  final double p50Ms;
  final double p95Ms;
  final double p99Ms;
  final double jitterMs;
  final int overrunCount;

  const LatencyMetrics({
    required this.tPreMs,
    required this.tTensorMs,
    required this.tSynthMs,
    this.budgetMs = 16.0,
    required this.p50Ms,
    required this.p95Ms,
    required this.p99Ms,
    required this.jitterMs,
    this.overrunCount = 0,
  });

  /// Total frame compute latency across all 3 non-overlapping stages.
  double get totalMs => tPreMs + tTensorMs + tSynthMs;

  /// True if the total processing latency exceeds the real-time frame budget.
  bool get isOverrun => totalMs > budgetMs;

  /// Available real-time budget headroom as a percentage.
  double get headroomPercent {
    if (budgetMs <= 0.0) return 0.0;
    final remaining = budgetMs - totalMs;
    final pct = (remaining / budgetMs) * 100.0;
    return pct.clamp(0.0, 100.0);
  }

  /// Default baseline reference metrics from Edge AI benchmark audio tests.
  factory LatencyMetrics.defaultReference() {
    return const LatencyMetrics(
      tPreMs: 0.043,
      tTensorMs: 0.167,
      tSynthMs: 0.036,
      budgetMs: 16.0,
      p50Ms: 0.245,
      p95Ms: 0.312,
      p99Ms: 0.450,
      jitterMs: 0.015,
      overrunCount: 0,
    );
  }

  /// Simulated frame overrun metrics for boundary/warning testing.
  factory LatencyMetrics.overrunExample() {
    return const LatencyMetrics(
      tPreMs: 2.500,
      tTensorMs: 12.800,
      tSynthMs: 3.200,
      budgetMs: 16.0,
      p50Ms: 15.200,
      p95Ms: 18.500,
      p99Ms: 21.400,
      jitterMs: 2.150,
      overrunCount: 5,
    );
  }

  LatencyMetrics copyWith({
    double? tPreMs,
    double? tTensorMs,
    double? tSynthMs,
    double? budgetMs,
    double? p50Ms,
    double? p95Ms,
    double? p99Ms,
    double? jitterMs,
    int? overrunCount,
  }) {
    return LatencyMetrics(
      tPreMs: tPreMs ?? this.tPreMs,
      tTensorMs: tTensorMs ?? this.tTensorMs,
      tSynthMs: tSynthMs ?? this.tSynthMs,
      budgetMs: budgetMs ?? this.budgetMs,
      p50Ms: p50Ms ?? this.p50Ms,
      p95Ms: p95Ms ?? this.p95Ms,
      p99Ms: p99Ms ?? this.p99Ms,
      jitterMs: jitterMs ?? this.jitterMs,
      overrunCount: overrunCount ?? this.overrunCount,
    );
  }
}
