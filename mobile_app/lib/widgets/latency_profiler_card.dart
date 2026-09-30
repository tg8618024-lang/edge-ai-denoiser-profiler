import 'package:flutter/material.dart';
import '../models/latency_metrics.dart';

/// NVIDIA-style 3-stage Hardware Latency Profiler card.
/// Displays millisecond telemetry across Pre-processing, Tensor Compute,
/// and Output Synthesis, along with budget headroom and rolling percentiles.
class LatencyProfilerCard extends StatelessWidget {
  final LatencyMetrics metrics;
  final VoidCallback onResetMetrics;

  const LatencyProfilerCard({
    super.key,
    required this.metrics,
    required this.onResetMetrics,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isOverrun = metrics.isOverrun;
    final totalMsStr = metrics.totalMs.toStringAsFixed(3);
    final budgetMsStr = metrics.budgetMs.toStringAsFixed(1);
    final headroomStr = metrics.headroomPercent.toStringAsFixed(1);

    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title & Reset Action
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.timer_outlined, color: Colors.greenAccent),
                    const SizedBox(width: 8),
                    Text(
                      'Hardware Latency Profiler',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                TextButton.icon(
                  key: const Key('reset_metrics_btn'),
                  onPressed: onResetMetrics,
                  icon: const Icon(Icons.refresh, size: 16),
                  label: const Text('Reset'),
                ),
              ],
            ),
            const Divider(),

            // Overrun Warning Banner (if latency exceeds budget)
            if (isOverrun)
              Container(
                key: const Key('overrun_warning_banner'),
                margin: const EdgeInsets.only(bottom: 12.0),
                padding: const EdgeInsets.all(10.0),
                decoration: BoxDecoration(
                  color: Colors.redAccent.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.redAccent),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning_amber_rounded, color: Colors.redAccent),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Frame Budget Overrun! Latency: ${totalMsStr}ms > ${budgetMsStr}ms budget (${metrics.overrunCount} overruns)',
                        style: const TextStyle(
                          color: Colors.redAccent,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

            // Total Latency & Headroom Summary
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Total Frame Latency', style: TextStyle(color: Colors.white70)),
                    Text(
                      '$totalMsStr ms / $budgetMsStr ms',
                      key: const Key('total_latency_text'),
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: isOverrun ? Colors.redAccent : Colors.greenAccent,
                      ),
                    ),
                  ],
                ),
                Chip(
                  key: const Key('headroom_chip'),
                  avatar: Icon(
                    isOverrun ? Icons.arrow_downward : Icons.speed,
                    color: isOverrun ? Colors.redAccent : Colors.greenAccent,
                    size: 18,
                  ),
                  label: Text(
                    '$headroomStr% Headroom',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: isOverrun ? Colors.redAccent : Colors.greenAccent,
                    ),
                  ),
                  backgroundColor: (isOverrun ? Colors.redAccent : Colors.greenAccent).withOpacity(0.15),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // 3-Stage Progress Breakdown Bars
            _buildStageBar(
              label: 'Stage 1: Pre-processing (Hann + rFFT)',
              ms: metrics.tPreMs,
              fraction: (metrics.tPreMs / metrics.budgetMs).clamp(0.0, 1.0),
              color: Colors.blueAccent,
            ),
            const SizedBox(height: 8),
            _buildStageBar(
              label: 'Stage 2: Tensor Compute (GRUMaskNet / Wiener)',
              ms: metrics.tTensorMs,
              fraction: (metrics.tTensorMs / metrics.budgetMs).clamp(0.0, 1.0),
              color: Colors.purpleAccent,
            ),
            const SizedBox(height: 8),
            _buildStageBar(
              label: 'Stage 3: Output Synthesis (irFFT + Overlap-Add)',
              ms: metrics.tSynthMs,
              fraction: (metrics.tSynthMs / metrics.budgetMs).clamp(0.0, 1.0),
              color: Colors.tealAccent,
            ),
            const SizedBox(height: 16),

            // Rolling Percentile Telemetry
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black26,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildStatColumn('P50 (Median)', '${metrics.p50Ms.toStringAsFixed(3)} ms'),
                  _buildStatColumn('P95', '${metrics.p95Ms.toStringAsFixed(3)} ms'),
                  _buildStatColumn('P99', '${metrics.p99Ms.toStringAsFixed(3)} ms'),
                  _buildStatColumn('Jitter', '${metrics.jitterMs.toStringAsFixed(3)} ms'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStageBar({
    required String label,
    required double ms,
    required double fraction,
    required Color color,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontSize: 12, color: Colors.white70)),
            Text('${ms.toStringAsFixed(3)} ms', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: color)),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: fraction,
            backgroundColor: Colors.white12,
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }

  Widget _buildStatColumn(String label, String value) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.white54)),
        const SizedBox(height: 2),
        Text(value, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white)),
      ],
    );
  }
}
