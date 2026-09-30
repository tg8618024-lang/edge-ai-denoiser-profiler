import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/models/latency_metrics.dart';
import 'package:edge_ai_denoiser_mobile/widgets/latency_profiler_card.dart';

void main() {
  group('LatencyProfilerCard Component Tests', () {
    testWidgets('Renders 3 isolated stages and within-budget headroom correctly', (WidgetTester tester) async {
      final defaultMetrics = LatencyMetrics.defaultReference();

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: LatencyProfilerCard(
              metrics: defaultMetrics,
              onResetMetrics: () {},
            ),
          ),
        ),
      );

      // Verify title and stages
      expect(find.text('Hardware Latency Profiler'), findsOneWidget);
      expect(find.text('Stage 1: Pre-processing (Hann + rFFT)'), findsOneWidget);
      expect(find.text('Stage 2: Tensor Compute (GRUMaskNet / Wiener)'), findsOneWidget);
      expect(find.text('Stage 3: Output Synthesis (irFFT + Overlap-Add)'), findsOneWidget);

      // Verify stage ms text
      expect(find.text('0.043 ms'), findsOneWidget);
      expect(find.text('0.167 ms'), findsOneWidget);
      expect(find.text('0.036 ms'), findsOneWidget);

      // Verify total latency text and headroom
      expect(find.byKey(const Key('total_latency_text')), findsOneWidget);
      expect(find.byKey(const Key('headroom_chip')), findsOneWidget);
      expect(find.textContaining('98.5% Headroom'), findsOneWidget);

      // Verify rolling percentiles
      expect(find.text('P50 (Median)'), findsOneWidget);
      expect(find.text('0.245 ms'), findsOneWidget);
      expect(find.text('P95'), findsOneWidget);
      expect(find.text('0.312 ms'), findsOneWidget);

      // Normal conditions must NOT show the overrun warning banner
      expect(find.byKey(const Key('overrun_warning_banner')), findsNothing);
    });

    testWidgets('Renders overrun warning banner when latency exceeds 16.0 ms frame budget', (WidgetTester tester) async {
      final overrunMetrics = LatencyMetrics.overrunExample();

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: LatencyProfilerCard(
              metrics: overrunMetrics,
              onResetMetrics: () {},
            ),
          ),
        ),
      );

      // Overrun banner MUST be visible
      expect(find.byKey(const Key('overrun_warning_banner')), findsOneWidget);
      expect(find.textContaining('Frame Budget Overrun!'), findsOneWidget);
      expect(find.textContaining('18.500ms > 16.0ms budget'), findsOneWidget);
      expect(find.textContaining('5 overruns'), findsOneWidget);

      // Headroom should display 0.0%
      expect(find.textContaining('0.0% Headroom'), findsOneWidget);
    });

    testWidgets('Tapping reset metrics button triggers onResetMetrics callback', (WidgetTester tester) async {
      bool resetCalled = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: LatencyProfilerCard(
              metrics: LatencyMetrics.defaultReference(),
              onResetMetrics: () {
                resetCalled = true;
              },
            ),
          ),
        ),
      );

      final resetBtnFinder = find.byKey(const Key('reset_metrics_btn'));
      expect(resetBtnFinder, findsOneWidget);

      await tester.tap(resetBtnFinder);
      await tester.pump();

      expect(resetCalled, isTrue);
    });
  });
}
