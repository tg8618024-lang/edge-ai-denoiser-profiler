import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/widgets/telemetry_event_list.dart';

void main() {
  group('TelemetryEventList Component Tests', () {
    testWidgets('Renders frame stream list header and initial items', (WidgetTester tester) async {
      final sampleEvents = List.generate(
        40,
        (i) => 'Frame #$i: Denoised | Latency: 0.245 ms',
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: TelemetryEventList(events: sampleEvents),
          ),
        ),
      );

      // Verify header and badge
      expect(find.text('Real-Time Frame Stream Log'), findsOneWidget);
      expect(find.text('40 frames'), findsOneWidget);
      expect(find.byKey(const Key('telemetry_event_list')), findsOneWidget);

      // Verify item 0 is rendered
      expect(find.byKey(const Key('event_item_0')), findsOneWidget);
      expect(find.text('Frame #0: Denoised | Latency: 0.245 ms'), findsOneWidget);
    });

    testWidgets('Dynamic list scrolling with scrollUntilVisible renders off-screen frame item', (WidgetTester tester) async {
      final sampleEvents = List.generate(
        40,
        (i) => 'Frame #$i: Denoised | Latency: 0.245 ms',
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: TelemetryEventList(events: sampleEvents),
          ),
        ),
      );

      final targetItemFinder = find.byKey(const Key('event_item_25'));
      final listFinder = find.byKey(const Key('telemetry_event_list'));

      // Scroll ListView dynamically until off-screen item 25 is rendered
      await tester.scrollUntilVisible(
        targetItemFinder,
        300.0,
        scrollable: listFinder,
      );
      await tester.pumpAndSettle();

      // Verify target item is now rendered and visible
      expect(targetItemFinder, findsOneWidget);
      expect(find.text('Frame #25: Denoised | Latency: 0.245 ms'), findsOneWidget);
    });

    testWidgets('Handles empty event stream gracefully with 0 frames badge', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: TelemetryEventList(events: []),
          ),
        ),
      );

      expect(find.text('Real-Time Frame Stream Log'), findsOneWidget);
      expect(find.text('0 frames'), findsOneWidget);
      expect(find.byKey(const Key('event_item_0')), findsNothing);
    });
  });
}
