import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/screens/denoiser_dashboard_screen.dart';

void main() {
  group('DenoiserDashboardScreen End-to-End Integration Widget Tests', () {
    testWidgets('Dashboard renders and allows recording toggle and telemetry export', (WidgetTester tester) async {
      tester.view.physicalSize = const Size(1080, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      await tester.pumpWidget(
        const MaterialApp(
          home: DenoiserDashboardScreen(),
        ),
      );

      // Verify initial title and cards
      expect(find.byKey(const Key('app_title_text')), findsOneWidget);
      expect(find.byKey(const Key('record_session_btn')), findsOneWidget);
      expect(find.byKey(const Key('export_telemetry_btn')), findsOneWidget);

      // Verify export status banner is NOT visible initially
      expect(find.byKey(const Key('telemetry_export_status')), findsNothing);

      // Tap Record Session button
      final recordBtnFinder = find.byKey(const Key('record_session_btn'));
      await tester.ensureVisible(recordBtnFinder);
      await tester.tap(recordBtnFinder);
      await tester.pump();

      // Verify recording state changed
      expect(find.text('Stop Recording'), findsOneWidget);

      // Tap Export Telemetry button
      final exportBtnFinder = find.byKey(const Key('export_telemetry_btn'));
      await tester.ensureVisible(exportBtnFinder);
      await tester.tap(exportBtnFinder);
      await tester.pumpAndSettle();

      // Verify export status banner is now visible
      expect(find.byKey(const Key('telemetry_export_status')), findsOneWidget);
      expect(find.textContaining('Exported Successfully!'), findsOneWidget);
    });

    testWidgets('Scrolls through dynamic telemetry event list until off-screen item is visible', (WidgetTester tester) async {
      tester.view.physicalSize = const Size(1080, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      await tester.pumpWidget(
        const MaterialApp(
          home: DenoiserDashboardScreen(),
        ),
      );

      final scrollableListFinder = find.byKey(const Key('telemetry_event_list'));
      await tester.ensureVisible(scrollableListFinder);
      await tester.pumpAndSettle();

      // Item 0 should be visible initially in the event list
      expect(find.byKey(const Key('event_item_0')), findsOneWidget);

      // Item 25 is initially off-screen in the scrollable list
      final targetItemFinder = find.byKey(const Key('event_item_25'));

      // Use scrollUntilVisible to scroll the ListView until target is rendered
      await tester.scrollUntilVisible(
        targetItemFinder,
        300.0,
        scrollable: scrollableListFinder,
      );
      await tester.pumpAndSettle();

      // Verify item 25 is now found and visible in the widget tree
      expect(targetItemFinder, findsOneWidget);
      expect(find.textContaining('Frame #25: Denoised'), findsOneWidget);
    });

    testWidgets('End-to-end translation interaction in full dashboard context', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: DenoiserDashboardScreen(),
        ),
      );

      // Enter phrase into translate input
      final inputFinder = find.byKey(const Key('live_translate_input'));
      await tester.enterText(inputFinder, 'Audio stream clean');
      await tester.pump();

      // Tap translate
      final translateBtn = find.byKey(const Key('translate_action_btn'));
      await tester.tap(translateBtn);
      await tester.pump();

      // Verify updated subtitle texts
      expect(find.byKey(const Key('source_subtitle_text')), findsOneWidget);
      expect(find.text('Audio stream clean'), findsOneWidget);
      expect(find.text('ध्वनि बफर पर न्यूरल ऑडियो वृद्धि सक्रिय।'), findsOneWidget);
    });
  });
}
