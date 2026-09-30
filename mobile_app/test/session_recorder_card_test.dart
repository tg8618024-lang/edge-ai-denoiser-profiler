import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/widgets/session_recorder_card.dart';

void main() {
  group('SessionRecorderCard Component Tests', () {
    testWidgets('Renders initial idle state with Record Session button and hidden export status', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SessionRecorderCard(
              isRecording: false,
              isExported: false,
              onRecordToggled: () {},
              onExportTelemetry: () {},
            ),
          ),
        ),
      );

      // Verify header and buttons
      expect(find.text('Multi-Track Session Studio'), findsOneWidget);
      expect(find.byKey(const Key('record_session_btn')), findsOneWidget);
      expect(find.byKey(const Key('export_telemetry_btn')), findsOneWidget);
      expect(find.text('Record Session'), findsOneWidget);
      expect(find.byIcon(Icons.fiber_manual_record), findsOneWidget);
      expect(find.byIcon(Icons.file_download_outlined), findsOneWidget);

      // Export status banner should be hidden
      expect(find.byKey(const Key('telemetry_export_status')), findsNothing);
    });

    testWidgets('Tapping Record Session button triggers onRecordToggled callback', (WidgetTester tester) async {
      bool recordToggled = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return SessionRecorderCard(
                  isRecording: false,
                  isExported: false,
                  onRecordToggled: () {
                    recordToggled = true;
                  },
                  onExportTelemetry: () {},
                );
              },
            ),
          ),
        ),
      );

      final recordBtnFinder = find.byKey(const Key('record_session_btn'));
      await tester.tap(recordBtnFinder);
      await tester.pump();

      expect(recordToggled, isTrue);
    });

    testWidgets('Renders active recording state with Stop Recording text and stop icon', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SessionRecorderCard(
              isRecording: true,
              isExported: false,
              onRecordToggled: () {},
              onExportTelemetry: () {},
            ),
          ),
        ),
      );

      expect(find.text('Stop Recording'), findsOneWidget);
      expect(find.byIcon(Icons.stop), findsOneWidget);
      expect(find.text('Record Session'), findsNothing);
    });

    testWidgets('Tapping Export Telemetry button triggers onExportTelemetry callback', (WidgetTester tester) async {
      bool exportTriggered = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SessionRecorderCard(
              isRecording: false,
              isExported: false,
              onRecordToggled: () {},
              onExportTelemetry: () {
                exportTriggered = true;
              },
            ),
          ),
        ),
      );

      final exportBtnFinder = find.byKey(const Key('export_telemetry_btn'));
      await tester.tap(exportBtnFinder);
      await tester.pump();

      expect(exportTriggered, isTrue);
    });

    testWidgets('Renders export success banner when isExported is true', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SessionRecorderCard(
              isRecording: false,
              isExported: true,
              onRecordToggled: () {},
              onExportTelemetry: () {},
            ),
          ),
        ),
      );

      expect(find.byKey(const Key('telemetry_export_status')), findsOneWidget);
      expect(find.text('Telemetry & WAV Multi-Track Bundle Exported Successfully!'), findsOneWidget);
      expect(find.byIcon(Icons.check_circle_outline), findsOneWidget);
    });
  });
}
