import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/main.dart';
import 'package:edge_ai_denoiser_mobile/screens/denoiser_dashboard_screen.dart';
import 'package:edge_ai_denoiser_mobile/widgets/denoiser_control_card.dart';
import 'package:edge_ai_denoiser_mobile/widgets/latency_profiler_card.dart';
import 'package:edge_ai_denoiser_mobile/widgets/precision_selector_card.dart';
import 'package:edge_ai_denoiser_mobile/widgets/subtitle_hero_card.dart';
import 'package:edge_ai_denoiser_mobile/widgets/session_recorder_card.dart';
import 'package:edge_ai_denoiser_mobile/widgets/telemetry_event_list.dart';

void main() {
  group('EdgeAiDenoiserApp Main Widget Tests', () {
    testWidgets('App renders main dashboard with all cards in initial state', (WidgetTester tester) async {
      // 1. Build the root app widget
      await tester.pumpWidget(const EdgeAiDenoiserApp());

      // 2. Verify initial UI state and title
      expect(find.byKey(const Key('app_title_text')), findsOneWidget);
      expect(find.text('Edge AI Audio Denoiser & Profiler'), findsOneWidget);
      expect(find.text('16kHz / 16.0ms Real-Time'), findsOneWidget);

      // 3. Verify presence of all major modular cards in the dashboard
      expect(find.byType(DenoiserDashboardScreen), findsOneWidget);
      expect(find.byType(SubtitleHeroCard), findsOneWidget);
      expect(find.byType(DenoiserControlCard), findsOneWidget);
      expect(find.byType(LatencyProfilerCard), findsOneWidget);
      expect(find.byType(PrecisionSelectorCard), findsOneWidget);
      expect(find.byType(SessionRecorderCard), findsOneWidget);
      expect(find.byType(TelemetryEventList), findsOneWidget);

      // 4. Verify initial default metrics are displayed
      expect(find.text('Denoised (Clean Voice)'), findsOneWidget);
      expect(find.text('FP32'), findsOneWidget);
      expect(find.text('165.8 KB'), findsOneWidget);

      // 5. Verify MaterialApp properties and Dark Material 3 theme configuration
      final materialAppFinder = find.byType(MaterialApp);
      expect(materialAppFinder, findsOneWidget);
      final MaterialApp app = tester.widget(materialAppFinder);
      expect(app.themeMode, equals(ThemeMode.dark));
      expect(app.darkTheme?.useMaterial3, isTrue);
    });
  });
}
