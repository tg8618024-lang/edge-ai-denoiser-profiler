# 📱 Edge AI Audio Denoiser — Flutter Mobile Companion & Widget Testing Guide

## 1. Overview & Architecture

The **Edge AI Audio Denoiser Flutter Mobile Companion** (`mobile_app/`) provides an edge-optimized cross-platform dashboard for real-time acoustic speech enhancement, hardware latency profiling, multi-precision quantization management, and multilingual translation.

The architecture is partitioned into clean, modular Flutter components:

```
mobile_app/
├── pubspec.yaml                          # Package spec declaring flutter_test dev_dependency
├── lib/
│   ├── main.dart                         # Entrypoint & dark theme configuration
│   ├── models/
│   │   ├── latency_metrics.dart          # 3-stage latency telemetry & budget headroom model
│   │   └── denoiser_state.dart           # Global state, precision modes & noise profiles
│   ├── widgets/
│   │   ├── denoiser_control_card.dart    # A/B toggle, bypass switch, mute button, suppression slider
│   │   ├── latency_profiler_card.dart    # 3-stage breakdown, P50/P95/P99 percentiles, overrun banner
│   │   ├── precision_selector_card.dart  # FP32 / FP16 / INT8 quantization chips & RAM SQNR
│   │   ├── subtitle_hero_card.dart       # Glowing bilingual subtitles & live translation input
│   │   ├── session_recorder_card.dart    # Multi-track WAV/telemetry recording & export
│   │   └── telemetry_event_list.dart     # Dynamic frame stream log (scrollUntilVisible)
│   └── screens/
│       └── denoiser_dashboard_screen.dart# Master dashboard assembling all cards
└── test/
    ├── widget_test.dart                  # App smoke test & top-level widget tree verification
    ├── denoiser_control_card_test.dart   # Toggle, slider drag, mute, dropdown interaction tests
    ├── latency_profiler_card_test.dart   # 3-stage isolation, budget headroom, overrun warning tests
    ├── precision_selector_card_test.dart # Multi-precision quantization mode switching tests
    ├── subtitle_hero_card_test.dart      # Bilingual display, text input, translate & clear tests
    ├── session_recorder_card_test.dart   # Studio recording toggle and telemetry export tests
    ├── telemetry_event_list_test.dart    # Dynamic list scrolling (scrollUntilVisible) & empty state
    └── denoiser_dashboard_screen_test.dart# Full integration, session recording & scrollUntilVisible tests
```

---

## 2. Widget Testing Implementation (`flutter-add-widget-test`)

All test suites strictly adhere to the 9-step workflow and interaction rules specified in `flutter-add-widget-test`:

### Workflow Adherence:
1. **Define the test**: Uses `testWidgets('description', (WidgetTester tester) async { ... })`.
2. **Build the widget**: Calls `await tester.pumpWidget(MyWidget())` wrapped in `MaterialApp`.
3. **Locate elements**: Utilizes `find.text()`, `find.byKey()`, `find.byType()`, and `find.byIcon()`.
4. **Verify initial state**: Uses `expect(finder, findsOneWidget)` and `expect(finder, findsNothing)`.
5. **Simulate interactions**:
   - Button & Switch taps: `await tester.tap(finder)`.
   - Text form input: `await tester.enterText(textFieldFinder, 'Input string')`.
   - Continuous sliders: `await tester.drag(sliderFinder, Offset(100.0, 0.0))`.
   - Dynamic list items: `await tester.scrollUntilVisible(itemFinder, 300.0, scrollable: listFinder)`.
6. **Rebuild the tree**:
   - Single frame state rebuild: `await tester.pump()`.
   - Asynchronous animations and settle: `await tester.pumpAndSettle()`.
7. **Verify updated state**: Asserts UI transitions, icon toggles, and callback invocations.

---

## 3. Running Widget Tests

To execute the test suite in a Flutter-enabled environment:

```bash
cd mobile_app
flutter test
```

To run individual widget test suites:

```bash
flutter test test/widget_test.dart
flutter test test/denoiser_control_card_test.dart
flutter test test/latency_profiler_card_test.dart
flutter test test/precision_selector_card_test.dart
flutter test test/subtitle_hero_card_test.dart
flutter test test/session_recorder_card_test.dart
flutter test test/telemetry_event_list_test.dart
flutter test test/denoiser_dashboard_screen_test.dart
```

To run the repository's automated companion verification test:

```bash
pytest tests/unit/test_flutter_companion.py -v
```
