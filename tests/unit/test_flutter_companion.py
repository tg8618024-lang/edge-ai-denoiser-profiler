"""Unit Tests for Flutter Mobile Companion App & Widget Test Suite Integrity.

Authority: flutter-add-widget-test skill (SKILL.md) & ORIGINAL_REQUEST.md
Tests:
- pubspec.yaml configuration and flutter_test dev dependency.
- Dart models and widgets structural integrity, keys, and state management.
- Flutter widget test suites adherence to the 9-step WidgetTester workflow.
- Complete coverage of Finders, Matchers, Gestures (tap, drag, enterText, scrollUntilVisible),
  and Pump cycles (pump, pumpAndSettle).
"""

import os
import re
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOBILE_DIR = os.path.join(PROJECT_ROOT, "mobile_app")
LIB_DIR = os.path.join(MOBILE_DIR, "lib")
TEST_DIR = os.path.join(MOBILE_DIR, "test")


class TestFlutterProjectConfiguration:
    def test_pubspec_file_exists_and_valid(self):
        """Verify mobile_app/pubspec.yaml exists and defines proper metadata."""
        pubspec_path = os.path.join(MOBILE_DIR, "pubspec.yaml")
        assert os.path.isfile(pubspec_path), f"pubspec.yaml missing at {pubspec_path}"

        with open(pubspec_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "name: edge_ai_denoiser_mobile" in content
        assert "environment:" in content
        assert "sdk:" in content
        assert "flutter_test:" in content
        assert "dev_dependencies:" in content
        assert "uses-material-design: true" in content


class TestFlutterSourceCodeStructure:
    def test_all_required_dart_files_exist(self):
        """Verify all core models, widgets, and screens are present."""
        expected_files = [
            "lib/main.dart",
            "lib/models/latency_metrics.dart",
            "lib/models/denoiser_state.dart",
            "lib/widgets/denoiser_control_card.dart",
            "lib/widgets/latency_profiler_card.dart",
            "lib/widgets/precision_selector_card.dart",
            "lib/widgets/subtitle_hero_card.dart",
            "lib/widgets/session_recorder_card.dart",
            "lib/widgets/telemetry_event_list.dart",
            "lib/screens/denoiser_dashboard_screen.dart",
        ]
        for rel_path in expected_files:
            full_path = os.path.join(MOBILE_DIR, rel_path)
            assert os.path.isfile(full_path), f"Expected Dart file missing: {rel_path}"

    def test_latency_metrics_model_integrity(self):
        """Verify 3-stage breakdown fields and headroom calculation in LatencyMetrics."""
        model_path = os.path.join(LIB_DIR, "models", "latency_metrics.dart")
        with open(model_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "tPreMs" in code
        assert "tTensorMs" in code
        assert "tSynthMs" in code
        assert "budgetMs" in code
        assert "totalMs => tPreMs + tTensorMs + tSynthMs" in code
        assert "isOverrun => totalMs > budgetMs" in code
        assert "headroomPercent" in code
        assert "p50Ms" in code
        assert "p95Ms" in code
        assert "p99Ms" in code

    def test_denoiser_state_model_integrity(self):
        """Verify precision modes and noise profiles in DenoiserState."""
        state_path = os.path.join(LIB_DIR, "models", "denoiser_state.dart")
        with open(state_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "enum PrecisionMode" in code
        assert "fp32" in code
        assert "fp16" in code
        assert "int8" in code
        assert "165.8" in code
        assert "42.6" in code
        assert "enum NoiseProfile" in code
        assert "white" in code
        assert "pink" in code
        assert "drone" in code
        assert "rfStatic" in code


class TestFlutterWidgetTestSuites:
    """Verifies compliance with flutter-add-widget-test skill specification."""

    EXPECTED_TEST_FILES = [
        "widget_test.dart",
        "denoiser_control_card_test.dart",
        "latency_profiler_card_test.dart",
        "precision_selector_card_test.dart",
        "subtitle_hero_card_test.dart",
        "session_recorder_card_test.dart",
        "telemetry_event_list_test.dart",
        "denoiser_dashboard_screen_test.dart",
    ]

    def test_test_files_exist_and_named_correctly(self):
        """Verify all test files are in test/ and end with _test.dart."""
        for filename in self.EXPECTED_TEST_FILES:
            full_path = os.path.join(TEST_DIR, filename)
            assert os.path.isfile(full_path), f"Widget test file missing: {filename}"
            assert filename.endswith("_test.dart"), f"Test filename must end with _test.dart: {filename}"

    def test_widget_tester_imports_and_declarations(self):
        """Step 1 & Setup: Verify package:flutter_test and testWidgets declarations."""
        for filename in self.EXPECTED_TEST_FILES:
            full_path = os.path.join(TEST_DIR, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "package:flutter_test/flutter_test.dart" in content, (
                f"{filename} must import package:flutter_test/flutter_test.dart"
            )
            assert "testWidgets(" in content, f"{filename} must contain testWidgets() calls"
            assert "WidgetTester tester" in content, f"{filename} must use WidgetTester tester parameter"

    def test_pump_widget_and_material_app_wrapping(self):
        """Step 2: Verify tester.pumpWidget() wraps widgets in MaterialApp."""
        for filename in self.EXPECTED_TEST_FILES:
            full_path = os.path.join(TEST_DIR, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "tester.pumpWidget(" in content, f"{filename} must call tester.pumpWidget"
            assert "MaterialApp(" in content or "EdgeAiDenoiserApp(" in content, (
                f"{filename} must wrap target widget in MaterialApp"
            )

    def test_finders_and_matchers_coverage(self):
        """Steps 3 & 4: Verify Finder methods and Matcher assertions."""
        all_test_content = ""
        for filename in self.EXPECTED_TEST_FILES:
            full_path = os.path.join(TEST_DIR, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                all_test_content += f.read() + "\n"

        # Finders
        assert "find.text(" in all_test_content, "Must utilize find.text Finder"
        assert "find.byKey(" in all_test_content, "Must utilize find.byKey Finder"
        assert "find.byType(" in all_test_content, "Must utilize find.byType Finder"
        assert "find.byIcon(" in all_test_content, "Must utilize find.byIcon Finder"

        # Matchers
        assert "findsOneWidget" in all_test_content, "Must utilize findsOneWidget Matcher"
        assert "findsNothing" in all_test_content, "Must utilize findsNothing Matcher"

    def test_interaction_modes_and_gestures(self):
        """Steps 5-7: Verify tap, enterText, drag, scrollUntilVisible, and pump cycles."""
        all_test_content = ""
        for filename in self.EXPECTED_TEST_FILES:
            full_path = os.path.join(TEST_DIR, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                all_test_content += f.read() + "\n"

        # Button taps
        assert "tester.tap(" in all_test_content, "Must utilize tester.tap for standard state changes"
        # Text input
        assert "tester.enterText(" in all_test_content, "Must utilize tester.enterText for form inputs"
        # Slider drag gesture
        assert "tester.drag(" in all_test_content, "Must utilize tester.drag for slider gestures"
        # Dynamic list scrolling
        assert "tester.scrollUntilVisible(" in all_test_content, (
            "Must utilize tester.scrollUntilVisible for dynamic list items"
        )

        # Single frame rebuild and animation settling
        assert "tester.pump()" in all_test_content, "Must call tester.pump() for single-frame rebuilds"
        assert "tester.pumpAndSettle()" in all_test_content, (
            "Must call tester.pumpAndSettle() for animations and transitions"
        )

    def test_dropdown_and_viewport_handling(self):
        """Verify dropdown interaction testing and viewport clipping prevention."""
        control_test = os.path.join(TEST_DIR, "denoiser_control_card_test.dart")
        with open(control_test, "r", encoding="utf-8") as f:
            control_code = f.read()
        assert "noise_profile_dropdown" in control_code
        assert "Drone / Fan Hum" in control_code

        subtitle_test = os.path.join(TEST_DIR, "subtitle_hero_card_test.dart")
        with open(subtitle_test, "r", encoding="utf-8") as f:
            subtitle_code = f.read()
        assert "target_language_dropdown" in subtitle_code
        assert "Tamil (ta)" in subtitle_code

        dashboard_test = os.path.join(TEST_DIR, "denoiser_dashboard_screen_test.dart")
        with open(dashboard_test, "r", encoding="utf-8") as f:
            dashboard_code = f.read()
        assert "tester.ensureVisible" in dashboard_code
        assert "tester.view.physicalSize" in dashboard_code

    def test_overrun_edge_case_and_error_paths(self):
        """Verify edge case testing for frame overruns and empty text submissions."""
        latency_test = os.path.join(TEST_DIR, "latency_profiler_card_test.dart")
        with open(latency_test, "r", encoding="utf-8") as f:
            code = f.read()

        assert "overrun_warning_banner" in code, "Must test overrun warning banner display"
        assert "overrunExample" in code, "Must test simulated overrun condition"

        subtitle_test = os.path.join(TEST_DIR, "subtitle_hero_card_test.dart")
        with open(subtitle_test, "r", encoding="utf-8") as f:
            code = f.read()

        assert "whitespace or empty string" in code, "Must test empty input edge case"

        event_list_test = os.path.join(TEST_DIR, "telemetry_event_list_test.dart")
        with open(event_list_test, "r", encoding="utf-8") as f:
            event_code = f.read()
        assert "Handles empty event stream gracefully" in event_code
