import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/models/denoiser_state.dart';
import 'package:edge_ai_denoiser_mobile/widgets/precision_selector_card.dart';

void main() {
  group('PrecisionSelectorCard Component Tests', () {
    testWidgets('Renders FP32 mode initially with 165.8 KB RAM and 100 dB SQNR', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PrecisionSelectorCard(
              selectedPrecision: PrecisionMode.fp32,
              onPrecisionSelected: (_) {},
            ),
          ),
        ),
      );

      expect(find.text('Multi-Precision Engine'), findsOneWidget);
      expect(find.byKey(const Key('precision_chip_fp32')), findsOneWidget);
      expect(find.byKey(const Key('precision_chip_fp16')), findsOneWidget);
      expect(find.byKey(const Key('precision_chip_int8')), findsOneWidget);

      expect(find.text('165.8 KB'), findsOneWidget);
      expect(find.text('100.0%'), findsOneWidget);
      expect(find.text('100.0 dB'), findsOneWidget);
    });

    testWidgets('Tapping FP16 choice chip selects FP16 mode and invokes callback', (WidgetTester tester) async {
      PrecisionMode currentMode = PrecisionMode.fp32;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return PrecisionSelectorCard(
                  selectedPrecision: currentMode,
                  onPrecisionSelected: (mode) {
                    setState(() {
                      currentMode = mode;
                    });
                  },
                );
              },
            ),
          ),
        ),
      );

      final fp16ChipFinder = find.byKey(const Key('precision_chip_fp16'));
      await tester.tap(fp16ChipFinder);
      await tester.pump();

      expect(currentMode, equals(PrecisionMode.fp16));
      expect(find.text('82.9 KB'), findsOneWidget);
      expect(find.text('50.0%'), findsOneWidget);
      expect(find.text('73.6 dB'), findsOneWidget);
    });

    testWidgets('Tapping INT8 choice chip displays 42.6 KB and 74.3% RAM savings', (WidgetTester tester) async {
      PrecisionMode currentMode = PrecisionMode.fp32;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return PrecisionSelectorCard(
                  selectedPrecision: currentMode,
                  onPrecisionSelected: (mode) {
                    setState(() {
                      currentMode = mode;
                    });
                  },
                );
              },
            ),
          ),
        ),
      );

      final int8ChipFinder = find.byKey(const Key('precision_chip_int8'));
      await tester.tap(int8ChipFinder);
      await tester.pump();

      expect(currentMode, equals(PrecisionMode.int8));
      expect(find.text('42.6 KB'), findsOneWidget);
      expect(find.text('25.7%'), findsOneWidget);
      expect(find.text('39.9 dB'), findsOneWidget);
    });
  });
}
