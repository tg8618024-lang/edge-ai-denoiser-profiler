import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/widgets/subtitle_hero_card.dart';

void main() {
  group('SubtitleHeroCard Component Tests', () {
    testWidgets('Renders bilingual subtitles and live translate controls', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SubtitleHeroCard(
              sourceText: 'Neural audio streaming active on 16.0 ms frame buffer.',
              translatedText: '16.0 एमएस फ्रेम बफर पर न्यूरल ऑडियो स्ट्रीमिंग सक्रिय।',
              selectedLanguage: 'Hindi (hi)',
              onLanguageChanged: (_) {},
              onTranslateSubmitted: (_) {},
              onClear: () {},
            ),
          ),
        ),
      );

      expect(find.text('Live Multilingual Subtitles'), findsOneWidget);
      expect(find.byKey(const Key('source_subtitle_text')), findsOneWidget);
      expect(find.byKey(const Key('target_subtitle_text')), findsOneWidget);
      expect(find.text('Neural audio streaming active on 16.0 ms frame buffer.'), findsOneWidget);
      expect(find.text('16.0 एमएस फ्रेम बफर पर न्यूरल ऑडियो स्ट्रीमिंग सक्रिय।'), findsOneWidget);
      expect(find.byKey(const Key('live_translate_input')), findsOneWidget);
      expect(find.byKey(const Key('translate_action_btn')), findsOneWidget);
      expect(find.byKey(const Key('clear_translate_btn')), findsOneWidget);
      expect(find.byKey(const Key('target_language_dropdown')), findsOneWidget);
    });

    testWidgets('Entering text into live translate input and tapping Translate invokes callback', (WidgetTester tester) async {
      String? submittedPhrase;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SubtitleHeroCard(
              sourceText: 'Default text',
              translatedText: 'डिफ़ॉल्ट टेक्स्ट',
              selectedLanguage: 'Hindi (hi)',
              onLanguageChanged: (_) {},
              onTranslateSubmitted: (phrase) {
                submittedPhrase = phrase;
              },
              onClear: () {},
            ),
          ),
        ),
      );

      // Locate TextField and enter custom text
      final inputFinder = find.byKey(const Key('live_translate_input'));
      await tester.enterText(inputFinder, 'Real-time noise suppression test');
      await tester.pump();

      // Tap the Translate button
      final translateBtnFinder = find.byKey(const Key('translate_action_btn'));
      await tester.tap(translateBtnFinder);
      await tester.pump();

      // Verify callback triggered with entered text
      expect(submittedPhrase, equals('Real-time noise suppression test'));
    });

    testWidgets('Tapping clear button empties text input and triggers onClear callback', (WidgetTester tester) async {
      bool clearCalled = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SubtitleHeroCard(
              sourceText: 'Some source text',
              translatedText: 'अनुवादित टेक्स्ट',
              selectedLanguage: 'Hindi (hi)',
              onLanguageChanged: (_) {},
              onTranslateSubmitted: (_) {},
              onClear: () {
                clearCalled = true;
              },
            ),
          ),
        ),
      );

      // Enter text first
      final inputFinder = find.byKey(const Key('live_translate_input'));
      await tester.enterText(inputFinder, 'Temporary query');
      await tester.pump();

      // Tap clear button
      final clearBtnFinder = find.byKey(const Key('clear_translate_btn'));
      await tester.tap(clearBtnFinder);
      await tester.pump();

      expect(clearCalled, isTrue);
      expect(find.text('Temporary query'), findsNothing);
    });

    testWidgets('Submitting whitespace or empty string does not trigger translation callback', (WidgetTester tester) async {
      bool submitCalled = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SubtitleHeroCard(
              sourceText: 'Default text',
              translatedText: 'डिफ़ॉल्ट टेक्स्ट',
              selectedLanguage: 'Hindi (hi)',
              onLanguageChanged: (_) {},
              onTranslateSubmitted: (_) {
                submitCalled = true;
              },
              onClear: () {},
            ),
          ),
        ),
      );

      // Enter only spaces
      await tester.enterText(find.byKey(const Key('live_translate_input')), '   ');
      await tester.pump();

      // Tap translate button
      await tester.tap(find.byKey(const Key('translate_action_btn')));
      await tester.pump();

      // Callback should NOT be invoked
      expect(submitCalled, isFalse);
    });

    testWidgets('Selecting a target language from dropdown invokes onLanguageChanged', (WidgetTester tester) async {
      String currentLanguage = 'Hindi (hi)';

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return SubtitleHeroCard(
                  sourceText: 'Live speech frame',
                  translatedText: 'लाइव स्पीच फ्रेम',
                  selectedLanguage: currentLanguage,
                  onLanguageChanged: (newLang) {
                    setState(() {
                      if (newLang != null) currentLanguage = newLang;
                    });
                  },
                  onTranslateSubmitted: (_) {},
                  onClear: () {},
                );
              },
            ),
          ),
        ),
      );

      // Tap dropdown to open menu
      final dropdownFinder = find.byKey(const Key('target_language_dropdown'));
      await tester.tap(dropdownFinder);
      await tester.pumpAndSettle();

      // Tap 'Tamil (ta)' in the popup list
      final itemFinder = find.text('Tamil (ta)').last;
      await tester.tap(itemFinder);
      await tester.pumpAndSettle();

      // Verify callback updated language
      expect(currentLanguage, equals('Tamil (ta)'));
    });

    testWidgets('Submitting translation via keyboard enter key invokes callback', (WidgetTester tester) async {
      String? submittedText;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SubtitleHeroCard(
              sourceText: 'Initial text',
              translatedText: 'प्रारंभिक पाठ',
              selectedLanguage: 'Hindi (hi)',
              onLanguageChanged: (_) {},
              onTranslateSubmitted: (text) {
                submittedText = text;
              },
              onClear: () {},
            ),
          ),
        ),
      );

      // Enter text and trigger submit via text input action
      final inputFinder = find.byKey(const Key('live_translate_input'));
      await tester.enterText(inputFinder, 'Edge neural denoising benchmark');
      await tester.testTextInput.receiveAction(TextInputAction.done);
      await tester.pump();

      expect(submittedText, equals('Edge neural denoising benchmark'));
    });
  });
}
