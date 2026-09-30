import 'package:flutter/material.dart';

/// Top-Level Hero Suite Card for Live Subtitles & Multilingual Translation.
/// Supports 9 Indian and 6 Global Languages with instant client-side lookup.
class SubtitleHeroCard extends StatefulWidget {
  final String sourceText;
  final String translatedText;
  final String selectedLanguage;
  final ValueChanged<String?> onLanguageChanged;
  final ValueChanged<String> onTranslateSubmitted;
  final VoidCallback onClear;

  const SubtitleHeroCard({
    super.key,
    required this.sourceText,
    required this.translatedText,
    required this.selectedLanguage,
    required this.onLanguageChanged,
    required this.onTranslateSubmitted,
    required this.onClear,
  });

  @override
  State<SubtitleHeroCard> createState() => _SubtitleHeroCardState();
}

class _SubtitleHeroCardState extends State<SubtitleHeroCard> {
  late final TextEditingController _controller;

  static const List<String> supportedLanguages = [
    'Hindi (hi)',
    'Tamil (ta)',
    'Telugu (te)',
    'Bengali (bn)',
    'Marathi (mr)',
    'Gujarati (gu)',
    'Kannada (kn)',
    'Malayalam (ml)',
    'Punjabi (pa)',
    'Spanish (es)',
    'French (fr)',
    'German (de)',
    'Japanese (ja)',
    'Chinese (zh)',
    'Italian (it)',
  ];

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _submitTranslation() {
    final text = _controller.text.trim();
    if (text.isNotEmpty) {
      widget.onTranslateSubmitted(text);
    }
  }

  void _clearInput() {
    _controller.clear();
    widget.onClear();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header & Language Selector
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.translate, color: Colors.blueAccent),
                    const SizedBox(width: 8),
                    Text(
                      'Live Multilingual Subtitles',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                DropdownButton<String>(
                  key: const Key('target_language_dropdown'),
                  value: widget.selectedLanguage,
                  underline: const SizedBox(),
                  items: supportedLanguages.map((lang) {
                    return DropdownMenuItem<String>(
                      value: lang,
                      child: Text(lang, style: const TextStyle(fontSize: 12)),
                    );
                  }).toList(),
                  onChanged: widget.onLanguageChanged,
                ),
              ],
            ),
            const Divider(),

            // Glowing Dual-Line Subtitles Container
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.blueGrey.shade900.withOpacity(0.6),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blueAccent.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'ENGLISH SOURCE (ASR):',
                    style: TextStyle(fontSize: 10, color: Colors.white54, letterSpacing: 0.8),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    widget.sourceText,
                    key: const Key('source_subtitle_text'),
                    style: const TextStyle(
                      fontSize: 14,
                      color: Colors.white70,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'TRANSLATED (${widget.selectedLanguage.toUpperCase()}):',
                    style: const TextStyle(fontSize: 10, color: Colors.cyanAccent, letterSpacing: 0.8),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    widget.translatedText,
                    key: const Key('target_subtitle_text'),
                    style: const TextStyle(
                      fontSize: 16,
                      color: Colors.cyanAccent,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),

            // Interactive Live Translate Input & Actions
            Row(
              children: [
                Expanded(
                  child: TextField(
                    key: const Key('live_translate_input'),
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Type custom phrase to translate...',
                      hintStyle: const TextStyle(fontSize: 13, color: Colors.white38),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                    onSubmitted: (_) => _submitTranslation(),
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  key: const Key('translate_action_btn'),
                  onPressed: _submitTranslation,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blueAccent,
                    foregroundColor: Colors.white,
                  ),
                  child: const Text('Translate'),
                ),
                const SizedBox(width: 6),
                IconButton(
                  key: const Key('clear_translate_btn'),
                  icon: const Icon(Icons.clear, size: 20),
                  tooltip: 'Clear Input',
                  onPressed: _clearInput,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
