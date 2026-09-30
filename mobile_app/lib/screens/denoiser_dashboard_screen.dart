import 'package:flutter/material.dart';
import '../models/denoiser_state.dart';
import '../models/latency_metrics.dart';
import '../widgets/denoiser_control_card.dart';
import '../widgets/latency_profiler_card.dart';
import '../widgets/precision_selector_card.dart';
import '../widgets/subtitle_hero_card.dart';
import '../widgets/session_recorder_card.dart';
import '../widgets/telemetry_event_list.dart';

/// Main Edge AI Audio Denoiser & Latency Profiler Dashboard Screen.
class DenoiserDashboardScreen extends StatefulWidget {
  const DenoiserDashboardScreen({super.key});

  @override
  State<DenoiserDashboardScreen> createState() => _DenoiserDashboardScreenState();
}

class _DenoiserDashboardScreenState extends State<DenoiserDashboardScreen> {
  late DenoiserState _state;
  bool _isExported = false;

  @override
  void initState() {
    super.initState();
    _state = DenoiserState.initial();
  }

  void _onDenoisingToggled(bool value) {
    setState(() {
      _state = _state.copyWith(isDenoisingActive: value);
    });
  }

  void _onMuteToggled() {
    setState(() {
      _state = _state.copyWith(isMuted: !_state.isMuted);
    });
  }

  void _onSuppressionChanged(double value) {
    setState(() {
      _state = _state.copyWith(suppressionIntensity: value);
    });
  }

  void _onProfileChanged(NoiseProfile? profile) {
    if (profile != null) {
      setState(() {
        _state = _state.copyWith(noiseProfile: profile);
      });
    }
  }

  void _onPrecisionSelected(PrecisionMode mode) {
    setState(() {
      _state = _state.copyWith(precision: mode);
    });
  }

  void _onResetMetrics() {
    setState(() {
      _state = _state.copyWith(
        metrics: LatencyMetrics.defaultReference(),
      );
    });
  }

  void _onLanguageChanged(String? language) {
    if (language != null) {
      setState(() {
        _state = _state.copyWith(
          targetLanguage: language,
          translatedText: _translateForLanguage(_state.sourceText, language),
        );
      });
    }
  }

  void _onTranslateSubmitted(String text) {
    setState(() {
      _state = _state.copyWith(
        sourceText: text,
        translatedText: _translateForLanguage(text, _state.targetLanguage),
      );
    });
  }

  void _onClearTranslation() {
    setState(() {
      _state = _state.copyWith(
        sourceText: '',
        translatedText: '',
      );
    });
  }

  void _onRecordToggled() {
    setState(() {
      _state = _state.copyWith(isRecording: !_state.isRecording);
    });
  }

  void _onExportTelemetry() {
    setState(() {
      _isExported = true;
    });
  }

  String _translateForLanguage(String text, String lang) {
    if (lang.contains('Hindi')) {
      return 'ध्वनि बफर पर न्यूरल ऑडियो वृद्धि सक्रिय।';
    } else if (lang.contains('Tamil')) {
      return 'நரம்பியல் ஆடியோ மேம்பாடு செயலில் உள்ளது.';
    } else if (lang.contains('Telugu')) {
      return 'న్యూరల్ ఆడియో పెంపుదల సక్రియంగా ఉంది.';
    } else if (lang.contains('Spanish')) {
      return 'Mejora de audio neuronal activa en el búfer.';
    } else if (lang.contains('French')) {
      return 'Amélioration audio neuronale active sur le tampon.';
    } else if (lang.contains('German')) {
      return 'Neuronale Audioverbesserung im Puffer aktiv.';
    } else if (lang.contains('Japanese')) {
      return 'ニューラルオーディオ強化がバッファで有効です。';
    }
    return 'Denoised voice audio stream processed successfully.';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: const [
            Icon(Icons.graphic_eq, color: Colors.greenAccent),
            SizedBox(width: 10),
            Flexible(
              child: Text(
                'Edge AI Audio Denoiser & Profiler',
                key: Key('app_title_text'),
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.greenAccent.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.greenAccent),
            ),
            child: const Row(
              children: [
                Icon(Icons.wifi, color: Colors.greenAccent, size: 14),
                SizedBox(width: 4),
                Text(
                  '16kHz / 16.0ms Real-Time',
                  style: TextStyle(color: Colors.greenAccent, fontSize: 11, fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        key: const Key('dashboard_scroll_view'),
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // 1. Hero Suite: Subtitles & Multilingual Translation
            SubtitleHeroCard(
              sourceText: _state.sourceText,
              translatedText: _state.translatedText,
              selectedLanguage: _state.targetLanguage,
              onLanguageChanged: _onLanguageChanged,
              onTranslateSubmitted: _onTranslateSubmitted,
              onClear: _onClearTranslation,
            ),
            const SizedBox(height: 16),

            // 2. Audio Denoiser Control Card
            DenoiserControlCard(
              isDenoisingActive: _state.isDenoisingActive,
              isMuted: _state.isMuted,
              suppressionIntensity: _state.suppressionIntensity,
              selectedProfile: _state.noiseProfile,
              onDenoisingToggled: _onDenoisingToggled,
              onMuteToggled: _onMuteToggled,
              onSuppressionChanged: _onSuppressionChanged,
              onProfileChanged: _onProfileChanged,
            ),
            const SizedBox(height: 16),

            // 3. Hardware Latency Profiler Card
            LatencyProfilerCard(
              metrics: _state.metrics,
              onResetMetrics: _onResetMetrics,
            ),
            const SizedBox(height: 16),

            // 4. Multi-Precision Engine Card
            PrecisionSelectorCard(
              selectedPrecision: _state.precision,
              onPrecisionSelected: _onPrecisionSelected,
            ),
            const SizedBox(height: 16),

            // 5. Studio Multi-Track Recording & Export
            SessionRecorderCard(
              isRecording: _state.isRecording,
              isExported: _isExported,
              onRecordToggled: _onRecordToggled,
              onExportTelemetry: _onExportTelemetry,
            ),
            const SizedBox(height: 16),

            // 6. Dynamic Rolling Telemetry Event Stream Log
            TelemetryEventList(events: _state.frameEvents),
          ],
        ),
      ),
    );
  }
}
