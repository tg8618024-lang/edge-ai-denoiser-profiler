import 'latency_metrics.dart';

/// Neural inference precision quantization modes.
enum PrecisionMode {
  fp32('FP32', 165.8, '100.0%', '100.0 dB'),
  fp16('FP16', 82.9, '50.0%', '73.6 dB'),
  int8('INT8', 42.6, '25.7%', '39.9 dB');

  final String label;
  final double memoryKb;
  final String compressionRate;
  final String sqnrDb;

  const PrecisionMode(
    this.label,
    this.memoryKb,
    this.compressionRate,
    this.sqnrDb,
  );
}

/// Supported acoustic and RF noise profiles for edge testing.
enum NoiseProfile {
  white('White Noise', 0.0),
  pink('Pink Acoustic', 0.0),
  drone('Drone / Fan Hum', 5.0),
  rfStatic('RF Static Burst', -5.0);

  final String displayName;
  final double baselineSnrDb;

  const NoiseProfile(this.displayName, this.baselineSnrDb);
}

/// Global operational state of the Edge AI Audio Denoiser companion client.
class DenoiserState {
  final bool isDenoisingActive;
  final bool isMuted;
  final bool isRecording;
  final double suppressionIntensity;
  final PrecisionMode precision;
  final NoiseProfile noiseProfile;
  final String sourceText;
  final String translatedText;
  final String targetLanguage;
  final LatencyMetrics metrics;
  final List<String> frameEvents;

  const DenoiserState({
    required this.isDenoisingActive,
    required this.isMuted,
    required this.isRecording,
    required this.suppressionIntensity,
    required this.precision,
    required this.noiseProfile,
    required this.sourceText,
    required this.translatedText,
    required this.targetLanguage,
    required this.metrics,
    required this.frameEvents,
  });

  factory DenoiserState.initial() {
    return DenoiserState(
      isDenoisingActive: true,
      isMuted: false,
      isRecording: false,
      suppressionIntensity: 0.85,
      precision: PrecisionMode.fp32,
      noiseProfile: NoiseProfile.white,
      sourceText: 'Neural audio streaming active on 16.0 ms frame buffer.',
      translatedText: '16.0 एमएस फ्रेम बफर पर न्यूरल ऑडियो स्ट्रीमिंग सक्रिय।',
      targetLanguage: 'Hindi (hi)',
      metrics: LatencyMetrics.defaultReference(),
      frameEvents: List.generate(
        30,
        (i) => 'Frame #$i: Denoised | Latency: 0.25 ms | Headroom: >98%',
      ),
    );
  }

  DenoiserState copyWith({
    bool? isDenoisingActive,
    bool? isMuted,
    bool? isRecording,
    double? suppressionIntensity,
    PrecisionMode? precision,
    NoiseProfile? noiseProfile,
    String? sourceText,
    String? translatedText,
    String? targetLanguage,
    LatencyMetrics? metrics,
    List<String>? frameEvents,
  }) {
    return DenoiserState(
      isDenoisingActive: isDenoisingActive ?? this.isDenoisingActive,
      isMuted: isMuted ?? this.isMuted,
      isRecording: isRecording ?? this.isRecording,
      suppressionIntensity: suppressionIntensity ?? this.suppressionIntensity,
      precision: precision ?? this.precision,
      noiseProfile: noiseProfile ?? this.noiseProfile,
      sourceText: sourceText ?? this.sourceText,
      translatedText: translatedText ?? this.translatedText,
      targetLanguage: targetLanguage ?? this.targetLanguage,
      metrics: metrics ?? this.metrics,
      frameEvents: frameEvents ?? this.frameEvents,
    );
  }
}
