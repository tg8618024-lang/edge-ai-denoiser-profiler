/**
 * System Constants & Configuration for Edge AI Audio Denoiser & Profiler
 */

export const CONFIG = {
  SAMPLE_RATE: 16000,
  FRAME_SIZE: 256,
  OSC_BUF_LEN: 2048,
  SPEC_BINS: 64,
  SPEC_3D_SLICES: 24,
  RING_BUFFER_CAPACITY: 16384,
  PRE_ROLL_CUSHION: 512,
  LATENCY_BUDGET_MS: 20.0,
  CIRCUMFERENCE_HEADROOM: 427.26,
  CIRCUMFERENCE_PURITY: 263.89,
};

export const REGIONAL_LOCALES = {
  hi: "hi-IN",
  ta: "ta-IN",
  te: "te-IN",
  bn: "bn-IN",
  mr: "mr-IN",
  gu: "gu-IN",
  kn: "kn-IN",
  ml: "ml-IN",
  pa: "pa-IN",
  es: "es-ES",
  fr: "fr-FR",
  de: "de-DE",
  ja: "ja-JP",
  zh: "zh-CN",
  it: "it-IT",
  en: "en-IN",
};

export const DEFAULT_EQ_BANDS = [
  { type: "high_pass", freq: 60.0, gain: 0.0, q: 0.707, enabled: true, color: "#FF5252" },
  { type: "low_shelf", freq: 200.0, gain: 0.0, q: 0.707, enabled: true, color: "#FFB300" },
  { type: "bell", freq: 1000.0, gain: 0.0, q: 1.0, enabled: true, color: "#76B900" },
  { type: "bell", freq: 3500.0, gain: 0.0, q: 1.0, enabled: true, color: "#00E5FF" },
  { type: "high_shelf", freq: 8000.0, gain: 0.0, q: 0.707, enabled: true, color: "#B388FF" },
];
