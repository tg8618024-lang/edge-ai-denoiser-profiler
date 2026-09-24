/**
 * Live Microphone Capture & Web Speech Recognition
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { REGIONAL_LOCALES } from "../config.js";
import { audioManager } from "./audio-manager.js";
import { $, safeText, safeClass } from "../utils/dom.js";

export class MicStreamer {
  constructor(wsClient) {
    this.wsClient = wsClient;
    this.micMediaStream = null;
    this.micAudioNode = null;
    this.micMuteNode = null;
    this.speechRecognizer = null;
    this.latestMicTranscript = "";
    this.resampleQueue = [];
    this.resamplePhase = 0.0;
    this.lastInputSample = 0.0;
  }

  getSpeechLocale(langCode) {
    return REGIONAL_LOCALES[langCode] || "en-IN";
  }

  /**
   * Resample input PCM buffer from audioCtx.sampleRate to 16,000 Hz using Catmull-Rom cubic interpolation.
   * Continuous fractional phase is maintained across chunk boundaries to prevent clicks.
   */
  resampleTo16k(inputData, inRate) {
    if (inRate === 16000) {
      for (let i = 0; i < inputData.length; i++) {
        this.resampleQueue.push(inputData[i]);
      }
      return;
    }

    const ratio = inRate / 16000.0;
    const len = inputData.length;
    let phase = this.resamplePhase;

    while (phase < len) {
      const idx0 = Math.floor(phase);
      const frac = phase - idx0;
      const s0 = idx0 === 0 ? this.lastInputSample : inputData[idx0 - 1];
      const s1 = inputData[idx0];
      const s2 = idx0 + 1 < len ? inputData[idx0 + 1] : s1;
      const s3 = idx0 + 2 < len ? inputData[idx0 + 2] : s2;

      // 4-point Catmull-Rom cubic interpolation
      const a0 = -0.5 * s0 + 1.5 * s1 - 1.5 * s2 + 0.5 * s3;
      const a1 = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3;
      const a2 = -0.5 * s0 + 0.5 * s2;
      const a3 = s1;
      const interpolated = ((a0 * frac + a1) * frac + a2) * frac + a3;

      this.resampleQueue.push(interpolated);
      phase += ratio;
    }

    this.resamplePhase = phase - len;
    this.lastInputSample = inputData[len - 1];
  }

  startLiveSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    try {
      if (this.speechRecognizer) {
        this.speechRecognizer.abort();
        this.speechRecognizer = null;
      }
      this.speechRecognizer = new SpeechRecognition();
      this.speechRecognizer.continuous = true;
      this.speechRecognizer.interimResults = true;
      this.speechRecognizer.maxAlternatives = 1;

      const targetLangSelect = $("targetLangSelect");
      const activeLang = targetLangSelect ? targetLangSelect.value : "hi";
      this.speechRecognizer.lang = this.getSpeechLocale(activeLang);

      this.speechRecognizer.onresult = (event) => {
        let interim = "";
        let finalStr = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalStr += event.results[i][0].transcript;
          } else {
            interim += event.results[i][0].transcript;
          }
        }
        const recognized = (finalStr || interim).trim();
        if (recognized) {
          this.latestMicTranscript = recognized;
          state.lastTranscriptText = recognized;
          const subOriginalText = $("subOriginalText");
          if (subOriginalText) subOriginalText.textContent = recognized;

          if (this.wsClient) {
            this.wsClient.send({
              type: "speech_transcript",
              transcript: recognized,
              target_lang: targetLangSelect ? targetLangSelect.value : "hi",
              is_final: Boolean(finalStr),
            });
          }
        }
      };

      this.speechRecognizer.onerror = (e) => {
        console.warn("Live speech recognition warning:", e.error);
      };

      this.speechRecognizer.onend = () => {
        if (state.micActive) {
          try {
            this.speechRecognizer.start();
          } catch (err) {}
        }
      };

      this.speechRecognizer.start();
    } catch (err) {
      console.warn("Could not start SpeechRecognition:", err);
    }
  }

  stopLiveSpeechRecognition() {
    if (this.speechRecognizer) {
      try {
        this.speechRecognizer.stop();
      } catch (err) {}
      this.speechRecognizer = null;
    }
    this.latestMicTranscript = "";
  }

  async startMicrophone() {
    await audioManager.init();
    const audioCtx = audioManager.audioCtx;
    if (!audioCtx) return;
    if (audioCtx.state === "suspended") await audioCtx.resume();

    try {
      this.micMediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      this.resampleQueue = [];
      this.resamplePhase = 0.0;
      this.lastInputSample = 0.0;

      const micSource = audioCtx.createMediaStreamSource(this.micMediaStream);
      // 1024-sample processor reduces callback dispatch overhead
      const scriptNode = audioCtx.createScriptProcessor(1024, 1, 1);
      const targetLangSelect = $("targetLangSelect");

      scriptNode.onaudioprocess = (e) => {
        if (!state.micActive) return;
        const inputData = e.inputBuffer.getChannelData(0);
        this.resampleTo16k(inputData, audioCtx.sampleRate);

        // Cap queue to 4096 samples (256ms) to prevent memory accumulation under backpressure
        if (this.resampleQueue.length > 4096) {
          this.resampleQueue.splice(0, this.resampleQueue.length - 2048);
        }

        // Package exactly 256 samples of 16 kHz PCM (16.0 ms frame) per dispatch
        while (this.resampleQueue.length >= 256) {
          const chunk = this.resampleQueue.splice(0, 256);
          if (this.wsClient) {
            this.wsClient.send({
              type: "audio_frame",
              pcm: chunk,
              transcript: this.latestMicTranscript || "",
              target_lang: targetLangSelect ? targetLangSelect.value : "hi",
            });
          }
        }
      };

      // Acoustic Feedback Elimination:
      // Route microphone through a muted GainNode (gain=0) to destination.
      // This satisfies the browser clock requirement for onaudioprocess while
      // ensuring zero live microphone energy leaks to the physical speakers.
      const muteNode = audioCtx.createGain();
      muteNode.gain.setValueAtTime(0.0, audioCtx.currentTime);

      micSource.connect(scriptNode);
      scriptNode.connect(muteNode);
      muteNode.connect(audioCtx.destination);

      this.micAudioNode = scriptNode;
      this.micMuteNode = muteNode;

      state.micActive = true;
      const btnMicToggle = $("btnMicToggle");
      const micLabel = $("micLabel");
      if (btnMicToggle) {
        btnMicToggle.textContent = "STOP LIVE MIC";
        btnMicToggle.classList.add("btn-warning");
      }
      if (micLabel) {
        const rateLabel = audioCtx.sampleRate === 16000 ? "16kHz Native" : `16kHz (Resampled from ${audioCtx.sampleRate}Hz)`;
        micLabel.textContent = `Microphone: Capturing (${rateLabel})`;
      }

      this.startLiveSpeechRecognition();
    } catch (err) {
      alert("Could not access microphone: " + err.message);
    }
  }

  stopMicrophone() {
    this.stopLiveSpeechRecognition();
    if (this.micMediaStream) {
      this.micMediaStream.getTracks().forEach((track) => track.stop());
      this.micMediaStream = null;
    }
    if (this.micAudioNode) {
      this.micAudioNode.disconnect();
      this.micAudioNode = null;
    }
    if (this.micMuteNode) {
      this.micMuteNode.disconnect();
      this.micMuteNode = null;
    }
    this.resampleQueue = [];
    this.resamplePhase = 0.0;
    this.lastInputSample = 0.0;
    state.micActive = false;

    const btnMicToggle = $("btnMicToggle");
    const micLabel = $("micLabel");
    if (btnMicToggle) {
      btnMicToggle.textContent = "ACTIVATE LIVE MIC";
      btnMicToggle.classList.remove("btn-warning");
    }
    if (micLabel) {
      micLabel.textContent = "Microphone: Standby";
    }
  }
}
