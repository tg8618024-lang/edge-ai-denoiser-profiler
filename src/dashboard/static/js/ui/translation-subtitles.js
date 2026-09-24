/**
 * Multilingual Translation & Live Subtitles Suite
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $, $$, safeText } from "../utils/dom.js";
import { configureTranslation, translateText } from "../net/api.js";
import { wsClient } from "../net/websocket.js";

export function updateSubtitlesUI(sub) {
  if (!sub) return;

  const subOriginalText = $("subOriginalText");
  const subTranslatedText = $("subTranslatedText");
  const subTargetFlag = $("subTargetFlag");
  const subTargetName = $("subTargetName");
  const speechVadDot = $("speechVadDot");
  const subVadConf = $("subVadConf");
  const subTimeline = $("subTimeline");
  const subSentenceCount = $("subSentenceCount");

  if (subOriginalText && sub.original_text) {
    subOriginalText.textContent = sub.original_text;
    state.lastTranscriptText = sub.original_text;
  }
  if (subTranslatedText && sub.translated_text) {
    subTranslatedText.textContent = sub.translated_text;
    state.lastTranslatedText = sub.translated_text;
  }
  if (subTargetFlag && sub.target_flag) {
    subTargetFlag.textContent = sub.target_flag;
  }
  if (subTargetName && sub.target_name) {
    subTargetName.textContent = `TRANSLATED (${sub.target_name.toUpperCase()})`;
  }
  if (speechVadDot) {
    if (sub.is_speech) speechVadDot.classList.add("speaking");
    else speechVadDot.classList.remove("speaking");
  }
  if (subVadConf && sub.confidence !== undefined) {
    subVadConf.textContent = `${(sub.confidence * 100).toFixed(1)}%`;
  }
  if (subTimeline && sub.timeline_sec !== undefined) {
    const sec = sub.timeline_sec;
    const mm = Math.floor(sec / 60).toString().padStart(2, "0");
    const ss = Math.floor(sec % 60).toString().padStart(2, "0");
    const cs = Math.floor((sec % 1) * 100).toString().padStart(2, "0");
    subTimeline.textContent = `${mm}:${ss}.${cs}`;
  }
  if (subSentenceCount && sub.original_text) {
    const count = sub.original_text.split(/[.!?]+/).filter(Boolean).length || 1;
    subSentenceCount.textContent = `${count} segment${count > 1 ? "s" : ""}`;
  }
}

export async function performLiveTranslate(customText) {
  const liveTranslateInput = $("liveTranslateInput");
  const liveTranslateOutput = $("liveTranslateOutput");
  const liveTranslateFlag = $("liveTranslateFlag");
  const liveTranslateTargetBadge = $("liveTranslateTargetBadge");
  const liveTranslateLatency = $("liveTranslateLatency");
  const targetLangSelect = $("targetLangSelect");

  const text = (customText !== undefined ? customText : (liveTranslateInput ? liveTranslateInput.value : "")).trim();
  if (!text) return;

  const targetLang = targetLangSelect ? targetLangSelect.value : "hi";
  const t0 = performance.now();
  try {
    const data = await translateText(text, targetLang);
    const elapsed = (performance.now() - t0).toFixed(3);

    if (liveTranslateOutput) liveTranslateOutput.textContent = data.translated_text;
    if (liveTranslateFlag) liveTranslateFlag.textContent = data.target_flag;
    if (liveTranslateTargetBadge) {
      liveTranslateTargetBadge.textContent = `${data.target_name.toUpperCase()} · TRANSLATION RESULT`;
    }
    if (liveTranslateLatency) {
      const lat = data.latency_ms !== undefined ? data.latency_ms.toFixed(3) : elapsed;
      liveTranslateLatency.textContent = `${lat} ms LATENCY`;
    }

    // Sync with top-level Hero Subtitles display
    const subOriginalText = $("subOriginalText");
    const subTranslatedText = $("subTranslatedText");
    const subTargetFlag = $("subTargetFlag");
    const subTargetName = $("subTargetName");

    if (subOriginalText) subOriginalText.textContent = text;
    if (subTranslatedText) subTranslatedText.textContent = data.translated_text;
    if (subTargetFlag) subTargetFlag.textContent = data.target_flag;
    if (subTargetName) subTargetName.textContent = `TRANSLATED (${data.target_name.toUpperCase()})`;

    state.lastTranscriptText = text;
    state.lastTranslatedText = data.translated_text;
  } catch (err) {
    console.error("Live translate error:", err);
  }
}

export function initTranslationSubtitles(micStreamer = null) {
  const targetLangSelect = $("targetLangSelect");
  const subTargetFlag = $("subTargetFlag");
  const subTargetName = $("subTargetName");
  const liveTranslateFlag = $("liveTranslateFlag");
  const liveTranslateTargetBadge = $("liveTranslateTargetBadge");
  const liveTranslateInput = $("liveTranslateInput");
  const subTranslatedText = $("subTranslatedText");
  const liveTranslateOutput = $("liveTranslateOutput");
  const liveTranslateLatency = $("liveTranslateLatency");
  const btnLiveTranslate = $("btnLiveTranslate");
  const btnLiveTranslateMic = $("btnLiveTranslateMic");
  const promptChips = $$(".btn-prompt-chip");

  if (targetLangSelect) {
    targetLangSelect.addEventListener("change", async (e) => {
      const lang = e.target.value;
      try {
        const data = await configureTranslation(lang);
        if (subTargetFlag) subTargetFlag.textContent = data.flag;
        if (subTargetName) subTargetName.textContent = `TRANSLATED (${data.name.toUpperCase()})`;
        if (liveTranslateFlag) liveTranslateFlag.textContent = data.flag;
        if (liveTranslateTargetBadge) liveTranslateTargetBadge.textContent = `${data.name.toUpperCase()} · TRANSLATION RESULT`;

        const currentText = (liveTranslateInput && liveTranslateInput.value.trim()) || state.lastTranscriptText;
        if (currentText) {
          const tData = await translateText(currentText, lang);
          if (subTranslatedText) subTranslatedText.textContent = tData.translated_text;
          if (liveTranslateOutput) liveTranslateOutput.textContent = tData.translated_text;
          if (liveTranslateLatency) liveTranslateLatency.textContent = `${tData.latency_ms} ms LATENCY`;
          state.lastTranslatedText = tData.translated_text;
        }

        if (micStreamer && micStreamer.speechRecognizer && state.micActive) {
          micStreamer.speechRecognizer.lang = micStreamer.getSpeechLocale(lang);
        }

        wsClient.send({ type: "set_language", language: lang });
      } catch (err) {
        console.error("Failed to configure translation language:", err);
      }
    });
  }

  if (btnLiveTranslate) {
    btnLiveTranslate.addEventListener("click", () => performLiveTranslate());
  }

  if (liveTranslateInput) {
    liveTranslateInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") performLiveTranslate();
    });
  }

  promptChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const phrase = chip.getAttribute("data-text");
      if (liveTranslateInput) liveTranslateInput.value = phrase;
      performLiveTranslate(phrase);
    });
  });

  // Single-shot Web Speech on the Live Translate Testbox Speak button
  if (btnLiveTranslateMic) {
    let testboxRecognizer = null;
    btnLiveTranslateMic.addEventListener("click", () => {
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRec) {
        alert("Web Speech recognition is not supported in this browser. Please type custom text.");
        return;
      }
      if (testboxRecognizer) {
        testboxRecognizer.stop();
        testboxRecognizer = null;
        btnLiveTranslateMic.classList.remove("listening");
        return;
      }
      try {
        testboxRecognizer = new SpeechRec();
        testboxRecognizer.continuous = false;
        testboxRecognizer.interimResults = true;
        testboxRecognizer.lang = "en-IN";

        btnLiveTranslateMic.classList.add("listening");

        testboxRecognizer.onresult = (event) => {
          const transcript = event.results[0][0].transcript;
          if (liveTranslateInput) liveTranslateInput.value = transcript;
          if (event.results[0].isFinal) {
            performLiveTranslate(transcript);
          }
        };

        testboxRecognizer.onend = () => {
          btnLiveTranslateMic.classList.remove("listening");
          testboxRecognizer = null;
        };

        testboxRecognizer.onerror = () => {
          btnLiveTranslateMic.classList.remove("listening");
          testboxRecognizer = null;
        };

        testboxRecognizer.start();
      } catch (err) {
        btnLiveTranslateMic.classList.remove("listening");
        testboxRecognizer = null;
      }
    });
  }
}
