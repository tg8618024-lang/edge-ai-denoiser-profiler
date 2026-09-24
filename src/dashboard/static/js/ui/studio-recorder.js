/**
 * Multi-Track Studio Audio & Subtitle Recorder Suite
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $ } from "../utils/dom.js";
import { saveRecording } from "../net/api.js";

export function accumulateRecordingData(denoised, rawNoisy) {
  if (!state.isRecording) return;
  if (state.recCleanPcm.length < 16000 * 60) {
    for (let i = 0; i < denoised.length; i++) {
      state.recCleanPcm.push(denoised[i]);
      state.recNoisyPcm.push(rawNoisy[i]);
    }
  } else {
    stopRecordingSession();
  }
}

export function startRecordingSession({ onStartStreamIfNeeded }) {
  state.isRecording = true;
  state.recCleanPcm = [];
  state.recNoisyPcm = [];
  state.recStartTime = Date.now();

  const btnRecordToggle = $("btnRecordToggle");
  const recBtnText = $("recBtnText");
  const recTimerBadge = $("recTimerBadge");
  const recTimerText = $("recTimerText");
  const recProgressFill = $("recProgressFill");
  const recStatusInfo = $("recStatusInfo");

  if (btnRecordToggle) btnRecordToggle.classList.add("recording");
  if (recBtnText) recBtnText.textContent = "⏹ STOP & SAVE";
  if (recTimerBadge) recTimerBadge.classList.add("active");
  if (recStatusInfo) recStatusInfo.textContent = "⏺ Recording studio multi-track audio (Clean + Noisy + Delta + Subtitles)...";

  if (!state.isStreaming && !state.micActive && onStartStreamIfNeeded) {
    onStartStreamIfNeeded();
  }

  if (state.recTimerInterval) clearInterval(state.recTimerInterval);
  state.recTimerInterval = setInterval(() => {
    const elapsedSec = (Date.now() - state.recStartTime) / 1000.0;
    const mm = Math.floor(elapsedSec / 60).toString().padStart(2, "0");
    const ss = Math.floor(elapsedSec % 60).toString().padStart(2, "0");
    if (recTimerText) recTimerText.textContent = `${mm}:${ss}`;
    const pct = Math.min(100, (elapsedSec / 60.0) * 100);
    if (recProgressFill) recProgressFill.style.width = `${pct}%`;

    if (elapsedSec >= 60) {
      stopRecordingSession();
    }
  }, 100);
}

export async function stopRecordingSession() {
  if (!state.isRecording) return;
  state.isRecording = false;

  if (state.recTimerInterval) {
    clearInterval(state.recTimerInterval);
    state.recTimerInterval = null;
  }

  const btnRecordToggle = $("btnRecordToggle");
  const recBtnText = $("recBtnText");
  const recTimerBadge = $("recTimerBadge");
  const recStatusInfo = $("recStatusInfo");
  const sessionDownloadsPanel = $("sessionDownloadsPanel");
  const targetLangSelect = $("targetLangSelect");

  if (btnRecordToggle) btnRecordToggle.classList.remove("recording");
  if (recBtnText) recBtnText.textContent = "⏺ START RECORDING";
  if (recTimerBadge) recTimerBadge.classList.remove("active");
  if (recStatusInfo) recStatusInfo.textContent = "Processing & saving session files...";

  const targetLang = targetLangSelect ? targetLangSelect.value : "hi";
  try {
    const data = await saveRecording({
      clean_pcm: state.recCleanPcm,
      noisy_pcm: state.recNoisyPcm,
      transcript: state.lastTranscriptText,
      target_lang: targetLang,
      translated_text: state.lastTranslatedText,
    });

    state.lastRecordedSessionId = data.session_id;
    if (recStatusInfo) {
      recStatusInfo.innerHTML = `&#10004; Saved Session <strong>#${data.session_id}</strong> (${data.duration_sec}s). Ready for 1-click download!`;
    }
    if (sessionDownloadsPanel) sessionDownloadsPanel.classList.remove("hidden");
  } catch (err) {
    if (recStatusInfo) recStatusInfo.textContent = `Error saving session: ${err.message}`;
  }
}

export function initStudioRecorder({ onStartStreamIfNeeded }) {
  const btnRecordToggle = $("btnRecordToggle");
  const btnDlCleanWav = $("btnDlCleanWav");
  const btnDlNoisyWav = $("btnDlNoisyWav");
  const btnDlDeltaWav = $("btnDlDeltaWav");
  const btnDlSrt = $("btnDlSrt");
  const btnDlTxt = $("btnDlTxt");

  if (btnRecordToggle) {
    btnRecordToggle.addEventListener("click", () => {
      if (!state.isRecording) {
        startRecordingSession({ onStartStreamIfNeeded });
      } else {
        stopRecordingSession();
      }
    });
  }

  if (btnDlCleanWav) {
    btnDlCleanWav.addEventListener("click", () => {
      if (state.lastRecordedSessionId) {
        window.location.href = `/api/recorder/download/${state.lastRecordedSessionId}?format=clean_wav`;
      } else {
        window.location.href = `/api/export/wav/clean`;
      }
    });
  }

  if (btnDlNoisyWav) {
    btnDlNoisyWav.addEventListener("click", () => {
      if (state.lastRecordedSessionId) {
        window.location.href = `/api/recorder/download/${state.lastRecordedSessionId}?format=noisy_wav`;
      } else {
        window.location.href = `/api/export/wav/noisy`;
      }
    });
  }

  if (btnDlDeltaWav) {
    btnDlDeltaWav.addEventListener("click", () => {
      if (state.lastRecordedSessionId) {
        window.location.href = `/api/recorder/download/${state.lastRecordedSessionId}?format=delta_wav`;
      } else {
        window.location.href = `/api/export/wav/noise`;
      }
    });
  }

  if (btnDlSrt) {
    btnDlSrt.addEventListener("click", () => {
      if (state.lastRecordedSessionId) {
        window.location.href = `/api/recorder/download/${state.lastRecordedSessionId}?format=srt`;
      } else {
        alert("Please record a session first!");
      }
    });
  }

  if (btnDlTxt) {
    btnDlTxt.addEventListener("click", () => {
      if (state.lastRecordedSessionId) {
        window.location.href = `/api/recorder/download/${state.lastRecordedSessionId}?format=txt`;
      } else {
        alert("Please record a session first!");
      }
    });
  }
}
