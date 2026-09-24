/**
 * Studio Sound Quality Suite, TSE Voice Lock & Auto-Adapt Noise Controller
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $ } from "../utils/dom.js";
import { lockTse, unlockTse, configureStudio, toggleAutoAdapt } from "../net/api.js";
import { wsClient } from "../net/websocket.js";

export function updateTseUI(tse) {
  if (!tse) return;
  const tseLockStatusBadge = $("tseLockStatusBadge");
  const tseSimilarityVal = $("tseSimilarityVal");
  const tseSimilarityBar = $("tseSimilarityBar");
  const tseTargetState = $("tseTargetState");
  const tseSuppressionVal = $("tseSuppressionVal");
  const tseEnrollmentVal = $("tseEnrollmentVal");
  const tseLastTriggerPhrase = $("tseLastTriggerPhrase");

  if (tseLockStatusBadge) {
    if (tse.is_locked) {
      tseLockStatusBadge.textContent = "🔒 TARGET SPEAKER LOCKED";
      tseLockStatusBadge.className = "status-tag tag-locked";
    } else if (tse.is_enrolling) {
      tseLockStatusBadge.textContent = `⏳ ENROLLING (${tse.enrollment_progress_pct}%)`;
      tseLockStatusBadge.className = "status-tag tag-enrolling";
    } else {
      tseLockStatusBadge.textContent = "🔓 UNLOCKED (ALL VOICES)";
      tseLockStatusBadge.className = "status-tag tag-unlocked";
    }
  }

  if (tseSimilarityVal && tse.similarity_pct !== undefined) {
    tseSimilarityVal.textContent = `${tse.similarity_pct.toFixed(1)}%`;
  }
  if (tseSimilarityBar && tse.similarity_pct !== undefined) {
    tseSimilarityBar.style.width = `${Math.min(100, Math.max(0, tse.similarity_pct))}%`;
  }
  if (tseTargetState) {
    if (tse.is_locked) {
      tseTargetState.textContent = tse.is_target_active ? "Primary Voice Active" : "Competing Speaker Suppressed";
      tseTargetState.className = tse.is_target_active ? "stat-val text-success" : "stat-val text-warning";
    } else if (tse.is_enrolling) {
      tseTargetState.textContent = "Capturing Voiceprint...";
      tseTargetState.className = "stat-val text-accent";
    } else {
      tseTargetState.textContent = "Pass-Through (All Voices)";
      tseTargetState.className = "stat-val text-dim";
    }
  }
  if (tseSuppressionVal && tse.secondary_speaker_suppression_db !== undefined) {
    tseSuppressionVal.textContent = `-${tse.secondary_speaker_suppression_db.toFixed(1)} dB`;
  }
  if (tseEnrollmentVal) {
    if (tse.is_locked) {
      tseEnrollmentVal.textContent = "Locked (64-D)";
    } else if (tse.is_enrolling) {
      tseEnrollmentVal.textContent = `${tse.enrollment_progress_pct}%`;
    } else {
      tseEnrollmentVal.textContent = "Ready";
    }
  }
  if (tseLastTriggerPhrase && tse.last_trigger_phrase) {
    tseLastTriggerPhrase.textContent = `Triggered: "${tse.last_trigger_phrase}"`;
  }
}

export function handleTseTriggerDetected(msg) {
  if (!msg) return;
  const tseLastTriggerPhrase = $("tseLastTriggerPhrase");
  if (tseLastTriggerPhrase && msg.phrase) {
    tseLastTriggerPhrase.textContent = `Triggered: "${msg.phrase}" (${msg.command.toUpperCase()})`;
  }
  if (msg.telemetry) {
    updateTseUI(msg.telemetry);
  }
}

export function updateStudioUI(studio) {
  if (!studio) return;
  const toggleStudioSuite = $("toggleStudioSuite");
  const badgeStudioState = $("badgeStudioState");
  const sliderDereverb = $("sliderDereverb");
  const dereverbValue = $("dereverbValue");
  const compGrVal = $("compGrVal");
  const deessRedVal = $("deessRedVal");
  const toggleCompressor = $("toggleCompressor");
  const toggleDeesser = $("toggleDeesser");
  const toggleWarmth = $("toggleWarmth");

  if (toggleStudioSuite && studio.suite_enabled !== undefined) {
    toggleStudioSuite.checked = studio.suite_enabled;
  }
  if (badgeStudioState && studio.suite_enabled !== undefined) {
    badgeStudioState.textContent = studio.suite_enabled ? "STUDIO ACTIVE" : "BYPASS";
    badgeStudioState.style.background = studio.suite_enabled ? "#00E5FF" : "";
    badgeStudioState.style.color = studio.suite_enabled ? "#000" : "";
  }
  if (dereverbValue && studio.dereverb_amount !== undefined) {
    const pct = Math.round(studio.dereverb_amount * 100);
    dereverbValue.textContent = pct === 0 ? "0% (Bypass)" : `${pct}% Active`;
  }
  if (compGrVal && studio.gain_reduction_db !== undefined) {
    compGrVal.textContent = `-${studio.gain_reduction_db.toFixed(1)} dB`;
  }
  if (deessRedVal && studio.deesser_reduction_db !== undefined) {
    deessRedVal.textContent = `-${studio.deesser_reduction_db.toFixed(1)} dB`;
  }
  if (toggleCompressor && studio.compressor_enabled !== undefined) {
    toggleCompressor.checked = studio.compressor_enabled;
  }
  if (toggleDeesser && studio.deesser_enabled !== undefined) {
    toggleDeesser.checked = studio.deesser_enabled;
  }
  if (toggleWarmth && studio.warmth_enabled !== undefined) {
    toggleWarmth.checked = studio.warmth_enabled;
  }
}

export function updateAutoAdaptUI(enabled) {
  state.autoAdaptEnabled = Boolean(enabled);
  const btnAutoAdaptToggle = $("btnAutoAdaptToggle");
  const autoAdaptText = $("autoAdaptText");
  if (btnAutoAdaptToggle) {
    if (state.autoAdaptEnabled) {
      btnAutoAdaptToggle.classList.add("active");
      if (autoAdaptText) autoAdaptText.textContent = "AUTO-ADAPT: ON";
    } else {
      btnAutoAdaptToggle.classList.remove("active");
      if (autoAdaptText) autoAdaptText.textContent = "AUTO-ADAPT: OFF";
    }
  }
}

export function initStudioSuite() {
  const btnTseLock = $("btnTseLock");
  const btnTseUnlock = $("btnTseUnlock");
  const toggleStudioSuite = $("toggleStudioSuite");
  const badgeStudioState = $("badgeStudioState");
  const sliderDereverb = $("sliderDereverb");
  const dereverbValue = $("dereverbValue");
  const toggleCompressor = $("toggleCompressor");
  const toggleDeesser = $("toggleDeesser");
  const toggleWarmth = $("toggleWarmth");
  const btnAutoAdaptToggle = $("btnAutoAdaptToggle");
  const btnQuickExportWav = $("btnQuickExportWav");

  if (btnTseLock) {
    btnTseLock.addEventListener("click", () => {
      wsClient.send({ type: "lock_target_speaker" });
      lockTse().catch(console.error);
    });
  }

  if (btnTseUnlock) {
    btnTseUnlock.addEventListener("click", () => {
      wsClient.send({ type: "unlock_target_speaker" });
      unlockTse().catch(console.error);
    });
  }

  if (toggleStudioSuite) {
    toggleStudioSuite.addEventListener("change", () => {
      const active = toggleStudioSuite.checked;
      if (badgeStudioState) {
        badgeStudioState.textContent = active ? "STUDIO ACTIVE" : "BYPASS";
        badgeStudioState.style.background = active ? "#00E5FF" : "";
        badgeStudioState.style.color = active ? "#000" : "";
      }
      wsClient.send({
        type: "configure_studio",
        vocal_suite_enabled: active,
      });
      configureStudio({ vocal_suite_enabled: active }).catch(console.error);
    });
  }

  if (sliderDereverb) {
    sliderDereverb.addEventListener("input", () => {
      const val = parseInt(sliderDereverb.value, 10);
      if (dereverbValue) {
        dereverbValue.textContent = val === 0 ? "0% (Bypass)" : `${val}% Active`;
      }
      const amt = val / 100.0;
      wsClient.send({
        type: "configure_studio",
        dereverb_amount: amt,
      });
      configureStudio({ dereverb_amount: amt }).catch(console.error);
    });
  }

  function sendStudioModuleConfig() {
    const payload = {
      type: "configure_studio",
      compressor_enabled: toggleCompressor ? toggleCompressor.checked : true,
      deesser_enabled: toggleDeesser ? toggleDeesser.checked : true,
      warmth_enabled: toggleWarmth ? toggleWarmth.checked : true,
    };
    wsClient.send(payload);
    configureStudio({
      compressor_enabled: payload.compressor_enabled,
      deesser_enabled: payload.deesser_enabled,
      warmth_enabled: payload.warmth_enabled,
    }).catch(console.error);
  }

  if (toggleCompressor) toggleCompressor.addEventListener("change", sendStudioModuleConfig);
  if (toggleDeesser) toggleDeesser.addEventListener("change", sendStudioModuleConfig);
  if (toggleWarmth) toggleWarmth.addEventListener("change", sendStudioModuleConfig);

  if (btnAutoAdaptToggle) {
    btnAutoAdaptToggle.addEventListener("click", async () => {
      try {
        const data = await toggleAutoAdapt();
        updateAutoAdaptUI(data.auto_adapt_enabled);
      } catch (err) {
        console.error("Failed to toggle auto-adapt:", err);
      }
    });
  }

  if (btnQuickExportWav) {
    btnQuickExportWav.addEventListener("click", () => {
      window.location.href = "/api/export/wav/clean";
    });
  }
}
