/**
 * REST API Client for Edge AI Denoiser & Profiler
 * Zero Global Scope Pollution
 */

export async function fetchStatus() {
  const r = await fetch("/api/status");
  return await r.json();
}

export async function fetchClassifierStatus() {
  const r = await fetch("/api/classifier/status");
  return await r.json();
}

export async function fetchEqResponse() {
  const r = await fetch("/api/eq/response");
  return await r.json();
}

export async function updateSettings(settings) {
  const r = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  return await r.json();
}

export async function configureTranslation(lang) {
  const r = await fetch("/api/translation/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language: lang }),
  });
  return await r.json();
}

export async function translateText(text, targetLang) {
  const r = await fetch("/api/translation/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, target_lang: targetLang }),
  });
  return await r.json();
}

export async function saveRecording(payload) {
  const r = await fetch("/api/recorder/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return await r.json();
}

export async function lockTse() {
  const r = await fetch("/api/tse/lock", { method: "POST" });
  return await r.json();
}

export async function unlockTse() {
  const r = await fetch("/api/tse/unlock", { method: "POST" });
  return await r.json();
}

export async function configureStudio(payload) {
  const r = await fetch("/api/studio/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return await r.json();
}

export async function configureEq(payload) {
  const r = await fetch("/api/eq/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return await r.json();
}

export async function setEqPreset(preset) {
  const r = await fetch("/api/eq/preset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ preset }),
  });
  return await r.json();
}

export async function toggleAutoAdapt() {
  const r = await fetch("/api/adapt/toggle", { method: "POST" });
  return await r.json();
}

export async function uploadWav(file, intensity) {
  const buffer = await file.arrayBuffer();
  const r = await fetch(`/api/audio/upload?intensity=${intensity}&filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: buffer,
  });
  if (!r.ok) {
    const err = await r.json();
    throw new Error(err.detail || "Upload failed");
  }
  return await r.json();
}

export async function runBatchBenchmark(config) {
  const r = await fetch("/api/batch/benchmark", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return await r.json();
}
