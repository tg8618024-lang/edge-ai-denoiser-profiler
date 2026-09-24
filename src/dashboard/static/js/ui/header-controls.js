/**
 * Header Controls, View Modes & Global Keyboard Shortcuts
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $, safeClass } from "../utils/dom.js";

export function initHeaderControls({ onToggleStream, onToggleAB, onSetPrecision }) {
  const btnViewCombined = $("btnViewCombined");
  const btnViewSimple = $("btnViewSimple");
  const btnViewPro = $("btnViewPro");
  const heroStoryboard = $("heroStoryboard");
  const oscilloscopeSection = $("oscilloscopeSection");
  const telem = document.querySelector(".telemetry-card");

  if (btnViewCombined) {
    btnViewCombined.addEventListener("click", () => {
      btnViewCombined.classList.add("active");
      if (btnViewSimple) btnViewSimple.classList.remove("active");
      if (btnViewPro) btnViewPro.classList.remove("active");
      document.body.classList.remove("mode-simple", "mode-pro");
      document.body.classList.add("mode-studio");
      if (heroStoryboard) heroStoryboard.classList.remove("hidden");
      if (oscilloscopeSection) oscilloscopeSection.classList.remove("hidden");
      if (telem) telem.classList.remove("hidden");
    });
  }

  if (btnViewSimple) {
    btnViewSimple.addEventListener("click", () => {
      btnViewSimple.classList.add("active");
      if (btnViewCombined) btnViewCombined.classList.remove("active");
      if (btnViewPro) btnViewPro.classList.remove("active");
      document.body.classList.remove("mode-studio", "mode-pro");
      document.body.classList.add("mode-simple");
      if (heroStoryboard) heroStoryboard.classList.remove("hidden");
      if (oscilloscopeSection) oscilloscopeSection.classList.remove("hidden");
      if (telem) telem.classList.add("hidden");
    });
  }

  if (btnViewPro) {
    btnViewPro.addEventListener("click", () => {
      btnViewPro.classList.add("active");
      if (btnViewCombined) btnViewCombined.classList.remove("active");
      if (btnViewSimple) btnViewSimple.classList.remove("active");
      document.body.classList.remove("mode-studio", "mode-simple");
      document.body.classList.add("mode-pro");
      if (heroStoryboard) heroStoryboard.classList.add("hidden");
      if (oscilloscopeSection) oscilloscopeSection.classList.remove("hidden");
      if (telem) telem.classList.remove("hidden");
    });
  }

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return;

    if (e.code === "Space") {
      e.preventDefault();
      if (onToggleStream) onToggleStream();
    } else if (e.key === "m" || e.key === "M") {
      e.preventDefault();
      if (onToggleAB) onToggleAB();
    } else if (e.key === "1") {
      if (onSetPrecision) onSetPrecision("FP32");
    } else if (e.key === "2") {
      if (onSetPrecision) onSetPrecision("FP16");
    } else if (e.key === "3") {
      if (onSetPrecision) onSetPrecision("INT8");
    } else if (e.key === "k" || e.key === "K") {
      if (btnViewSimple) btnViewSimple.click();
    } else if (e.key === "p" || e.key === "P") {
      if (btnViewPro) btnViewPro.click();
    } else if (e.key === "s" || e.key === "S") {
      if (btnViewCombined) btnViewCombined.click();
    }
  });
}
