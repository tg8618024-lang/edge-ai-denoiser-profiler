/**
 * Edge AI Real-Time Neural Audio Denoiser & Profiler
 * ES Module Entrypoint & Backward-Compatibility Bridge
 *
 * Re-exports core visualizer functions and auto-initializes the application
 * to ensure 100% backward compatibility with existing tests and script tags.
 */

import { initApp } from "./js/main.js";
import { renderOscilloscope, drawDifferenceArea } from "./js/renderers/oscilloscope.js";

export { renderOscilloscope, drawDifferenceArea, initApp };

// Auto-bootstrap when loaded directly
if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
}
