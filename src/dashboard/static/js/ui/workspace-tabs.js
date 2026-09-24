/**
 * 5-Workspace Tab Switcher & Workflow Navigation
 * Zero Global Scope Pollution
 */

import { $, $$ } from "../utils/dom.js";
import { fetchEqResponse } from "../net/api.js";
import { updateEqCurveUI, drawEqCurve } from "../renderers/parametric-eq.js";
import { drawVectorscope } from "../renderers/vectorscope.js";

export function switchWorkspaceTab(targetViewId) {
  if (!targetViewId) return;

  const tabBtns = $$("#workspaceTabsBar .tab-btn");
  const container = $("workspaceViewsContainer");
  const views = $$(".workspace-view");

  tabBtns.forEach((btn) => {
    if (btn.getAttribute("data-view") === targetViewId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  if (targetViewId === "viewAll") {
    if (container) container.classList.add("show-all");
    views.forEach((v) => v.classList.remove("active"));
  } else {
    if (container) container.classList.remove("show-all");
    views.forEach((v) => {
      if (v.id === targetViewId) {
        v.classList.add("active");
      } else {
        v.classList.remove("active");
      }
    });
  }

  try {
    localStorage.setItem("rtx_active_workspace", targetViewId);
    localStorage.setItem("rtx_active_workspace_view", targetViewId);
  } catch (e) {}

  // Trigger resize and canvas redraws for newly revealed views
  window.dispatchEvent(new Event("resize"));
  setTimeout(() => {
    window.dispatchEvent(new Event("resize"));
    if (targetViewId === "viewEqSuite" || targetViewId === "viewAll") {
      fetchEqResponse()
        .then((data) => {
          if (data) updateEqCurveUI(data);
          else drawEqCurve();
        })
        .catch(() => drawEqCurve());
    }
    if (targetViewId === "viewVectorscope" || targetViewId === "viewAll") {
      drawVectorscope({
        phase_correlation: 0.98,
        mono_compatibility_pct: 98.9,
        stereo_width: 0.14,
        orbit_x: [],
        orbit_y: [],
      });
    }
  }, 60);
}

export function initWorkspaceTabs() {
  const workspaceTabsBar = $("workspaceTabsBar");
  if (workspaceTabsBar) {
    workspaceTabsBar.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab-btn");
      if (btn && btn.dataset.view) {
        switchWorkspaceTab(btn.dataset.view);
      }
    });
  }

  const savedTab =
    localStorage.getItem("rtx_active_workspace_view") ||
    localStorage.getItem("rtx_active_workspace") ||
    "viewBroadcast";
  switchWorkspaceTab(savedTab);
}
