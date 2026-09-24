/**
 * Batch Benchmark Matrix Studio & Automated Report Generator
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $ } from "../utils/dom.js";
import { runBatchBenchmark } from "../net/api.js";

export function initBatchBenchmark() {
  const btnRunBatchBenchmark = $("btnRunBatchBenchmark");
  const batchProgressWrap = $("batchProgressWrap");
  const batchProgressFill = $("batchProgressFill");
  const batchProgressText = $("batchProgressText");
  const batchSummaryGrid = $("batchSummaryGrid");
  const batchPassRate = $("batchPassRate");
  const batchMeanSnr = $("batchMeanSnr");
  const batchMaxSnr = $("batchMaxSnr");
  const batchMeanLatency = $("batchMeanLatency");
  const batchVerdict = $("batchVerdict");
  const batchActionsRow = $("batchActionsRow");
  const btnDownloadBatchJson = $("btnDownloadBatchJson");
  const btnDownloadBatchHtml = $("btnDownloadBatchHtml");
  const batchTableWrap = $("batchTableWrap");
  const batchTableBody = $("batchTableBody");

  if (btnRunBatchBenchmark) {
    btnRunBatchBenchmark.addEventListener("click", async () => {
      btnRunBatchBenchmark.disabled = true;
      btnRunBatchBenchmark.innerHTML = `<span class="icon">&#8987;</span> RUNNING BATCH MATRIX...`;

      if (batchProgressWrap) batchProgressWrap.classList.remove("hidden");
      if (batchProgressFill) batchProgressFill.style.width = "25%";
      if (batchProgressText) batchProgressText.textContent = "Simulating 24 test combinations across 8 noise categories & 3 precisions...";

      try {
        const data = await runBatchBenchmark({
          presets: ["white", "pink", "drone", "rf_static", "cafe", "rain", "keyboard", "air_conditioner"],
          snr_levels_db: [-5.0, 0.0, 5.0],
          precisions: ["FP32", "FP16", "INT8"],
          duration_sec: 1.0,
        });

        state.latestBatchReport = data;

        if (batchProgressFill) batchProgressFill.style.width = "100%";
        if (batchProgressText) batchProgressText.textContent = `Completed ${data.summary.total_tests} evaluations in ${data.summary.elapsed_sec}s!`;

        // Render Summary
        if (batchSummaryGrid) batchSummaryGrid.classList.remove("hidden");
        if (batchPassRate) batchPassRate.textContent = `${data.summary.pass_rate_pct}%`;
        if (batchMeanSnr) batchMeanSnr.textContent = `+${data.summary.mean_snr_gain_db} dB`;
        if (batchMaxSnr) batchMaxSnr.textContent = `+${data.summary.max_snr_gain_db} dB`;
        if (batchMeanLatency) batchMeanLatency.textContent = `${data.summary.mean_p95_latency_ms} ms`;
        if (batchVerdict) {
          batchVerdict.textContent = data.summary.overall_verdict;
          batchVerdict.className = data.summary.overall_verdict === "PASSED" ? "batch-val tag-good" : "batch-val tag-warning";
        }

        // Render Table
        if (batchTableBody && data.results) {
          batchTableBody.innerHTML = "";
          data.results.forEach((r) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td><strong>${r.preset}</strong></td>
              <td>${r.target_snr_db > 0 ? "+" : ""}${r.target_snr_db} dB</td>
              <td><span class="badge-prec">${r.precision}</span></td>
              <td>${r.in_snr_db} dB</td>
              <td>${r.out_snr_db} dB</td>
              <td class="text-accent"><strong>+${r.snr_gain_db} dB</strong></td>
              <td>${r.p50_latency_ms} ms</td>
              <td>${r.p95_latency_ms} ms</td>
              <td>${r.phase_correlation}</td>
              <td>${r.mono_compat_pct}%</td>
              <td><span class="${r.passed ? "tag-good" : "tag-bad"}">${r.passed ? "PASS" : "FAIL"}</span></td>
            `;
            batchTableBody.appendChild(tr);
          });
          if (batchTableWrap) batchTableWrap.classList.remove("hidden");
        }

        if (batchActionsRow) batchActionsRow.classList.remove("hidden");
      } catch (err) {
        alert("Batch benchmark error: " + err.message);
      } finally {
        btnRunBatchBenchmark.disabled = false;
        btnRunBatchBenchmark.innerHTML = `<span class="btn-icon">&#9658;</span> RUN BATCH BENCHMARK MATRIX`;
      }
    });
  }

  if (btnDownloadBatchJson) {
    btnDownloadBatchJson.addEventListener("click", () => {
      if (!state.latestBatchReport) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.latestBatchReport, null, 2));
      const a = document.createElement("a");
      a.href = dataStr;
      a.download = `edge_ai_batch_benchmark_report_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  }

  if (btnDownloadBatchHtml) {
    btnDownloadBatchHtml.addEventListener("click", () => {
      if (!state.latestBatchReport) return;
      const s = state.latestBatchReport.summary;
      const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Edge AI Batch Benchmark Audit Report</title>
  <style>
    body { font-family: monospace; background: #0c1117; color: #f0f6fc; padding: 24px; }
    h1 { color: #76b900; }
    .summary { display: flex; gap: 16px; margin-bottom: 24px; }
    .card { background: #161b22; border: 1px solid #30363d; padding: 12px 18px; border-radius: 6px; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    th, td { border: 1px solid #30363d; padding: 8px; text-align: left; font-size: 13px; }
    th { background: #161b22; color: #8b949e; }
    .pass { color: #76b900; font-weight: bold; }
    .fail { color: #ff5252; font-weight: bold; }
  </style>
</head>
<body>
  <h1>⚡ Edge AI Audio Denoiser & Profiler — Batch Benchmark Audit</h1>
  <p>Generated on ${new Date().toISOString()} &bull; Total Tests: ${s.total_tests} &bull; Overall Verdict: <strong class="pass">${s.overall_verdict}</strong></p>
  <div class="summary">
    <div class="card">Pass Rate: <strong>${s.pass_rate_pct}%</strong></div>
    <div class="card">Mean SNR Gain: <strong>+${s.mean_snr_gain_db} dB</strong></div>
    <div class="card">Max SNR Gain: <strong>+${s.max_snr_gain_db} dB</strong></div>
    <div class="card">Mean P95 Latency: <strong>${s.mean_p95_latency_ms} ms</strong></div>
  </div>
  <table>
    <thead>
      <tr><th>Preset</th><th>Target SNR</th><th>Precision</th><th>In SNR</th><th>Out SNR</th><th>Gain</th><th>P50 (ms)</th><th>P95 (ms)</th><th>Phase</th><th>Mono Compat</th><th>Verdict</th></tr>
    </thead>
    <tbody>
      ${state.latestBatchReport.results
        .map(
          (r) => `<tr>
        <td>${r.preset}</td><td>${r.target_snr_db} dB</td><td>${r.precision}</td><td>${r.in_snr_db} dB</td><td>${r.out_snr_db} dB</td>
        <td class="pass">+${r.snr_gain_db} dB</td><td>${r.p50_latency_ms} ms</td><td>${r.p95_latency_ms} ms</td><td>${r.phase_correlation}</td><td>${r.mono_compat_pct}%</td>
        <td class="${r.passed ? "pass" : "fail"}">${r.passed ? "PASS" : "FAIL"}</td>
      </tr>`
        )
        .join("")}
    </tbody>
  </table>
</body>
</html>`;
      const blob = new Blob([htmlContent], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `edge_ai_batch_benchmark_audit_${Date.now()}.html`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  }
}
