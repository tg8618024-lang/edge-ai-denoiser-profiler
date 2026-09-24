"""
Clean line-based reorganization of src/dashboard/static/index.html.
Splits crowded monolithic page into 5 dedicated workspace tabs + All Panels option.
"""

def main():
    with open("src/dashboard/static/index.html", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Locate boundaries by searching line contents
    def find_line(pattern, start_idx=0):
        for idx in range(start_idx, len(lines)):
            if pattern in lines[idx]:
                return idx
        raise ValueError(f"Pattern not found: {pattern}")

    header_end = find_line("</header>") + 1
    subtitles_start = find_line('id="subtitlesSection"')
    # Find start of section tag for subtitles
    subtitles_sec_start = find_line('<section class="hero-subtitles-suite', 0)
    subtitles_sec_end = find_line('</section>', subtitles_sec_start) + 1

    storyboard_sec_start = find_line('<section class="hero-storyboard-panel"', subtitles_sec_end)
    storyboard_sec_end = find_line('</section>', storyboard_sec_start) + 1

    main_start = find_line('<main class="dashboard-grid">', storyboard_sec_end)

    # In control-panel:
    # 1. precision-section
    prec_start = find_line('<!-- Multi-Precision Engine Section -->', main_start)
    # End of precision section is before dualModelSection
    dual_start = find_line('id="dualModelSection"', prec_start)
    dual_sec_start = find_line('<div class="dual-model-panel', prec_start)

    tse_sec_start = find_line('id="tseSection"', dual_sec_start)
    tse_div_start = find_line('<div class="panel-card tse-panel"', dual_sec_start)
    studio_sec_start = find_line('id="studioSuiteSection"', tse_div_start)
    studio_div_start = find_line('<div class="panel-card studio-suite-panel"', tse_div_start)

    eq_sec_start = find_line('id="eqSculptorSection"', studio_div_start)
    eq_div_start = find_line('<div class="panel-card eq-sculptor-panel"', studio_div_start)

    vec_sec_start = find_line('id="vectorscopeSection"', eq_div_start)
    vec_div_start = find_line('<div class="panel-card vectorscope-panel"', eq_div_start)

    batch_sec_start = find_line('id="batchBenchmarkSection"', vec_div_start)
    batch_div_start = find_line('<div class="panel-card batch-benchmark-panel"', vec_div_start)

    snr_start = find_line('<!-- SNR Improvement Meter -->', batch_div_start)
    dnsmos_start = find_line('<!-- ITU-T P.835 Perceptual Voice Quality', snr_start)
    export_rep_start = find_line('<!-- Quick Benchmark Telemetry Reports -->', dnsmos_start)
    control_panel_end = find_line('</section>', export_rep_start) + 1

    # In visualizer-panel:
    vis_start = find_line('<section class="visualizer-panel">', control_panel_end)
    osc_start = find_line('id="oscilloscopeSection"', vis_start)
    osc_div_start = find_line('<div class="oscilloscope-card', vis_start)

    spec_start = find_line('id="spectrogramSection"', osc_div_start)
    spec_div_start = find_line('<div class="spectrogram-card', osc_div_start)

    telem_div_start = find_line('<div class="telemetry-card panel-card">', spec_div_start)
    vis_panel_end = find_line('</section>', telem_div_start) + 1
    main_end = find_line('</main>', vis_panel_end) + 1
    footer_start = find_line('<!-- Studio Footer -->', main_end)

    # Slices
    header_block = "".join(lines[:header_end])
    footer_block = "".join(lines[footer_start:])

    subtitles_block = "".join(lines[subtitles_sec_start:subtitles_sec_end])
    storyboard_block = "".join(lines[storyboard_sec_start:storyboard_sec_end])

    # Left column parts
    ctrl_head = "".join(lines[main_start:prec_start])
    prec_block = "".join(lines[prec_start:dual_sec_start])
    dual_block = "".join(lines[dual_sec_start:tse_div_start])
    tse_block = "".join(lines[tse_div_start:studio_div_start])
    studio_block = "".join(lines[studio_div_start:eq_div_start])
    eq_block = "".join(lines[eq_div_start:vec_div_start])
    vec_block = "".join(lines[vec_div_start:batch_div_start])
    batch_block = "".join(lines[batch_div_start:snr_start])
    snr_block = "".join(lines[snr_start:dnsmos_start])
    dnsmos_block = "".join(lines[dnsmos_start:export_rep_start])
    export_rep_block = "".join(lines[export_rep_start:control_panel_end])

    # Right column parts
    vis_head = "".join(lines[control_panel_end:osc_div_start])
    osc_block = "".join(lines[osc_div_start:spec_div_start])
    spec_block = "".join(lines[spec_div_start:telem_div_start])
    telem_block = "".join(lines[telem_div_start:vis_panel_end])

    # Construct Clean Broadcast Main Grid
    # Left column: ctrl_head + tse_block + snr_block + export_rep_block (ends with </section>)
    # Right column: vis_head + osc_block + spec_block + </section>
    cleaned_dashboard_grid = (
        ctrl_head +
        tse_block +
        snr_block +
        export_rep_block + "\n" +
        vis_head +
        osc_block +
        spec_block +
        "    </section>\n  </main>"
    )

    # Nav Bar HTML
    nav_bar = """  <!-- Workspace Tab Navigation Bar (De-cluttered Workstation Layout) -->
  <nav class="workspace-tabs-bar" id="workspaceTabsBar">
    <button class="tab-btn active" data-view="viewBroadcast">
      <span class="tab-icon">&#127908;</span>
      <div class="tab-text-wrap">
        <span class="tab-title">BROADCAST STUDIO</span>
        <span class="tab-sub">Voice AI &bull; Purity &bull; Waveforms</span>
      </div>
    </button>
    <button class="tab-btn" data-view="viewEqSuite">
      <span class="tab-icon">&#127918;</span>
      <div class="tab-text-wrap">
        <span class="tab-title">STUDIO EQ &amp; DYNAMICS</span>
        <span class="tab-sub">5-Band Parametric EQ &bull; Vocal Suite &bull; De-Reverb</span>
      </div>
    </button>
    <button class="tab-btn" data-view="viewVectorscope">
      <span class="tab-icon">&#129517;</span>
      <div class="tab-text-wrap">
        <span class="tab-title">POLAR VECTORSCOPE</span>
        <span class="tab-sub">Phase Coherence Radar &bull; DNSMOS Quality</span>
      </div>
    </button>
    <button class="tab-btn" data-view="viewBatch">
      <span class="tab-icon">&#128202;</span>
      <div class="tab-text-wrap">
        <span class="tab-title">BATCH BENCHMARK</span>
        <span class="tab-sub">Multi-File Matrix &bull; HTML/JSON Reports</span>
      </div>
    </button>
    <button class="tab-btn" data-view="viewTelemetry">
      <span class="tab-icon">&#9889;</span>
      <div class="tab-text-wrap">
        <span class="tab-title">HARDWARE PROFILER</span>
        <span class="tab-sub">3-Stage Latency &bull; FP32 / FP16 / INT8</span>
      </div>
    </button>
    <button class="tab-btn tab-all-panels" data-view="viewAll">
      <span class="tab-icon">&#128065;</span>
      <div class="tab-text-wrap">
        <span class="tab-title">ALL PANELS</span>
        <span class="tab-sub">Power-User Grid</span>
      </div>
    </button>
  </nav>"""

    # Assemble Workspace Views Container
    views_container = f"""  <!-- ========================================================================= -->
  <!-- WORKSPACE VIEWS CONTAINER (DE-CLUTTERED WORKSTATION ARCHITECTURE)         -->
  <!-- ========================================================================= -->
  <div class="workspace-views-container" id="workspaceViewsContainer">

    <!-- VIEW 1: BROADCAST STUDIO (DEFAULT, UNCLUTTERED EVERYDAY EXPERIENCE) -->
    <div class="workspace-view active" id="viewBroadcast">
      {storyboard_block}

      {cleaned_dashboard_grid}

      {subtitles_block}
    </div>

    <!-- VIEW 2: STUDIO EQ & DYNAMICS WORKSPACE -->
    <div class="workspace-view" id="viewEqSuite">
      <div class="standalone-suite-grid">
        {eq_block}
        {studio_block}
      </div>
    </div>

    <!-- VIEW 3: POLAR VECTORSCOPE & ACOUSTIC QUALITY WORKSPACE -->
    <div class="workspace-view" id="viewVectorscope">
      <div class="standalone-dual-grid">
        {vec_block}
        <div class="panel-card">
          {dnsmos_block}
        </div>
      </div>
    </div>

    <!-- VIEW 4: MULTI-FILE BATCH BENCHMARK WORKSPACE -->
    <div class="workspace-view" id="viewBatch">
      <div class="standalone-suite-grid">
        {batch_block}
      </div>
    </div>

    <!-- VIEW 5: HARDWARE PROFILER & MULTI-PRECISION TELEMETRY -->
    <div class="workspace-view" id="viewTelemetry">
      <div class="standalone-suite-grid">
        <div class="panel-card">
          {prec_block}
          {dual_block}
        </div>
        {telem_block}
      </div>
    </div>

  </div>"""

    new_html = header_block + "\n\n" + nav_bar + "\n\n" + views_container + "\n\n  " + footer_block

    with open("src/dashboard/static/index.html", "w", encoding="utf-8") as f:
        f.write(new_html)

    print("Successfully reorganized index.html with line-based precision!")

if __name__ == "__main__":
    main()
