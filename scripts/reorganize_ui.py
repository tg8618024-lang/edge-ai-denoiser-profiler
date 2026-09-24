"""
Script to reorganize src/dashboard/static/index.html into a clean, intuitive
5-Workspace Tab architecture that removes visual crowding and provides dedicated,
spacious views for each workflow:
  1. Broadcast Studio (Default, uncluttered view for everyday voice AI)
  2. Studio EQ & Dynamics (5-band parametric EQ & vocal suite)
  3. Polar Vectorscope & Acoustics (Lissajous phase radar & DNSMOS)
  4. Batch Benchmark Studio (multi-file matrix & downloadable reports)
  5. Hardware Profiler (3-stage latency, budget headroom & multi-precision)
  6. All Panels (unified power-user grid)
"""

import re

def main():
    with open("src/dashboard/static/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Subtitles Block
    sub_start = html.find('<!-- ========================================================================= -->\n  <!-- TOP-LEVEL HERO SUITE: LIVE SUBTITLES')
    sub_end = html.find('<!-- ========================================================================= -->\n  <!-- HERO STORYBOARD:')
    assert sub_start != -1 and sub_end != -1, "Subtitles not found"
    subtitles_block = html[sub_start:sub_end].strip()

    # 2. Hero Storyboard Block
    hero_start = sub_end
    hero_end = html.find('<!-- ========================================================================= -->\n  <!-- MAIN INTERACTIVE STUDIO WORKSPACE')
    assert hero_start != -1 and hero_end != -1, "Storyboard not found"
    storyboard_block = html[hero_start:hero_end].strip()

    # 3. Precision Section
    prec_start = html.find('<!-- Multi-Precision Engine Section -->')
    prec_end = html.find('<!-- ===================================================================== -->\n      <!-- TARGET SPEAKER EXTRACTION')
    assert prec_start != -1 and prec_end != -1, "Precision not found"
    prec_block = html[prec_start:prec_end].strip()

    # 4. Studio Vocal Suite Section
    studio_start = html.find('<!-- ===================================================================== -->\n      <!-- STUDIO SOUND QUALITY SUITE')
    studio_end = html.find('<!-- ===================================================================== -->\n      <!-- OPTION B: 5-BAND STUDIO PARAMETRIC EQ')
    assert studio_start != -1 and studio_end != -1, "Studio suite not found"
    studio_suite_block = html[studio_start:studio_end].strip()

    # 5. EQ Sculptor Section
    eq_start = studio_end
    eq_end = html.find('<!-- ===================================================================== -->\n      <!-- POLAR LISSAJOUS VECTORSCOPE')
    assert eq_start != -1 and eq_end != -1, "EQ sculptor not found"
    eq_block = html[eq_start:eq_end].strip()

    # 6. Vectorscope Section
    vec_start = eq_end
    vec_end = html.find('<!-- ===================================================================== -->\n      <!-- MULTI-FILE BATCH BENCHMARK')
    assert vec_start != -1 and vec_end != -1, "Vectorscope not found"
    vec_block = html[vec_start:vec_end].strip()

    # 7. Batch Benchmark Section
    batch_start = vec_end
    batch_end = html.find('<!-- SNR Improvement Meter -->')
    assert batch_start != -1 and batch_end != -1, "Batch benchmark not found"
    batch_block = html[batch_start:batch_end].strip()

    # 8. DNSMOS Panel
    dnsmos_start = html.find('<!-- ITU-T P.835 Perceptual Voice Quality (DNSMOS) -->')
    dnsmos_end = html.find('<!-- Quick Benchmark Telemetry Reports -->')
    assert dnsmos_start != -1 and dnsmos_end != -1, "DNSMOS panel not found"
    dnsmos_block = html[dnsmos_start:dnsmos_end].strip()

    # 9. Telemetry Section in Visualizer Panel
    telem_start = html.find('<!-- NVIDIA-Style Latency & Pipeline Profiler Stage Breakdown -->')
    telem_end = html.find('</section>\n    </section>\n  </main>')
    if telem_end == -1:
        telem_end = html.find('</section>\n  </main>')
    assert telem_start != -1 and telem_end != -1, "Telemetry section not found"
    telem_block = html[telem_start:telem_end].strip()

    # Clean the central dashboard grid:
    c_html = html
    c_html = c_html.replace(subtitles_block, "")
    c_html = c_html.replace(storyboard_block, "")
    c_html = c_html.replace(prec_block, "")
    c_html = c_html.replace(studio_suite_block, "")
    c_html = c_html.replace(eq_block, "")
    c_html = c_html.replace(vec_block, "")
    c_html = c_html.replace(batch_block, "")
    c_html = c_html.replace(dnsmos_block, "")
    c_html = c_html.replace(telem_block, "")

    main_grid_start = c_html.find('<main class="dashboard-grid">')
    main_grid_end = c_html.find('</main>') + len('</main>')
    cleaned_main_grid = c_html[main_grid_start:main_grid_end].strip()

    # Workspace Navigation Bar
    nav_bar = """  <!-- Workspace Tab Navigation Bar -->
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

    # Assemble views container:
    views_container = f"""  <!-- ========================================================================= -->
  <!-- WORKSPACE VIEWS CONTAINER (DE-CLUTTERED WORKSTATION ARCHITECTURE)         -->
  <!-- ========================================================================= -->
  <div class="workspace-views-container" id="workspaceViewsContainer">

    <!-- VIEW 1: BROADCAST STUDIO (DEFAULT, UNCLUTTERED EVERYDAY EXPERIENCE) -->
    <div class="workspace-view active" id="viewBroadcast">
      {storyboard_block}

      {cleaned_main_grid}

      {subtitles_block}
    </div>

    <!-- VIEW 2: STUDIO EQ & DYNAMICS WORKSPACE -->
    <div class="workspace-view" id="viewEqSuite">
      <div class="standalone-suite-grid">
        {eq_block}
        {studio_suite_block}
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
        </div>
        {telem_block}
      </div>
    </div>

  </div>"""

    header_end = html.find("</header>") + len("</header>")
    footer_start = html.find("<!-- Studio Footer -->")

    header_part = html[:header_end]
    footer_part = html[footer_start:]

    new_html = header_part + "\n\n" + nav_bar + "\n\n" + views_container + "\n\n  " + footer_part

    with open("src/dashboard/static/index.html", "w", encoding="utf-8") as f:
        f.write(new_html)

    print("Successfully reorganized index.html into 5 dedicated workspace views!")

if __name__ == "__main__":
    main()
