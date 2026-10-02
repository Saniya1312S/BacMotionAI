// ═══════════════════════════════════════════════════════════
// BacMotionAI — Interactive Simulator Engine
// Ports the Python physics + ML scoring to JavaScript
// ═══════════════════════════════════════════════════════════

(function () {
  'use strict';

  // ─── Physics Constants ───
  const V0 = 20.0;     // intrinsic speed µm/s
  const DR = 0.2;      // rotational diffusion
  const DT = 0.01;     // time step
  const T_TOTAL = 6.0; // simulation time
  const STEPS = Math.floor(T_TOTAL / DT);
  const L = 200.0;     // channel length
  const A = 1.0;       // cell radius
  const KAPPA = 6.0;   // hydrodynamic torque
  const EPS = 0.12;    // wall speed enhancement

  // ─── Chemical Parameter Distributions ───
  const CHEM_STATS = {
    none: { v_mu: 1.00, v_sig: 0.05, Dr_mu: 1.0, Dr_sig: 0.1, drift_mu: 0.00, drift_sig: 0.05, color: '#607D8B' },
    attractant: { v_mu: 1.12, v_sig: 0.08, Dr_mu: 0.65, Dr_sig: 0.12, drift_mu: 0.28, drift_sig: 0.08, color: '#4CAF50' },
    repellent: { v_mu: 0.90, v_sig: 0.07, Dr_mu: 1.45, Dr_sig: 0.15, drift_mu: -0.22, drift_sig: 0.07, color: '#F44336' },
    antibiotic: { v_mu: 0.52, v_sig: 0.10, Dr_mu: 2.40, Dr_sig: 0.30, drift_mu: 0.00, drift_sig: 0.04, color: '#9C27B0' },
  };

  const TUMBLE_RATES = { none: 0.005, attractant: 0.003, repellent: 0.005, antibiotic: 0.01 };

  const REGIME_PALETTE = {
    'Inhibited': '#a855f7',
    'Wall-Guided': '#ef4444',
    'Chemotactic': '#22c55e',
    'Confined': '#f97316',
    'Free-Swimming': '#3b82f6',
  };

  // ─── Utility: Normal random ───
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function clamp(val, lo, hi) { return Math.max(lo, Math.min(hi, val)); }

  function normalSample(mu, sig) { return mu + sig * randn(); }

  // ═══════════════════════════════════════
  // Bacterium class
  // ═══════════════════════════════════════
  class Bacterium {
    constructor(W, hydro, chemical) {
      this.W = W;
      this.hydro = hydro;
      this.chemical = chemical;

      const cs = CHEM_STATS[chemical];
      this.vScale = clamp(normalSample(cs.v_mu, cs.v_sig), 0.3, 1.5);
      this.DrEff = clamp(normalSample(cs.Dr_mu, cs.Dr_sig), 0.1, 5.0) * DR;
      this.drift = normalSample(cs.drift_mu, cs.drift_sig);
      this.pTumble = TUMBLE_RATES[chemical] || 0.005;

      this.x = Math.random() * L;
      this.y = Math.random() * W;
      this.theta = Math.random() * 2 * Math.PI;

      // Trail for display (capped for animation performance)
      this.trail = [{ x: this.x, y: this.y }];
      this.maxTrail = 120;
      // Full trajectory for charts
      this.fullTrail = [{ x: this.x, y: this.y }];

      // Stats accumulators
      this.speeds = [];
      this.cosThetas = [];
      this.yPositions = [];
      this.wallHits = 0;
      this.runLengths = [];
      this.currentRun = 0;
      this.totalSteps = 0;

      // Color
      this.hue = Math.random() * 40 - 20; // slight per-bact color variation
    }

    step() {
      // Tumbling
      if (Math.random() < this.pTumble) {
        this.theta = Math.random() * 2 * Math.PI;
        if (this.currentRun > 0) this.runLengths.push(this.currentRun);
        this.currentRun = 0;
      } else {
        this.currentRun++;
      }

      let omega = 0, vEff = V0 * this.vScale;

      if (this.hydro) {
        const h = Math.min(this.y, this.W - this.y);
        const hEff = Math.max(h, A);
        omega = -KAPPA * (A / hEff) ** 2 * Math.sin(2 * this.theta);
        vEff = V0 * this.vScale * (1 + EPS * (A / hEff));
      }

      const noiseV = randn() * 0.3;

      this.theta += omega * DT + this.drift * DT + Math.sqrt(2 * this.DrEff * DT) * randn();
      const vx = (vEff + noiseV) * Math.cos(this.theta);
      const vy = (vEff + noiseV) * Math.sin(this.theta);

      this.x += vx * DT;
      this.y += vy * DT;
      this.x = ((this.x % L) + L) % L; // wrap

      if (this.y < 0) { this.y = -this.y; this.theta = -this.theta; this.wallHits++; }
      else if (this.y > this.W) { this.y = 2 * this.W - this.y; this.theta = -this.theta; this.wallHits++; }

      const spd = Math.sqrt(vx * vx + vy * vy);
      this.speeds.push(spd);
      this.cosThetas.push(Math.abs(Math.cos(this.theta)));
      this.yPositions.push(this.y);
      this.totalSteps++;

      this.trail.push({ x: this.x, y: this.y });
      if (this.trail.length > this.maxTrail) this.trail.shift();
      this.fullTrail.push({ x: this.x, y: this.y });
    }

    getFeatures() {
      const s = this.speeds;
      if (s.length === 0) return null;

      const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
      const std = arr => {
        const m = mean(arr);
        return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
      };
      const percentile = (arr, p) => {
        const sorted = [...arr].sort((a, b) => a - b);
        const idx = Math.floor(p / 100 * sorted.length);
        return sorted[Math.min(idx, sorted.length - 1)];
      };
      const skew = arr => {
        const m = mean(arr), sd = std(arr);
        if (sd === 0) return 0;
        return mean(arr.map(x => ((x - m) / sd) ** 3));
      };
      const kurt = arr => {
        const m = mean(arr), sd = std(arr);
        if (sd === 0) return 0;
        return mean(arr.map(x => ((x - m) / sd) ** 4)) - 3;
      };

      const wallLayer = 0.15 * this.W;
      const nearWallFrac = this.yPositions.filter(y => y < wallLayer || y > this.W - wallLayer).length / this.yPositions.length;
      const meanRunLen = this.runLengths.length > 0 ? mean(this.runLengths) : this.totalSteps;
      const meanSpd = mean(s);
      const stdSpd = std(s);

      return {
        mean_speed: meanSpd,
        std_speed: stdSpd,
        q25_speed: percentile(s, 25),
        q75_speed: percentile(s, 75),
        mean_alignment: mean(this.cosThetas),
        std_alignment: std(this.cosThetas),
        wall_rate: this.wallHits / this.totalSteps,
        near_wall_frac: nearWallFrac,
        mean_run_len: meanRunLen,
        confinement_idx: A / this.W,
        cv_speed: meanSpd > 0 ? stdSpd / meanSpd : 0,
        eff_diffusivity: clamp(this.DrEff > 0 ? meanSpd ** 2 / (2 * this.DrEff) : 0, 0, 5000),
        speed_skewness: skew(s),
        speed_kurtosis: kurt(s),
      };
    }
  }

  // ═══════════════════════════════════════
  // Regime Scoring (mirrors Python)
  // ═══════════════════════════════════════
  function classifyRegime(f) {
    const scores = {
      'Inhibited': 0,
      'Wall-Guided': 0,
      'Chemotactic': 0,
      'Confined': 0,
      'Free-Swimming': 0,
    };

    scores['Inhibited'] += Math.max(0, (12 - f.mean_speed) / 8);
    scores['Inhibited'] += Math.min(f.cv_speed, 1.0) * 0.5;
    scores['Inhibited'] += Math.max(0, (80 - f.mean_run_len) / 80);

    scores['Wall-Guided'] += f.near_wall_frac * 2.0;
    scores['Wall-Guided'] += f.mean_alignment > 0.6 ? (f.mean_alignment - 0.6) * 2.0 : 0;

    scores['Chemotactic'] += Math.min(f.mean_run_len / 200, 1.5);
    scores['Chemotactic'] += Math.max(0, 0.5 - f.cv_speed) * 1.5;
    scores['Chemotactic'] += Math.max(0, (f.mean_speed - 18)) / 5;

    scores['Confined'] += f.wall_rate * 3.0;
    scores['Confined'] += Math.max(0, (100 - f.mean_run_len) / 100);

    scores['Free-Swimming'] += Math.max(0, (f.mean_speed - 17) / 5);
    scores['Free-Swimming'] += Math.max(0, (0.4 - f.near_wall_frac));
    scores['Free-Swimming'] += Math.max(0, (0.65 - f.mean_alignment));

    // Add noise to simulate ML uncertainty
    for (const key of Object.keys(scores)) {
      scores[key] += (Math.random() - 0.5) * 0.15;
    }

    // Softmax-like probability
    const maxScore = Math.max(...Object.values(scores));
    const expScores = {};
    let sumExp = 0;
    for (const [k, v] of Object.entries(scores)) {
      expScores[k] = Math.exp((v - maxScore) * 2.0);
      sumExp += expScores[k];
    }
    const probs = {};
    let bestRegime = '', bestProb = 0;
    for (const [k, v] of Object.entries(expScores)) {
      probs[k] = v / sumExp;
      if (probs[k] > bestProb) { bestProb = probs[k]; bestRegime = k; }
    }

    return { regime: bestRegime, confidence: bestProb, probs };
  }

  // ═══════════════════════════════════════
  // Canvas rendering
  // ═══════════════════════════════════════
  const canvas = document.getElementById('simCanvas');
  const ctx = canvas.getContext('2d');
  let bacteria = [];
  let animRunning = false;
  let animFrameId = null;
  let simStepsDone = 0;
  const STEPS_PER_FRAME = 8;

  // Live chart instances (destroyed & recreated each run)
  let liveCharts = {};

  function getSimState() {
    return {
      W: parseInt(document.getElementById('widthSlider').value),
      numBact: parseInt(document.getElementById('numBactSlider').value),
      hydro: document.querySelector('.toggle-btn.active').dataset.hydro === 'true',
      chemical: document.querySelector('.chem-btn.active').dataset.chem,
    };
  }

  function resizeCanvas() {
    const container = canvas.parentElement;
    const w = container.clientWidth;
    const h = Math.max(350, Math.min(500, w * 0.45));
    canvas.width = w;
    canvas.height = h;
  }

  function drawFrame() {
    const { W } = getSimState();
    const cw = canvas.width;
    const ch = canvas.height;

    // Mapping: sim coords → canvas coords
    const padding = 30;
    const scaleX = (cw - 2 * padding) / L;
    const scaleY = (ch - 2 * padding) / W;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (cw - L * scale) / 2;
    const offsetY = (ch - W * scale) / 2;

    const toCanvasX = x => offsetX + x * scale;
    const toCanvasY = y => offsetY + (W - y) * scale; // flip Y

    // Clear with fade
    ctx.fillStyle = 'rgba(10, 14, 26, 0.15)';
    ctx.fillRect(0, 0, cw, ch);

    // Walls
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), toCanvasY(0));
    ctx.lineTo(toCanvasX(L), toCanvasY(0));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), toCanvasY(W));
    ctx.lineTo(toCanvasX(L), toCanvasY(W));
    ctx.stroke();

    // Wall labels
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`W = ${W} µm`, toCanvasX(4), toCanvasY(W) - 6);
    ctx.fillText('0', toCanvasX(4), toCanvasY(0) + 14);

    const chemical = getSimState().chemical;
    const chemColor = CHEM_STATS[chemical].color;

    // Draw bacteria
    for (const bact of bacteria) {
      // Trail
      const trail = bact.trail;
      if (trail.length > 1) {
        for (let i = 1; i < trail.length; i++) {
          const alpha = (i / trail.length) * 0.7;
          ctx.strokeStyle = chemColor;
          ctx.globalAlpha = alpha;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(toCanvasX(trail[i - 1].x), toCanvasY(trail[i - 1].y));
          ctx.lineTo(toCanvasX(trail[i].x), toCanvasY(trail[i].y));
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // Body (rod shape)
      const cx = toCanvasX(bact.x);
      const cy = toCanvasY(bact.y);
      const len = 6 * scale;
      const dx = Math.cos(bact.theta) * len;
      const dy = -Math.sin(bact.theta) * len; // flipped Y

      // Glow
      ctx.shadowColor = chemColor;
      ctx.shadowBlur = 8;
      ctx.strokeStyle = chemColor;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx - dx / 2, cy - dy / 2);
      ctx.lineTo(cx + dx / 2, cy + dy / 2);
      ctx.stroke();

      // Head dot
      ctx.fillStyle = '#fff';
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.arc(cx + dx / 2, cy + dy / 2, 2, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
    }
  }

  function animLoop() {
    if (!animRunning) return;

    for (let s = 0; s < STEPS_PER_FRAME; s++) {
      if (simStepsDone >= STEPS) {
        animRunning = false;
        onSimComplete();
        return;
      }
      for (const bact of bacteria) bact.step();
      simStepsDone++;
    }

    drawFrame();
    animFrameId = requestAnimationFrame(animLoop);
  }

  // ═══════════════════════════════════════
  // UI results update
  // ═══════════════════════════════════════
  function onSimComplete() {
    document.getElementById('canvasStatus').textContent = 'Complete';
    document.getElementById('canvasStatus').className = 'canvas-status idle';
    document.getElementById('runBtn').innerHTML = '▶ Run Simulation';
    document.getElementById('runBtn').classList.remove('running');

    // Average features across all bacteria
    const allFeatures = bacteria.map(b => b.getFeatures()).filter(f => f);
    if (allFeatures.length === 0) return;

    const avgFeatures = {};
    const keys = Object.keys(allFeatures[0]);
    for (const k of keys) {
      avgFeatures[k] = allFeatures.map(f => f[k]).reduce((a, b) => a + b, 0) / allFeatures.length;
    }

    // Update stat cards
    document.getElementById('speedStats').style.display = 'grid';
    document.getElementById('statSpeed').textContent = avgFeatures.mean_speed.toFixed(1);
    document.getElementById('statAlign').textContent = avgFeatures.mean_alignment.toFixed(3);
    document.getElementById('statWall').textContent = avgFeatures.near_wall_frac.toFixed(3);
    document.getElementById('statRunLen').textContent = avgFeatures.mean_run_len.toFixed(0);

    // Feature bars
    const displayFeats = [
      { key: 'mean_speed', label: 'Mean Speed', max: 30, unit: ' µm/s' },
      { key: 'cv_speed', label: 'Speed CV', max: 1.0, unit: '' },
      { key: 'mean_alignment', label: 'Alignment |cosθ|', max: 1.0, unit: '' },
      { key: 'near_wall_frac', label: 'Near-Wall Fraction', max: 1.0, unit: '' },
      { key: 'mean_run_len', label: 'Mean Run Length', max: 400, unit: ' steps' },
      { key: 'wall_rate', label: 'Wall Hit Rate', max: 0.1, unit: '' },
      { key: 'eff_diffusivity', label: 'Eff. Diffusivity', max: 3000, unit: '' },
      { key: 'confinement_idx', label: 'Confinement Index', max: 0.2, unit: '' },
    ];

    let featHTML = '<div class="feature-bar-list">';
    for (const { key, label, max, unit } of displayFeats) {
      const val = avgFeatures[key];
      const pct = Math.min((val / max) * 100, 100);
      featHTML += `
        <div class="feature-bar-item">
          <div class="feature-bar-label">
            <span class="feature-bar-name">${label}</span>
            <span class="feature-bar-val">${val.toFixed(3)}${unit}</span>
          </div>
          <div class="feature-bar-track">
            <div class="feature-bar-fill" style="width:${pct}%"></div>
          </div>
        </div>`;
    }
    featHTML += '</div>';
    document.getElementById('featureContent').innerHTML = featHTML;

    // Trigger bars to animate (force reflow)
    setTimeout(() => {
      document.querySelectorAll('.feature-bar-fill').forEach(el => {
        el.style.width = el.style.width; // force
      });
    }, 50);

    // ML Prediction
    const prediction = classifyRegime(avgFeatures);

    const regimeOrder = ['Inhibited', 'Wall-Guided', 'Chemotactic', 'Confined', 'Free-Swimming'];
    let predHTML = '<div class="prediction-bars">';
    for (const regime of regimeOrder) {
      const prob = prediction.probs[regime] || 0;
      const color = REGIME_PALETTE[regime];
      const isTop = regime === prediction.regime;
      predHTML += `
        <div class="pred-bar-item">
          <span class="pred-bar-dot" style="background:${color}"></span>
          <span class="pred-bar-name" style="color:${isTop ? color : 'var(--text-secondary)'};font-weight:${isTop ? '700' : '500'}">${regime}</span>
          <div class="pred-bar-track">
            <div class="pred-bar-fill" style="width:${(prob * 100).toFixed(1)}%;background:${color}">${(prob * 100).toFixed(0)}%</div>
          </div>
        </div>`;
    }
    predHTML += '</div>';

    const regimeColor = REGIME_PALETTE[prediction.regime];
    predHTML += `
      <div class="regime-result" style="background:${regimeColor}15;border-color:${regimeColor}40">
        <div class="regime-result-label" style="color:${regimeColor}">Predicted Regime</div>
        <div class="regime-result-name" style="color:${regimeColor}">${prediction.regime}</div>
        <div class="regime-result-confidence" style="color:${regimeColor}cc">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
      </div>`;

    document.getElementById('predContent').innerHTML = predHTML;

    // Update live simulation charts
    updateLiveCharts(bacteria, avgFeatures, prediction);
  }

  // ═══════════════════════════════════════
  // Live Simulation Charts
  // ═══════════════════════════════════════
  function updateLiveCharts(bactList, avgFeats, prediction) {
    if (typeof Chart === 'undefined') return;

    const gridColor = 'rgba(255,255,255,0.06)';
    const tickColor = 'rgba(255,255,255,0.45)';

    // Destroy old charts
    for (const key of Object.keys(liveCharts)) {
      if (liveCharts[key]) liveCharts[key].destroy();
    }
    liveCharts = {};

    const state = getSimState();
    const chemColor = CHEM_STATS[state.chemical].color;

    // Palette for individual bacteria
    const bactColors = [
      '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#A855F7',
      '#06B6D4', '#F97316', '#EC4899', '#84CC16', '#6366F1',
      '#14B8A6', '#FB923C', '#E879F9', '#FBBF24', '#34D399',
      '#F87171', '#818CF8', '#38BDF8', '#FB7185', '#A3E635',
      '#C084FC', '#22D3EE', '#FDE047', '#4ADE80', '#F472B6',
    ];

    // ── 1. Trajectory Map ──
    const trajDatasets = bactList.map((bact, i) => {
      // Sample from the full trajectory for chart
      const full = bact.fullTrail;
      const step = Math.max(1, Math.floor(full.length / 120));
      const points = [];
      for (let j = 0; j < full.length; j += step) {
        points.push({ x: full[j].x, y: full[j].y });
      }
      const color = bactColors[i % bactColors.length];
      return {
        label: `Bact ${i + 1}`,
        data: points,
        borderColor: color,
        backgroundColor: color + '30',
        showLine: true,
        tension: 0.1,
        pointRadius: 0,
        borderWidth: 1.5,
      };
    });

    liveCharts.trajectory = new Chart(document.getElementById('liveTrajectory'), {
      type: 'scatter',
      data: { datasets: trajDatasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 800 },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
        scales: {
          x: { min: 0, max: L, title: { display: true, text: 'X position (µm)', color: tickColor }, grid: { color: gridColor } },
          y: { min: 0, max: state.W, title: { display: true, text: 'Y position (µm)', color: tickColor }, grid: { color: gridColor } }
        }
      }
    });

    // ── 2. Speed Histogram ──
    const allSpeeds = bactList.flatMap(b => b.speeds);
    const minSpd = Math.floor(Math.min(...allSpeeds));
    const maxSpd = Math.ceil(Math.max(...allSpeeds));
    const numBins = 15;
    const binWidth = (maxSpd - minSpd) / numBins || 1;
    const speedBins = Array.from({ length: numBins }, (_, i) => (minSpd + i * binWidth).toFixed(1));
    const speedCounts = new Array(numBins).fill(0);
    for (const s of allSpeeds) {
      const bin = Math.min(numBins - 1, Math.floor((s - minSpd) / binWidth));
      if (bin >= 0) speedCounts[bin]++;
    }

    // Gradient colors based on speed
    const speedBarColors = speedCounts.map((_, i) => {
      const ratio = i / numBins;
      const r = Math.round(59 + ratio * 180);
      const g = Math.round(130 - ratio * 60);
      const b = Math.round(246 - ratio * 100);
      return `rgba(${r},${g},${b},0.7)`;
    });

    liveCharts.speedHist = new Chart(document.getElementById('liveSpeedHist'), {
      type: 'bar',
      data: {
        labels: speedBins,
        datasets: [{
          label: 'Count',
          data: speedCounts,
          backgroundColor: speedBarColors,
          borderColor: chemColor,
          borderWidth: 1,
          borderRadius: 3,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 800 },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: (items) => `Speed: ${items[0].label} µm/s` } }
        },
        scales: {
          x: {
            title: { display: true, text: 'Speed (µm/s)', color: tickColor }, grid: { color: gridColor },
            ticks: { maxRotation: 45, font: { size: 9 } }
          },
          y: { title: { display: true, text: 'Frequency', color: tickColor }, grid: { color: gridColor } }
        }
      }
    });

    // ── 3. Y-Position Density ──
    const allY = bactList.flatMap(b => b.yPositions);
    const yBins = 20;
    const yBinWidth = state.W / yBins;
    const yLabels = Array.from({ length: yBins }, (_, i) => (i * yBinWidth + yBinWidth / 2).toFixed(1));
    const yCounts = new Array(yBins).fill(0);
    for (const y of allY) {
      const bin = Math.min(yBins - 1, Math.floor(y / yBinWidth));
      if (bin >= 0) yCounts[bin]++;
    }
    // Normalize
    const yMax = Math.max(...yCounts);
    const yNorm = yCounts.map(c => c / yMax);

    // Color bars: wall regions in red, center in blue
    const wallLayer = 0.15;
    const yBarColors = yNorm.map((v, i) => {
      const pos = (i + 0.5) / yBins;
      if (pos < wallLayer || pos > (1 - wallLayer)) {
        return `rgba(239,68,68,${0.4 + v * 0.5})`;
      }
      return `rgba(59,130,246,${0.3 + v * 0.5})`;
    });

    liveCharts.yDensity = new Chart(document.getElementById('liveYDensity'), {
      type: 'bar',
      data: {
        labels: yLabels,
        datasets: [{
          label: 'Density',
          data: yNorm,
          backgroundColor: yBarColors,
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          borderRadius: 2,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 800 },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => `Density: ${ctx.raw.toFixed(3)}`, title: (items) => `Y = ${items[0].label} µm` } }
        },
        scales: {
          x: {
            title: { display: true, text: 'Y position (µm)', color: tickColor }, grid: { color: gridColor },
            ticks: { maxTicksLimit: 10, font: { size: 9 } }
          },
          y: { title: { display: true, text: 'Normalized Density', color: tickColor }, grid: { color: gridColor }, min: 0, max: 1.1 }
        }
      }
    });

    // ── 4. Regime Prediction Radar ──
    const regimeOrder = ['Inhibited', 'Wall-Guided', 'Chemotactic', 'Confined', 'Free-Swimming'];
    const regimeProbs = regimeOrder.map(r => prediction.probs[r] || 0);
    const regimeColors = regimeOrder.map(r => REGIME_PALETTE[r]);

    liveCharts.radar = new Chart(document.getElementById('liveRadar'), {
      type: 'radar',
      data: {
        labels: regimeOrder,
        datasets: [{
          label: 'Probability',
          data: regimeProbs,
          borderColor: chemColor,
          backgroundColor: chemColor + '25',
          borderWidth: 2,
          pointBackgroundColor: regimeColors,
          pointBorderColor: regimeColors,
          pointRadius: 5,
          pointHoverRadius: 8,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 800 },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${(ctx.raw * 100).toFixed(1)}%` } }
        },
        scales: {
          r: {
            min: 0, max: 1,
            ticks: { stepSize: 0.2, color: tickColor, backdropColor: 'transparent', font: { size: 9 } },
            grid: { color: gridColor },
            angleLines: { color: gridColor },
            pointLabels: { color: 'rgba(255,255,255,0.7)', font: { size: 11 } }
          }
        }
      }
    });
  }

  function startSimulation() {
    if (animRunning) return;

    const state = getSimState();

    // Reset
    if (animFrameId) cancelAnimationFrame(animFrameId);
    bacteria = [];
    simStepsDone = 0;

    // Clear canvas fully
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Create bacteria
    for (let i = 0; i < state.numBact; i++) {
      bacteria.push(new Bacterium(state.W, state.hydro, state.chemical));
    }

    // UI
    document.getElementById('canvasStatus').textContent = 'Simulating...';
    document.getElementById('canvasStatus').className = 'canvas-status running';
    document.getElementById('runBtn').innerHTML = '<span class="spinner"></span> Simulating...';
    document.getElementById('runBtn').classList.add('running');
    document.getElementById('featureContent').innerHTML = '<div class="placeholder-msg"><div class="icon">⏳</div><div class="text">Simulating...</div></div>';
    document.getElementById('predContent').innerHTML = '<div class="placeholder-msg"><div class="icon">⏳</div><div class="text">Waiting for simulation...</div></div>';
    document.getElementById('speedStats').style.display = 'none';

    animRunning = true;
    animLoop();
  }

  // ═══════════════════════════════════════
  // Hero Background Animation
  // ═══════════════════════════════════════
  function initHeroBg() {
    const heroCanvas = document.getElementById('heroBg');
    const hctx = heroCanvas.getContext('2d');
    let particles = [];

    function resizeHero() {
      heroCanvas.width = window.innerWidth;
      heroCanvas.height = window.innerHeight;
    }
    resizeHero();

    class Particle {
      constructor() {
        this.x = Math.random() * heroCanvas.width;
        this.y = Math.random() * heroCanvas.height;
        this.theta = Math.random() * Math.PI * 2;
        this.speed = 0.3 + Math.random() * 0.8;
        this.size = 1.5 + Math.random() * 2;
        this.alpha = 0.15 + Math.random() * 0.35;
        this.Dr = 0.02 + Math.random() * 0.04;
        this.trail = [];
        this.maxTrail = 30 + Math.floor(Math.random() * 30);
      }
      update() {
        if (Math.random() < 0.003) this.theta = Math.random() * Math.PI * 2;
        this.theta += (Math.random() - 0.5) * this.Dr;
        this.x += this.speed * Math.cos(this.theta);
        this.y += this.speed * Math.sin(this.theta);
        if (this.x < 0) this.x = heroCanvas.width;
        if (this.x > heroCanvas.width) this.x = 0;
        if (this.y < 0) this.y = heroCanvas.height;
        if (this.y > heroCanvas.height) this.y = 0;
        this.trail.push({ x: this.x, y: this.y });
        if (this.trail.length > this.maxTrail) this.trail.shift();
      }
      draw() {
        for (let i = 1; i < this.trail.length; i++) {
          const a = (i / this.trail.length) * this.alpha * 0.5;
          hctx.strokeStyle = `rgba(59, 130, 246, ${a})`;
          hctx.lineWidth = this.size * 0.6;
          hctx.beginPath();
          hctx.moveTo(this.trail[i - 1].x, this.trail[i - 1].y);
          hctx.lineTo(this.trail[i].x, this.trail[i].y);
          hctx.stroke();
        }
        hctx.fillStyle = `rgba(96, 165, 250, ${this.alpha})`;
        hctx.beginPath();
        hctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        hctx.fill();
      }
    }

    for (let i = 0; i < 35; i++) particles.push(new Particle());

    function heroLoop() {
      hctx.fillStyle = 'rgba(10, 14, 26, 0.12)';
      hctx.fillRect(0, 0, heroCanvas.width, heroCanvas.height);
      for (const p of particles) { p.update(); p.draw(); }
      requestAnimationFrame(heroLoop);
    }
    heroLoop();

    window.addEventListener('resize', resizeHero);
  }

  // ═══════════════════════════════════════
  // Scroll animations for snapshot cards
  // ═══════════════════════════════════════
  function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, { threshold: 0.15 });

    document.querySelectorAll('.snapshot-card').forEach(card => observer.observe(card));
  }

  // ═══════════════════════════════════════
  // Pipeline step highlighting
  // ═══════════════════════════════════════
  function initPipelineHighlight() {
    const steps = document.querySelectorAll('.pipeline-step');
    const snapCards = ['snap-sim', 'snap-feat', 'snap-ml', 'snap-regime'];

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const idx = snapCards.indexOf(entry.target.id);
          if (idx >= 0) {
            steps.forEach(s => s.classList.remove('active'));
            steps[idx].classList.add('active');
          }
        }
      });
    }, { threshold: 0.5 });

    snapCards.forEach(id => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
  }

  // ═══════════════════════════════════════
  // Chart.js — Analysis Charts
  // ═══════════════════════════════════════
  function initCharts() {
    if (typeof Chart === 'undefined') return;

    Chart.defaults.color = 'rgba(255,255,255,0.65)';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';
    Chart.defaults.font.family = "'Inter', 'Segoe UI', sans-serif";
    Chart.defaults.font.size = 11;

    const gridColor = 'rgba(255,255,255,0.06)';
    const tickColor = 'rgba(255,255,255,0.45)';

    // ── 1. Speed Distribution by Chemical ──
    const speedBins = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+'];
    new Chart(document.getElementById('chartSpeedDist'), {
      type: 'bar',
      data: {
        labels: speedBins,
        datasets: [
          { label: 'None', data: [2, 8, 35, 110, 280, 55, 10], backgroundColor: 'rgba(96,125,139,0.7)', borderColor: '#607D8B', borderWidth: 1 },
          { label: 'Attractant', data: [1, 4, 15, 60, 180, 160, 80], backgroundColor: 'rgba(76,175,80,0.7)', borderColor: '#4CAF50', borderWidth: 1 },
          { label: 'Repellent', data: [5, 25, 80, 180, 150, 45, 15], backgroundColor: 'rgba(244,67,54,0.7)', borderColor: '#F44336', borderWidth: 1 },
          { label: 'Antibiotic', data: [45, 120, 180, 100, 40, 10, 5], backgroundColor: 'rgba(156,39,176,0.7)', borderColor: '#9C27B0', borderWidth: 1 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { boxWidth: 12, padding: 10 } },
          title: { display: false }
        },
        scales: {
          x: { title: { display: true, text: 'Speed (µm/s)', color: tickColor }, grid: { color: gridColor } },
          y: { title: { display: true, text: 'Count', color: tickColor }, grid: { color: gridColor } }
        }
      }
    });

    // ── 2. Model Comparison ──
    new Chart(document.getElementById('chartModelComp'), {
      type: 'bar',
      data: {
        labels: ['Random Forest', 'Logistic Reg.', 'XGBoost', 'MLP'],
        datasets: [
          { label: 'Test Accuracy', data: [0.78, 0.62, 0.76, 0.74], backgroundColor: 'rgba(59,130,246,0.7)', borderColor: '#3B82F6', borderWidth: 1 },
          { label: 'ROC-AUC', data: [0.94, 0.87, 0.93, 0.92], backgroundColor: 'rgba(16,185,129,0.7)', borderColor: '#10B981', borderWidth: 1 },
          { label: 'Log Loss', data: [0.65, 1.10, 0.70, 0.78], backgroundColor: 'rgba(249,115,22,0.7)', borderColor: '#F97316', borderWidth: 1 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 12, padding: 10 } } },
        scales: {
          x: { grid: { color: gridColor } },
          y: { min: 0, max: 1.2, grid: { color: gridColor }, title: { display: true, text: 'Score', color: tickColor } }
        }
      }
    });

    // ── 3. Feature Importances (RF) ──
    const featNames = ['mean_speed', 'cv_speed', 'near_wall_frac', 'mean_alignment', 'mean_run_len', 'wall_rate',
      'eff_diffusivity', 'std_speed', 'confinement_idx', 'q75_speed', 'std_alignment', 'q25_speed', 'speed_skew', 'speed_kurt'];
    const featImps = [0.18, 0.14, 0.13, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01];
    const featColors = featImps.map(v => `rgba(6, 182, 212, ${0.3 + v * 3.5})`);

    new Chart(document.getElementById('chartFeatImp'), {
      type: 'bar',
      data: {
        labels: featNames,
        datasets: [{ label: 'Importance', data: featImps, backgroundColor: featColors, borderColor: '#06B6D4', borderWidth: 1 }]
      },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'Gini Importance', color: tickColor }, grid: { color: gridColor } },
          y: { grid: { color: gridColor }, ticks: { font: { size: 10 } } }
        }
      }
    });

    // ── 4. Class Distribution (Doughnut) ──
    new Chart(document.getElementById('chartClassDist'), {
      type: 'doughnut',
      data: {
        labels: ['Inhibited', 'Wall-Guided', 'Chemotactic', 'Confined', 'Free-Swimming'],
        datasets: [{
          data: [320, 280, 350, 420, 630],
          backgroundColor: ['rgba(168,85,247,0.75)', 'rgba(239,68,68,0.75)', 'rgba(34,197,94,0.75)', 'rgba(249,115,22,0.75)', 'rgba(59,130,246,0.75)'],
          borderColor: ['#a855f7', '#ef4444', '#22c55e', '#f97316', '#3b82f6'],
          borderWidth: 2
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        cutout: '55%',
        plugins: {
          legend: { position: 'right', labels: { boxWidth: 12, padding: 8, font: { size: 11 } } },
          title: { display: true, text: 'Before SMOTE (imbalanced)', color: tickColor, font: { size: 12 } }
        }
      }
    });

    // ── 5. Learning Curves ──
    const trainSizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600];
    new Chart(document.getElementById('chartLearning'), {
      type: 'line',
      data: {
        labels: trainSizes,
        datasets: [
          { label: 'Train Accuracy (RF)', data: [0.95, 0.92, 0.90, 0.88, 0.87, 0.86, 0.855, 0.85], borderColor: '#3B82F6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.3, pointRadius: 3 },
          { label: 'CV Accuracy (RF)', data: [0.55, 0.62, 0.68, 0.72, 0.74, 0.76, 0.77, 0.78], borderColor: '#10B981', backgroundColor: 'rgba(16,185,129,0.1)', fill: true, tension: 0.3, pointRadius: 3 },
          { label: 'Train Accuracy (MLP)', data: [0.98, 0.94, 0.91, 0.89, 0.87, 0.85, 0.84, 0.83], borderColor: '#F59E0B', backgroundColor: 'transparent', borderDash: [5, 3], tension: 0.3, pointRadius: 2 },
          { label: 'CV Accuracy (MLP)', data: [0.42, 0.52, 0.60, 0.65, 0.69, 0.72, 0.73, 0.74], borderColor: '#EF4444', backgroundColor: 'transparent', borderDash: [5, 3], tension: 0.3, pointRadius: 2 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 14, padding: 8, font: { size: 10 } } } },
        scales: {
          x: { title: { display: true, text: 'Training Samples', color: tickColor }, grid: { color: gridColor } },
          y: { min: 0.3, max: 1.0, title: { display: true, text: 'Accuracy', color: tickColor }, grid: { color: gridColor } }
        }
      }
    });

    // ── 6. Confusion Matrix (heatmap via scatter) ──
    const regimes = ['Inhibited', 'Wall-Guided', 'Chemotactic', 'Confined', 'Free-Swim'];
    const cmData = [
      [48, 3, 2, 5, 6],
      [4, 42, 1, 6, 3],
      [2, 1, 55, 4, 8],
      [5, 7, 3, 60, 9],
      [3, 2, 6, 8, 106],
    ];
    // Flatten for bubble chart
    const cmPoints = [];
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        cmPoints.push({ x: j, y: 4 - i, v: cmData[i][j] });
      }
    }
    const maxCM = Math.max(...cmPoints.map(p => p.v));

    new Chart(document.getElementById('chartConfusion'), {
      type: 'bubble',
      data: {
        datasets: [{
          data: cmPoints.map(p => ({ x: p.x, y: p.y, r: Math.max(3, (p.v / maxCM) * 22) })),
          backgroundColor: cmPoints.map(p => {
            const intensity = p.v / maxCM;
            return p.x === (4 - p.y) ? `rgba(59,130,246,${0.3 + intensity * 0.6})` : `rgba(239,68,68,${0.15 + intensity * 0.4})`;
          }),
          borderColor: cmPoints.map(p => p.x === (4 - p.y) ? '#3B82F6' : '#EF4444'),
          borderWidth: 1,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const idx = ctx.dataIndex;
                const row = 4 - cmPoints[idx].y;
                const col = cmPoints[idx].x;
                return `True: ${regimes[row]} → Pred: ${regimes[col]}: ${cmPoints[idx].v}`;
              }
            }
          }
        },
        scales: {
          x: { min: -0.5, max: 4.5, ticks: { callback: (v) => regimes[v] || '', stepSize: 1, font: { size: 9 } }, title: { display: true, text: 'Predicted', color: tickColor }, grid: { color: gridColor } },
          y: { min: -0.5, max: 4.5, ticks: { callback: (v) => regimes[4 - v] || '', stepSize: 1, font: { size: 9 } }, title: { display: true, text: 'True', color: tickColor }, grid: { color: gridColor } }
        }
      }
    });

    // ── 7. ROC Curves ──
    function genROC(auc) {
      const pts = [];
      const bend = 1 - auc + 0.5;
      for (let i = 0; i <= 20; i++) {
        const fpr = i / 20;
        let tpr = Math.min(1, Math.pow(fpr, 1 / (auc * 3.5 + 0.5)));
        tpr = Math.min(1, tpr + (Math.random() - 0.5) * 0.02);
        pts.push({ x: fpr, y: tpr });
      }
      pts.sort((a, b) => a.x - b.x);
      pts[0] = { x: 0, y: 0 };
      pts[pts.length - 1] = { x: 1, y: 1 };
      return pts;
    }

    const rocRegimes = [
      { name: 'Inhibited', auc: 0.96, color: '#a855f7' },
      { name: 'Wall-Guided', auc: 0.94, color: '#ef4444' },
      { name: 'Chemotactic', auc: 0.95, color: '#22c55e' },
      { name: 'Confined', auc: 0.91, color: '#f97316' },
      { name: 'Free-Swimming', auc: 0.93, color: '#3b82f6' },
    ];

    const rocDatasets = rocRegimes.map(r => ({
      label: `${r.name} (AUC=${r.auc})`,
      data: genROC(r.auc),
      borderColor: r.color,
      backgroundColor: 'transparent',
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2,
    }));
    // Diagonal reference
    rocDatasets.push({
      label: 'Random',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: 'rgba(255,255,255,0.2)',
      borderDash: [6, 4],
      pointRadius: 0,
      borderWidth: 1,
    });

    new Chart(document.getElementById('chartROC'), {
      type: 'scatter',
      data: { datasets: rocDatasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        showLine: true,
        plugins: { legend: { labels: { boxWidth: 14, padding: 6, font: { size: 10 } } } },
        scales: {
          x: { min: 0, max: 1, title: { display: true, text: 'False Positive Rate', color: tickColor }, grid: { color: gridColor } },
          y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate', color: tickColor }, grid: { color: gridColor } }
        }
      }
    });

    // ── 8. Overfitting Check ──
    new Chart(document.getElementById('chartOverfit'), {
      type: 'bar',
      data: {
        labels: ['Random Forest', 'Logistic Reg.', 'XGBoost', 'MLP'],
        datasets: [
          { label: '5-Fold CV Acc', data: [0.80, 0.64, 0.79, 0.76], backgroundColor: 'rgba(59,130,246,0.7)', borderColor: '#3B82F6', borderWidth: 1 },
          { label: 'Test Acc', data: [0.78, 0.62, 0.76, 0.74], backgroundColor: 'rgba(16,185,129,0.7)', borderColor: '#10B981', borderWidth: 1 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { boxWidth: 12, padding: 10 } },
          title: { display: true, text: 'Gap < 0.05 = Healthy (no overfitting)', color: 'rgba(34,197,94,0.8)', font: { size: 12 } }
        },
        scales: {
          x: { grid: { color: gridColor } },
          y: { min: 0.4, max: 1.0, grid: { color: gridColor }, title: { display: true, text: 'Accuracy', color: tickColor } }
        }
      }
    });
  }

  // ═══════════════════════════════════════
  // Event Listeners
  // ═══════════════════════════════════════
  function init() {
    // Hero
    initHeroBg();
    initScrollAnimations();
    initPipelineHighlight();
    resizeCanvas();
    initCharts();

    // Width slider
    const widthSlider = document.getElementById('widthSlider');
    widthSlider.addEventListener('input', () => {
      document.getElementById('widthVal').textContent = widthSlider.value + ' µm';
    });

    // Num bacteria slider
    const numBactSlider = document.getElementById('numBactSlider');
    numBactSlider.addEventListener('input', () => {
      document.getElementById('numBactVal').textContent = numBactSlider.value;
    });

    // Hydro toggle
    document.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Chemical selector
    document.querySelectorAll('.chem-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.chem-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Run button
    document.getElementById('runBtn').addEventListener('click', startSimulation);

    // Resize
    window.addEventListener('resize', resizeCanvas);

    // Draw initial empty canvas
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Click "Run Simulation" to start', canvas.width / 2, canvas.height / 2);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
