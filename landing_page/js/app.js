/**
 * WILDFIRE MONITOR — CINEMATIC CONTINUOUS SCROLLYTELLING ENGINE
 * Smooth Video Canvas Scrubbing Tied to Page Scroll + Interactive Tactical Sandbox
 */

(function () {
  'use strict';

  const TOTAL_FRAMES = 165;
  const FRAME_BASE_PATH = 'assets/frames/frame_';
  const FRAME_EXT = '.jpg';
  const BG_SAND_COLOR = '#c9b79c';

  const images = [];
  let loadedCount = 0;
  let currentFrameIndex = 1;
  let targetFrameIndex = 1;
  let currentlyDrawnFrameIndex = -1;
  let isTicking = false;
  let canvas, ctx;

  // Cached layout metrics to avoid any reflow or recomputation during render
  let cachedW = 0;
  let cachedH = 0;
  let cachedRenderX = 0;
  let cachedRenderY = 0;
  let cachedRenderW = 0;
  let cachedRenderH = 0;

  function getFramePath(index) {
    const padded = String(index).padStart(3, '0');
    return `${FRAME_BASE_PATH}${padded}${FRAME_EXT}`;
  }

  function initPreloader() {
    canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas, { passive: true });

    const initialBatch = 20;
    for (let i = 1; i <= TOTAL_FRAMES; i++) {
      const img = new Image();
      img.src = getFramePath(i);
      
      const onLoaded = () => {
        loadedCount++;
        const percent = Math.round((loadedCount / TOTAL_FRAMES) * 100);
        const bar = document.getElementById('loader-bar');
        const text = document.getElementById('loader-text');
        if (bar) bar.style.width = `${percent}%`;
        if (text) text.textContent = `INITIALIZING TELEMETRY (${percent}%)`;

        if (i === 1 && currentlyDrawnFrameIndex === -1) {
          drawFrame(1);
        }

        if (loadedCount >= initialBatch) {
          const preloader = document.getElementById('preloader');
          if (preloader && !preloader.classList.contains('loaded')) {
            preloader.classList.add('loaded');
          }
        }
      };

      if ('decode' in img) {
        img.decode().then(onLoaded).catch(() => {
          img.onload = onLoaded;
        });
      } else {
        img.onload = onLoaded;
      }

      images[i] = img;
    }
  }

  function resizeCanvas() {
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cachedW = window.innerWidth;
    cachedH = window.innerHeight;

    canvas.width = cachedW * dpr;
    canvas.height = cachedH * dpr;
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // 2K QHD aspect ratio is 2560x1440 (16:9)
    const imgRatio = 16 / 9;
    const canvasRatio = cachedW / cachedH;

    if (canvasRatio > imgRatio) {
      cachedRenderW = cachedW;
      cachedRenderH = cachedW / imgRatio;
      cachedRenderX = 0;
      cachedRenderY = (cachedH - cachedRenderH) / 2;
    } else {
      cachedRenderH = cachedH;
      cachedRenderW = cachedH * imgRatio;
      cachedRenderX = (cachedW - cachedRenderW) / 2;
      cachedRenderY = 0;
    }

    currentlyDrawnFrameIndex = -1;
    drawFrame(Math.round(currentFrameIndex) || 1);
  }

  function drawFrame(frameNumber) {
    const img = images[frameNumber];
    if (!img || !img.complete || img.naturalWidth === 0 || !canvas || !ctx) return;

    if (frameNumber === currentlyDrawnFrameIndex) return;
    currentlyDrawnFrameIndex = frameNumber;

    ctx.drawImage(img, cachedRenderX, cachedRenderY, cachedRenderW, cachedRenderH);
  }

  function updateScroll() {
    const maxScroll = document.documentElement.scrollHeight - cachedH;
    const scrollTop = window.scrollY || window.pageYOffset;
    const progress = Math.min(Math.max(scrollTop / (maxScroll || 1), 0), 1);

    targetFrameIndex = 1 + progress * (TOTAL_FRAMES - 1);

    if (!isTicking) {
      window.requestAnimationFrame(renderLoop);
      isTicking = true;
    }
  }

  function renderLoop() {
    const diff = targetFrameIndex - currentFrameIndex;
    if (Math.abs(diff) > 0.005) {
      currentFrameIndex += diff * 0.11;
      const frameToDraw = Math.round(Math.min(Math.max(currentFrameIndex, 1), TOTAL_FRAMES));
      drawFrame(frameToDraw);
      window.requestAnimationFrame(renderLoop);
    } else {
      currentFrameIndex = targetFrameIndex;
      drawFrame(Math.round(currentFrameIndex));
      isTicking = false;
    }
  }

  // --- Interactive Weather Sandbox & 9x9 Grid ---
  function initWeatherWidget() {
    const tempInput = document.getElementById('slider-temp');
    const rhInput = document.getElementById('slider-rh');
    const windInput = document.getElementById('slider-wind');
    const dcInput = document.getElementById('slider-dc');

    const tempVal = document.getElementById('val-temp');
    const rhVal = document.getElementById('val-rh');
    const windVal = document.getElementById('val-wind');
    const dcVal = document.getElementById('val-dc');

    const vpdOutput = document.getElementById('out-vpd');
    const scoreOutput = document.getElementById('out-score');

    if (!tempInput || !rhInput || !windInput) return;

    function recalculate() {
      const T = parseFloat(tempInput.value);
      const RH = parseFloat(rhInput.value);
      const W = parseFloat(windInput.value);
      const DC = parseFloat(dcInput ? dcInput.value : 680);

      if (tempVal) tempVal.textContent = `${T.toFixed(1)} °C`;
      if (rhVal) rhVal.textContent = `${RH.toFixed(0)} %`;
      if (windVal) windVal.textContent = `${W.toFixed(1)} km/h`;
      if (dcVal) dcVal.textContent = `${DC.toFixed(0)}`;

      // Tetens formulation
      const es = 0.61078 * Math.exp((17.27 * T) / (T + 237.3));
      const vpd = es * (1 - RH / 100);

      const DMC = 40.0;
      const bui = (0.8 * DMC * DC) / (DMC + 0.4 * DC);
      const isi = 0.208 * W * (1 + 0.05 * (30 - RH));
      const spread = (isi * W) / 10;

      const riskScore = Math.min(Math.max((vpd * 0.35 + (spread / 15) * 0.45 + (bui / 100) * 0.2), 0), 1);

      if (vpdOutput) vpdOutput.textContent = `${vpd.toFixed(3)} kPa`;
      if (scoreOutput) {
        scoreOutput.textContent = riskScore.toFixed(3);
        scoreOutput.style.color = riskScore > 0.65 ? 'var(--fire-ember)' : 'var(--amber-dark)';
      }

      updateSimulationGrid(riskScore);
    }

    tempInput.addEventListener('input', recalculate);
    rhInput.addEventListener('input', recalculate);
    windInput.addEventListener('input', recalculate);
    if (dcInput) dcInput.addEventListener('input', recalculate);

    recalculate();
  }

  function initMatroidGrid() {
    const gridContainer = document.getElementById('interactive-matroid-grid');
    if (!gridContainer) return;

    gridContainer.innerHTML = '';
    for (let y = 1; y <= 9; y++) {
      for (let x = 1; x <= 9; x++) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        cell.id = `cell-${x}-${y}`;
        cell.textContent = `${x},${y}`;
        gridContainer.appendChild(cell);
      }
    }
  }

  function updateSimulationGrid(overallRisk) {
    const hotspots = [
      { x: 8, y: 6, weight: 1.0 },
      { x: 7, y: 4, weight: 0.85 },
      { x: 6, y: 5, weight: 0.75 },
      { x: 4, y: 4, weight: 0.60 },
      { x: 2, y: 2, weight: 0.40 },
      { x: 8, y: 3, weight: 0.70 },
      { x: 3, y: 8, weight: 0.50 }
    ];

    let crewAllocated = 0;
    const MAX_TOTAL_CREWS = 25;
    const cellAllocations = {};

    const allCells = document.querySelectorAll('.grid-cell');
    allCells.forEach(cell => {
      cell.className = 'grid-cell';
      const existingBadge = cell.querySelector('.crew-badge');
      if (existingBadge) existingBadge.remove();
    });

    const rankedCells = [];
    for (let y = 1; y <= 9; y++) {
      for (let x = 1; x <= 9; x++) {
        let cellRisk = 0.1 * overallRisk;
        hotspots.forEach(h => {
          const dist = Math.hypot(x - h.x, y - h.y);
          cellRisk += (h.weight * overallRisk * Math.exp(-dist * 0.65));
        });
        cellRisk = Math.min(Math.max(cellRisk, 0), 1);
        rankedCells.push({ x, y, risk: cellRisk });
      }
    }

    rankedCells.sort((a, b) => b.risk - a.risk);

    rankedCells.forEach(item => {
      const key = `${item.x}-${item.y}`;
      const cellEl = document.getElementById(`cell-${key}`);
      if (!cellEl) return;

      if (item.risk > 0.6) {
        cellEl.classList.add('active-risk-high');
      } else if (item.risk > 0.3) {
        cellEl.classList.add('active-risk-med');
      }

      const canTake = Math.min(4, Math.floor(item.risk * 4.5));
      const needed = Math.min(canTake, MAX_TOTAL_CREWS - crewAllocated);

      if (needed > 0 && item.risk > 0.35) {
        cellAllocations[key] = needed;
        crewAllocated += needed;

        const badge = document.createElement('span');
        badge.className = 'crew-badge';
        badge.textContent = needed;
        cellEl.appendChild(badge);
      }
    });

    const telemetryCount = document.getElementById('allocated-crew-count');
    if (telemetryCount) telemetryCount.textContent = `${crewAllocated} / 25 CREWS ALLOCATED`;
  }

  function initScrollReveal() {
    const reveals = document.querySelectorAll('.reveal, .reveal-left');
    if (!reveals.length) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15, rootMargin: '0px 0px -60px 0px' });
    reveals.forEach(el => observer.observe(el));
  }

  function initEmberParticles() {
    const container = document.getElementById('ember-container');
    if (!container) return;
    const EMBER_COUNT = 15;
    for (let i = 0; i < EMBER_COUNT; i++) {
      const ember = document.createElement('div');
      ember.className = 'ember-particle';
      ember.style.left = Math.random() * 100 + 'vw';
      ember.style.animationDuration = (8 + Math.random() * 12) + 's';
      ember.style.animationDelay = (Math.random() * 10) + 's';
      ember.style.width = (2 + Math.random() * 4) + 'px';
      ember.style.height = ember.style.width;
      container.appendChild(ember);
    }
  }

  function initNavScroll() {
    const nav = document.querySelector('.top-nav');
    if (!nav) return;
    function checkScroll() {
      if (window.scrollY > 80) {
        nav.classList.add('scrolled');
      } else {
        nav.classList.remove('scrolled');
      }
    }
    window.addEventListener('scroll', checkScroll, { passive: true });
    checkScroll();
  }

  function initBackToTop() {
    const btn = document.getElementById('back-to-top');
    if (!btn) return;
    function checkVisibility() {
      if (window.scrollY > window.innerHeight) {
        btn.classList.add('visible');
      } else {
        btn.classList.remove('visible');
      }
    }
    window.addEventListener('scroll', checkVisibility, { passive: true });
    checkVisibility();
  }

  function initSmoothAnchors() {
    document.querySelectorAll('a[href^="#"]').forEach(link => {
      link.addEventListener('click', e => {
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
          e.preventDefault();
          const offset = 80;
          const y = target.getBoundingClientRect().top + window.pageYOffset - offset;
          window.scrollTo({ top: y, behavior: 'smooth' });
        }
      });
    });
  }

  window.addEventListener('DOMContentLoaded', () => {
    initPreloader();
    initMatroidGrid();
    initWeatherWidget();
    initScrollReveal();
    initEmberParticles();
    initNavScroll();
    initBackToTop();
    initSmoothAnchors();
    window.addEventListener('scroll', updateScroll, { passive: true });
    updateScroll();
  });

})();
