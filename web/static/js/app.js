/**
 * DONEZO x WILDFIRE 3D TACTICAL COMMAND CENTER
 * Unified Frontend Application & 3D Spatial Grid Engine
 */

(function () {
  'use strict';

  // Global Application State
  let currentGridData = [];
  let currentCrews = [];
  let currentMetrics = {};
  let selectedCell = null;
  let isAutoRotating = false;

  // 3D Scene variables
  let scene, camera, renderer, controls;
  let pillarGroup, crewGroup, gridHelperGroup, labelGroup;
  let raycaster, mouse;
  let hoveredMesh = null;
  let isThreeAvailable = (typeof THREE !== 'undefined');

  // DOM Elements
  const canvasWrapper = document.getElementById('canvas-wrapper');
  const webglCanvas = document.getElementById('webgl-canvas');
  const tooltip = document.getElementById('cell-tooltip');
  const portfolioTbody = document.getElementById('portfolio-table-body');
  const matrixGrid = document.getElementById('matrix-grid');

  // Sliders
  const sliderTemp = document.getElementById('slider-temp');
  const sliderRh = document.getElementById('slider-rh');
  const sliderWind = document.getElementById('slider-wind');
  const valTemp = document.getElementById('val-temp-delta');
  const valRh = document.getElementById('val-rh-delta');
  const valWind = document.getElementById('val-wind-delta');

  // Top KPI Elements
  const kpiCrewsDeployed = document.getElementById('kpi-crews-deployed');
  const kpiMaxPerCell = document.getElementById('kpi-max-per-cell');
  const kpiUniqueSectors = document.getElementById('kpi-unique-sectors');
  const kpiTotalEvalRows = document.getElementById('kpi-total-eval-rows');
  const kpiDatasetTag = document.getElementById('kpi-dataset-tag');
  const kpiConstraintBadge = document.getElementById('kpi-constraint-badge');
  const sidebarCrewCount = document.getElementById('sidebar-crew-count');

  // Metric Accuracy Cards
  const metricNdcg = document.getElementById('metric-ndcg');
  const metricRecall = document.getElementById('metric-recall');
  const metricRubric = document.getElementById('metric-rubric');
  const evalSourceLabel = document.getElementById('eval-source-label');
  const hudStatusBadge = document.getElementById('hud-status-badge');

  // Inspector Elements
  const inspectorCoordsBadge = document.getElementById('inspector-coords-badge');
  const inspImpact = document.getElementById('insp-impact');
  const inspCrews = document.getElementById('insp-crews');
  const inspTemp = document.getElementById('insp-temp');
  const inspWind = document.getElementById('insp-wind');
  const inspRh = document.getElementById('insp-rh');
  const inspFfmc = document.getElementById('insp-ffmc');

  // =========================================================================
  // Initialize Application
  // =========================================================================
  async function init() {
    setupSliders();
    setupPresetButtons();
    setupUpload();
    setupSearchFilter();
    setupMissionTimer();

    if (isThreeAvailable) {
      initThreeScene();
    } else {
      console.warn("Three.js not loaded, using isometric Canvas fallback.");
      initCanvasFallback();
    }

    await fetchGridData();
  }

  // =========================================================================
  // 3D Three.js Spatial Grid Visualization
  // =========================================================================
  function initThreeScene() {
    try {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x06110b);
      scene.fog = new THREE.FogExp2(0x06110b, 0.015);

      const width = canvasWrapper.clientWidth || 700;
      const height = canvasWrapper.clientHeight || 320;

      camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      camera.position.set(16, 22, 24);

      renderer = new THREE.WebGLRenderer({
        canvas: webglCanvas,
        antialias: true,
        powerPreference: "high-performance"
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;

      if (typeof THREE.OrbitControls !== 'undefined') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2 - 0.05;
        controls.minDistance = 8;
        controls.maxDistance = 70;
        controls.target.set(0, 0, 0);
      }

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0x45c486, 1.2);
      dirLight.position.set(15, 30, 20);
      dirLight.castShadow = true;
      scene.add(dirLight);

      const fireLight = new THREE.PointLight(0xff4500, 1.5, 35);
      fireLight.position.set(-10, 15, -10);
      scene.add(fireLight);

      // Groups
      pillarGroup = new THREE.Group();
      crewGroup = new THREE.Group();
      gridHelperGroup = new THREE.Group();
      labelGroup = new THREE.Group();

      scene.add(gridHelperGroup);
      scene.add(pillarGroup);
      scene.add(crewGroup);
      scene.add(labelGroup);

      buildGroundPlane();
      buildCoordinateLabels();

      // Raycasting
      raycaster = new THREE.Raycaster();
      mouse = new THREE.Vector2();

      webglCanvas.addEventListener('mousemove', onMouseMove, false);
      webglCanvas.addEventListener('click', onMouseClick, false);
      window.addEventListener('resize', onWindowResize, false);

      setupViewControls();
      animate();
    } catch (err) {
      console.warn("WebGL initialization failed, falling back to Canvas:", err);
      isThreeAvailable = false;
      initCanvasFallback();
    }
  }

  function createTextSprite(text, color = '#45c486', fontSize = 26) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = `bold ${fontSize}px "Plus Jakarta Sans", sans-serif`;
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(2.0, 1.0, 1.0);
    return sprite;
  }

  function buildCoordinateLabels() {
    if (!labelGroup) return;
    while (labelGroup.children.length > 0) {
      labelGroup.remove(labelGroup.children[0]);
    }

    for (let x = 1; x <= 9; x++) {
      const wx = (x - 5) * 2;
      const sprite = createTextSprite(`X:${x}`, '#45c486', 22);
      sprite.position.set(wx, 0.15, 10.2);
      labelGroup.add(sprite);
    }

    for (let y = 1; y <= 9; y++) {
      const wz = -(y - 5) * 2;
      const sprite = createTextSprite(`Y:${y}`, '#f59e0b', 22);
      sprite.position.set(-10.2, 0.15, wz);
      labelGroup.add(sprite);
    }
  }

  function buildGroundPlane() {
    const size = 18;
    const divisions = 9;
    const gridHelper = new THREE.GridHelper(size, divisions, 0x1b533a, 0x0f2b1e);
    gridHelper.position.y = 0.01;
    gridHelperGroup.add(gridHelper);

    const groundGeo = new THREE.PlaneGeometry(24, 24);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x07150e,
      roughness: 0.9,
      metalness: 0.1
    });
    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.rotation.x = -Math.PI / 2;
    groundMesh.receiveShadow = true;
    gridHelperGroup.add(groundMesh);
  }

  function cellToWorld(x, y) {
    return {
      x: (x - 5) * 2,
      z: -(y - 5) * 2
    };
  }

  function getScoreColor(score) {
    if (score <= 0.05) return 0x10b981;  // Emerald Green
    if (score < 0.25) return 0x45c486;   // Mint Green
    if (score < 0.50) return 0xf59e0b;   // Amber Orange
    if (score < 0.75) return 0xf97316;   // Bright Orange
    return 0xef4444;                     // Red Fire
  }

  function render3DGrid() {
    if (!isThreeAvailable || !pillarGroup) {
      if (!isThreeAvailable) renderCanvasFallback();
      return;
    }

    while (pillarGroup.children.length > 0) {
      pillarGroup.remove(pillarGroup.children[0]);
    }
    while (crewGroup.children.length > 0) {
      crewGroup.remove(crewGroup.children[0]);
    }

    currentGridData.forEach(cell => {
      const cx = cell.x;
      const cy = cell.y;
      const { x: wx, z: wz } = cellToWorld(cx, cy);

      const baseScore = cell.max_impact || cell.avg_impact || 0;
      const height = Math.max(0.3, baseScore * 6.5 + (cell.observation_count > 0 ? 0.3 : 0.1));

      const geometry = new THREE.BoxGeometry(1.7, height, 1.7);
      const colorVal = getScoreColor(baseScore);

      const material = new THREE.MeshStandardMaterial({
        color: colorVal,
        roughness: 0.25,
        metalness: 0.6,
        transparent: true,
        opacity: cell.observation_count > 0 ? 0.92 : 0.45,
        emissive: colorVal,
        emissiveIntensity: baseScore * 0.4
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(wx, height / 2, wz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = { cellData: cell, height: height, baseColor: colorVal };

      pillarGroup.add(mesh);

      if (cell.crews_assigned > 0) {
        renderCrewMarkersForCell(cell, wx, wz, height);
      }
    });
  }

  function renderCrewMarkersForCell(cell, wx, wz, pillarHeight) {
    const crewCount = Math.min(4, cell.crews_assigned);
    const topY = pillarHeight;

    const offsets = [
      { dx: 0, dz: 0 },
      { dx: -0.4, dz: 0 }, { dx: 0.4, dz: 0 },
      { dx: -0.4, dz: -0.4 }, { dx: 0.4, dz: -0.4 }, { dx: 0, dz: 0.4 },
      { dx: -0.4, dz: -0.4 }, { dx: 0.4, dz: -0.4 }, { dx: -0.4, dz: 0.4 }, { dx: 0.4, dz: 0.4 }
    ];

    let cellOffsets;
    if (crewCount === 1) cellOffsets = [offsets[0]];
    else if (crewCount === 2) cellOffsets = [offsets[1], offsets[2]];
    else if (crewCount === 3) cellOffsets = [offsets[3], offsets[4], offsets[5]];
    else cellOffsets = [offsets[6], offsets[7], offsets[8], offsets[9]];

    // Glowing Base Ring
    const ringGeo = new THREE.RingGeometry(0.7, 0.82, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x45c486, side: THREE.DoubleSide });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.set(wx, topY + 0.02, wz);
    crewGroup.add(ringMesh);

    cellOffsets.forEach((pos) => {
      const pinX = wx + pos.dx;
      const pinZ = wz + pos.dz;

      const pinGeo = new THREE.ConeGeometry(0.22, 0.7, 8);
      const pinMat = new THREE.MeshStandardMaterial({
        color: 0x45c486,
        emissive: 0x45c486,
        emissiveIntensity: 0.8,
        metalness: 0.9,
        roughness: 0.1
      });
      const pinMesh = new THREE.Mesh(pinGeo, pinMat);
      pinMesh.rotation.x = Math.PI;
      pinMesh.position.set(pinX, topY + 0.5, pinZ);
      crewGroup.add(pinMesh);

      const sphereGeo = new THREE.SphereGeometry(0.14, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
      sphereMesh.position.set(pinX, topY + 0.95, pinZ);
      crewGroup.add(sphereMesh);
    });
  }

  function setupViewControls() {
    const btnIso = document.getElementById('view-iso');
    const btnTop = document.getElementById('view-top');
    const btnLow = document.getElementById('view-low');
    const btnRotate = document.getElementById('view-rotate');

    if (btnIso) {
      btnIso.addEventListener('click', () => {
        setCameraPreset(16, 22, 24);
        setActiveViewBtn('view-iso');
      });
    }
    if (btnTop) {
      btnTop.addEventListener('click', () => {
        setCameraPreset(0, 32, 0.01);
        setActiveViewBtn('view-top');
      });
    }
    if (btnLow) {
      btnLow.addEventListener('click', () => {
        setCameraPreset(0, 5, 26);
        setActiveViewBtn('view-low');
      });
    }
    if (btnRotate) {
      btnRotate.addEventListener('click', () => {
        isAutoRotating = !isAutoRotating;
        if (controls) controls.autoRotate = isAutoRotating;
        btnRotate.textContent = isAutoRotating ? 'Rotating' : 'Rotate';
        btnRotate.classList.toggle('active', isAutoRotating);
      });
    }
  }

  function setActiveViewBtn(id) {
    ['view-iso', 'view-top', 'view-low'].forEach(bId => {
      const el = document.getElementById(bId);
      if (el) el.classList.toggle('active', bId === id);
    });
  }

  function setCameraPreset(x, y, z) {
    if (!camera || !controls) return;
    camera.position.set(x, y, z);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function onMouseMove(event) {
    const rect = webglCanvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(pillarGroup.children);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hoveredMesh !== hit) {
        if (hoveredMesh && hoveredMesh.userData.baseColor) {
          hoveredMesh.material.emissive.setHex(hoveredMesh.userData.baseColor);
          hoveredMesh.material.emissiveIntensity = hoveredMesh.userData.cellData.max_impact * 0.4;
        }
        hoveredMesh = hit;
        hoveredMesh.material.emissive.setHex(0x45c486);
        hoveredMesh.material.emissiveIntensity = 0.9;
      }
      showTooltip(hit.userData.cellData, event.clientX, event.clientY);
    } else {
      if (hoveredMesh && hoveredMesh.userData.baseColor) {
        hoveredMesh.material.emissive.setHex(hoveredMesh.userData.baseColor);
        hoveredMesh.material.emissiveIntensity = hoveredMesh.userData.cellData.max_impact * 0.4;
        hoveredMesh = null;
      }
      hideTooltip();
    }
  }

  function onMouseClick(event) {
    const rect = webglCanvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(pillarGroup.children);

    if (intersects.length > 0) {
      const cell = intersects[0].object.userData.cellData;
      selectCell(cell);
    }
  }

  function onWindowResize() {
    if (!camera || !renderer || !canvasWrapper) return;
    const width = canvasWrapper.clientWidth;
    const height = canvasWrapper.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }

  // =========================================================================
  // Tooltip & Tactical Sector Inspector
  // =========================================================================
  function showTooltip(cell, clientX, clientY) {
    if (!cell || !tooltip) return;
    tooltip.innerHTML = `
      <div style="font-weight:700; margin-bottom:2px;">Sector (X:${cell.x}, Y:${cell.y})</div>
      <div>Fire Risk: <b>${(cell.avg_impact || 0).toFixed(3)}</b></div>
      <div>Crews: <b>${cell.crews_assigned} / 4</b></div>
      <div style="font-size:10px; opacity:0.8;">Obs: ${cell.observation_count}</div>
    `;
    const rect = canvasWrapper.getBoundingClientRect();
    tooltip.style.left = `${clientX - rect.left + 12}px`;
    tooltip.style.top = `${clientY - rect.top + 12}px`;
    tooltip.style.display = 'block';
  }

  function hideTooltip() {
    if (tooltip) tooltip.style.display = 'none';
  }

  function selectCell(cell) {
    selectedCell = cell;
    if (inspectorCoordsBadge) inspectorCoordsBadge.textContent = `Sector (X:${cell.x}, Y:${cell.y})`;
    if (inspImpact) inspImpact.textContent = (cell.max_impact || cell.avg_impact || 0).toFixed(4);
    if (inspCrews) inspCrews.textContent = `${cell.crews_assigned} / 4`;

    if (cell.sample_weather) {
      const w = cell.sample_weather;
      if (inspTemp) inspTemp.textContent = `${w.temp.toFixed(1)}°C`;
      if (inspWind) inspWind.textContent = `${w.wind.toFixed(1)} km/h`;
      if (inspRh) inspRh.textContent = `${w.RH.toFixed(0)}%`;
      if (inspFfmc) inspFfmc.textContent = `${w.FFMC.toFixed(1)}`;
    } else {
      if (inspTemp) inspTemp.textContent = '--';
      if (inspWind) inspWind.textContent = '--';
      if (inspRh) inspRh.textContent = '--';
      if (inspFfmc) inspFfmc.textContent = '--';
    }
  }

  // =========================================================================
  // Canvas Fallback
  // =========================================================================
  let fallbackCtx = null;
  function initCanvasFallback() {
    fallbackCtx = webglCanvas.getContext('2d');
    if (!fallbackCtx) return;
    renderCanvasFallback();
  }

  function renderCanvasFallback() {
    if (!fallbackCtx) return;
    const ctx = fallbackCtx;
    const cw = webglCanvas.width = canvasWrapper.clientWidth || 700;
    const ch = webglCanvas.height = canvasWrapper.clientHeight || 320;

    ctx.fillStyle = '#06110b';
    ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = '#45c486';
    ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
    ctx.fillText('9×9 Montesinho Spatial Grid (Tactical Canvas Mode)', 20, 30);
  }

  // =========================================================================
  // Fetch & Synchronize State with Server
  // =========================================================================
  async function fetchGridData() {
    try {
      const res = await fetch('/api/grid');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();

      currentGridData = data.grid || [];
      currentCrews = data.crews || [];
      currentMetrics = data.metrics || {};

      render3DGrid();
      renderPortfolioTable();
      renderMatrixGrid();
      updateMetricsHUD();

      if (data.simulation) {
        if (hudStatusBadge) hudStatusBadge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:middle; margin-right:4px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>Scenario: <b>${data.simulation.preset}</b>`;
      } else {
        if (hudStatusBadge) hudStatusBadge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:middle; margin-right:4px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>Status: <b>Optimal Response Active</b>`;
      }
    } catch (err) {
      console.error("Failed to load grid data:", err);
    }
  }

  function updateMetricsHUD() {
    const m = currentMetrics;
    if (kpiCrewsDeployed) kpiCrewsDeployed.textContent = `${m.total_crews_selected || 25} / 25`;
    if (sidebarCrewCount) sidebarCrewCount.textContent = m.total_crews_selected || 25;
    if (kpiMaxPerCell) kpiMaxPerCell.textContent = `${m.max_per_cell_observed || 4} / 4`;
    if (kpiUniqueSectors) kpiUniqueSectors.textContent = `${m.unique_cells_covered || 13} Sectors`;
    if (kpiTotalEvalRows) kpiTotalEvalRows.textContent = `${m.total_eval_rows || 517} Fires`;
    if (kpiDatasetTag) kpiDatasetTag.textContent = m.eval_source || 'forestfires.csv';
    if (evalSourceLabel) evalSourceLabel.textContent = m.eval_source || 'forestfires.csv';

    if (kpiConstraintBadge) {
      const ok = m.constraint_satisfied;
      kpiConstraintBadge.textContent = ok ? 'OK' : 'LIMIT EXCEEDED';
      kpiConstraintBadge.style.background = ok ? 'rgba(255,255,255,0.2)' : '#ef4444';
    }

    if (m.has_ground_truth) {
      if (metricNdcg) metricNdcg.textContent = m.ndcg_at_25 !== undefined ? m.ndcg_at_25.toFixed(4) : '0.2288';
      if (metricRecall) metricRecall.textContent = m.high_impact_recall !== undefined ? (m.high_impact_recall * 100).toFixed(1) + '%' : '32.2%';
      if (metricRubric) metricRubric.textContent = m.rubric_total_score !== undefined ? m.rubric_total_score.toFixed(2) : '29.19';
    }
  }

  function renderPortfolioTable(filterText = '') {
    if (!portfolioTbody) return;
    portfolioTbody.innerHTML = '';

    if (currentCrews.length === 0) {
      portfolioTbody.innerHTML = `<tr><td colspan="10" style="text-align:center;">No responses selected.</td></tr>`;
      return;
    }

    const cellTally = {};
    currentCrews.forEach(c => {
      const key = `${c.x},${c.y}`;
      cellTally[key] = (cellTally[key] || 0) + 1;
    });

    currentCrews.forEach(crew => {
      const searchMatch = !filterText ||
        crew.priority_rank.toString().includes(filterText) ||
        `x:${crew.x}`.includes(filterText) ||
        `y:${crew.y}`.includes(filterText) ||
        `(${crew.x}, ${crew.y})`.toLowerCase().includes(filterText) ||
        crew.month.toLowerCase().includes(filterText) ||
        crew.day.toLowerCase().includes(filterText);

      if (!searchMatch) return;

      const countInCell = cellTally[`${crew.x},${crew.y}`];
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span style="font-weight:800; color:var(--primary-dark);">#${crew.priority_rank}</span></td>
        <td><span class="score-pill-badge">${crew.impact_score.toFixed(4)}</span></td>
        <td><b>(${crew.x}, ${crew.y})</b></td>
        <td>${crew.month} / ${crew.day}</td>
        <td>${crew.temp}°C</td>
        <td>${crew.wind} km/h</td>
        <td>${crew.RH}%</td>
        <td>${crew.FFMC}</td>
        <td>${crew.ISI}</td>
        <td><span style="font-weight:700; color:${countInCell <= 4 ? '#166534' : '#ef4444'}">${countInCell}/4</span></td>
      `;

      tr.addEventListener('click', () => {
        const matchingCell = currentGridData.find(g => g.x === crew.x && g.y === crew.y);
        if (matchingCell) selectCell(matchingCell);
      });

      portfolioTbody.appendChild(tr);
    });
  }

  function renderMatrixGrid() {
    if (!matrixGrid) return;
    matrixGrid.innerHTML = '';

    for (let y = 9; y >= 1; y--) {
      for (let x = 1; x <= 9; x++) {
        const cell = currentGridData.find(c => c.x === x && c.y === y) || {
          x: x, y: y, avg_impact: 0, max_impact: 0, observation_count: 0, crews_assigned: 0
        };

        const cellEl = document.createElement('div');
        cellEl.className = `matrix-cell ${cell.crews_assigned > 0 ? 'has-crews' : ''}`;

        const s = cell.max_impact || cell.avg_impact || 0;
        if (s > 0) {
          const hex = getScoreColor(s).toString(16).padStart(6, '0');
          cellEl.style.backgroundColor = `#${hex}33`;
          cellEl.style.borderColor = `#${hex}88`;
        }

        cellEl.innerHTML = `<span>${x},${y}</span>`;

        cellEl.addEventListener('mouseenter', (e) => showTooltip(cell, e.clientX, e.clientY));
        cellEl.addEventListener('mouseleave', hideTooltip);
        cellEl.addEventListener('click', () => selectCell(cell));

        matrixGrid.appendChild(cellEl);
      }
    }
  }

  // =========================================================================
  // Sliders & What-If Simulation
  // =========================================================================
  function setupSliders() {
    if (sliderTemp) {
      sliderTemp.addEventListener('input', () => {
        const val = parseInt(sliderTemp.value, 10);
        valTemp.textContent = `${val >= 0 ? '+' : ''}${val}.0°C`;
      });
    }

    if (sliderRh) {
      sliderRh.addEventListener('input', () => {
        const val = parseInt(sliderRh.value, 10);
        valRh.textContent = `${val >= 0 ? '+' : ''}${val}%`;
      });
    }

    if (sliderWind) {
      sliderWind.addEventListener('input', () => {
        const val = parseInt(sliderWind.value, 10);
        valWind.textContent = `${val >= 0 ? '+' : ''}${val}.0 km/h`;
      });
    }

    const btnRunSim = document.getElementById('btn-run-sim');
    if (btnRunSim) {
      btnRunSim.addEventListener('click', runSimulation);
    }
  }

  function setupPresetButtons() {
    const pHeatwave = document.getElementById('preset-heatwave');
    const pGale = document.getElementById('preset-gale');
    const pRain = document.getElementById('preset-rain');
    const pNormal = document.getElementById('preset-normal');

    if (pHeatwave) {
      pHeatwave.addEventListener('click', () => {
        sliderTemp.value = 8;
        sliderRh.value = -20;
        sliderWind.value = 6;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("Heatwave & Drought Spike");
      });
    }

    if (pGale) {
      pGale.addEventListener('click', () => {
        sliderTemp.value = 3;
        sliderRh.value = -10;
        sliderWind.value = 18;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("High Wind Gale");
      });
    }

    if (pRain) {
      pRain.addEventListener('click', () => {
        sliderTemp.value = -6;
        sliderRh.value = 25;
        sliderWind.value = -4;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("Heavy Precipitation");
      });
    }

    if (pNormal) {
      pNormal.addEventListener('click', () => {
        sliderTemp.value = 0;
        sliderRh.value = 0;
        sliderWind.value = 0;
        sliderTemp.dispatchEvent(new Event('input'));
        sliderRh.dispatchEvent(new Event('input'));
        sliderWind.dispatchEvent(new Event('input'));
        runSimulationPreset("Baseline Unshifted");
      });
    }
  }

  async function runSimulation() {
    const tempDelta = parseFloat(sliderTemp.value);
    const rhDelta = parseFloat(sliderRh.value);
    const windDelta = parseFloat(sliderWind.value);

    await executeSimulationPayload({
      temp_delta: tempDelta,
      rh_delta: rhDelta,
      wind_delta: windDelta,
      scenario_preset: `Custom Shift (${tempDelta > 0 ? '+' : ''}${tempDelta}°C, RH:${rhDelta > 0 ? '+' : ''}${rhDelta}%, W:${windDelta > 0 ? '+' : ''}${windDelta}km/h)`
    });
  }

  async function runSimulationPreset(name) {
    const tempDelta = parseFloat(sliderTemp.value);
    const rhDelta = parseFloat(sliderRh.value);
    const windDelta = parseFloat(sliderWind.value);

    await executeSimulationPayload({
      temp_delta: tempDelta,
      rh_delta: rhDelta,
      wind_delta: windDelta,
      scenario_preset: name
    });
  }

  async function executeSimulationPayload(payload) {
    const btn = document.getElementById('btn-run-sim');
    try {
      if (btn) {
        btn.innerHTML = `<span>Simulating...</span>`;
        btn.disabled = true;
      }

      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`Simulation failed: ${res.statusText}`);
      const data = await res.json();

      currentGridData = data.grid || [];
      currentCrews = data.crews || [];
      currentMetrics = data.metrics || {};

      render3DGrid();
      renderPortfolioTable();
      renderMatrixGrid();
      updateMetricsHUD();

      if (hudStatusBadge) {
        hudStatusBadge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:middle; margin-right:4px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>Scenario: <b>${payload.scenario_preset}</b>`;
      }
    } catch (err) {
      alert(`Simulation error: ${err.message}`);
    } finally {
      if (btn) {
        btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg><span>Simulate &amp; Rebalance Crews</span>`;
        btn.disabled = false;
      }
    }
  }

  window.resetToBaseline = async function () {
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      if (!res.ok) throw new Error("Reset failed");
      const data = await res.json();

      currentGridData = data.grid || [];
      currentCrews = data.crews || [];
      currentMetrics = data.metrics || {};

      render3DGrid();
      renderPortfolioTable();
      renderMatrixGrid();
      updateMetricsHUD();

      if (hudStatusBadge) {
        hudStatusBadge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:middle; margin-right:4px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>Status: <b>Optimal Baseline Active</b>`;
      }
    } catch (e) {
      alert(`Reset error: ${e.message}`);
    }
  };

  // =========================================================================
  // Upload CSV Handling
  // =========================================================================
  function setupUpload() {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('csv-file-input');
    const statusBox = document.getElementById('upload-status');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
      }
    });

    async function handleFileUpload(file) {
      if (!file.name.endsWith('.csv')) {
        alert("Please upload a valid .csv file.");
        return;
      }

      statusBox.style.display = 'block';
      statusBox.style.color = 'var(--primary-dark)';
      statusBox.textContent = `Uploading ${file.name} and executing ML inference...`;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload error');

        statusBox.style.color = '#166534';
        statusBox.textContent = `Uploaded ${file.name}! 25 crews optimized.`;

        currentGridData = data.grid || [];
        currentCrews = data.crews || [];
        currentMetrics = data.metrics || {};

        render3DGrid();
        renderPortfolioTable();
        renderMatrixGrid();
        updateMetricsHUD();
      } catch (err) {
        statusBox.style.color = '#ef4444';
        statusBox.textContent = `Error: ${err.message}`;
      }
    }
  }

  // =========================================================================
  // Search & Navigation
  // =========================================================================
  function setupSearchFilter() {
    const searchInput = document.getElementById('global-search');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
      const term = e.target.value.toLowerCase().trim();
      renderPortfolioTable(term);
    });

    window.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
        e.preventDefault();
        searchInput.focus();
      }
    });
  }

  window.scrollToSection = function (elementId) {
    const el = document.getElementById(elementId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // =========================================================================
  // Mission Operations Clock (Donezo Timer)
  // =========================================================================
  let timerSeconds = 1 * 3600 + 24 * 60 + 8;
  let isTimerRunning = true;
  let timerInterval = null;

  function formatTime(totalSecs) {
    const hrs = Math.floor(totalSecs / 3600);
    const mins = Math.floor((totalSecs % 3600) / 60);
    const secs = totalSecs % 60;
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function updateTimerDisplay() {
    const display = document.getElementById('timerDigitalDisplay');
    if (display) display.textContent = formatTime(timerSeconds);
  }

  function setupMissionTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      if (isTimerRunning) {
        timerSeconds++;
        updateTimerDisplay();
      }
    }, 1000);
  }

  window.toggleTimer = function () {
    isTimerRunning = !isTimerRunning;
    const pauseIcon = document.getElementById('timerPauseIcon');
    if (!isTimerRunning) {
      pauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
    } else {
      pauseIcon.innerHTML = `
        <rect x="6" y="4" width="4" height="16" rx="1"></rect>
        <rect x="14" y="4" width="4" height="16" rx="1"></rect>
      `;
    }
  };

  window.resetTimer = function () {
    isTimerRunning = false;
    timerSeconds = 0;
    updateTimerDisplay();
    const pauseIcon = document.getElementById('timerPauseIcon');
    if (pauseIcon) {
      pauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
    }
  };

  // Kickstart on DOM content loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
