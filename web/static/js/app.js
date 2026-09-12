/**
 * MONTESINHO WILDFIRE RESPONSE 3D COMMAND CENTER
 * Frontend Application & 3D Spatial Grid Engine
 */

(function () {
  'use strict';

  // State
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
  const assetsList = document.getElementById('assets-list');

  // Slider Elements
  const sliderTemp = document.getElementById('slider-temp');
  const sliderRh = document.getElementById('slider-rh');
  const sliderWind = document.getElementById('slider-wind');
  const valTemp = document.getElementById('val-temp-delta');
  const valRh = document.getElementById('val-rh-delta');
  const valWind = document.getElementById('val-wind-delta');

  // Metric Elements
  const metricCrews = document.getElementById('metric-crews');
  const metricMaxCell = document.getElementById('metric-max-cell');
  const metricUniqueCells = document.getElementById('metric-unique-cells');
  const metricEvalRows = document.getElementById('metric-eval-rows');
  const metricNdcg = document.getElementById('metric-ndcg');
  const metricSpearman = document.getElementById('metric-spearman');
  const metricRecall = document.getElementById('metric-recall');
  const badgeConstraint = document.getElementById('badge-constraint');
  const constraintIndicator = document.getElementById('constraint-indicator');
  const hudDataset = document.getElementById('hud-dataset-name');
  const hudSimStatus = document.getElementById('hud-sim-status');

  // Inspector Elements
  const inspectorCoords = document.getElementById('inspector-coords');
  const inspectorBody = document.getElementById('inspector-body');

  // =========================================================================
  // Initialize Application
  // =========================================================================
  async function init() {
    setupTabs();
    setupSliders();
    setupPresetButtons();
    setupUpload();
    setupExportButtons();

    if (isThreeAvailable) {
      initThreeScene();
    } else {
      console.warn("Three.js not loaded, using isometric Canvas fallback.");
      initCanvasFallback();
    }

    await fetchGridData();
    await fetchAssets();
  }

  // =========================================================================
  // 3D Three.js Spatial Grid Visualization
  // =========================================================================
  function initThreeScene() {
    try {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x040810);
      scene.fog = new THREE.FogExp2(0x040810, 0.015);

      const width = canvasWrapper.clientWidth || 800;
      const height = canvasWrapper.clientHeight || 520;

      camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      // Isometric initial view
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

      // Controls
      if (typeof THREE.OrbitControls !== 'undefined') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground
        controls.minDistance = 8;
        controls.maxDistance = 70;
        controls.target.set(0, 0, 0);
      }

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0x00f2fe, 1.2);
      dirLight.position.set(15, 30, 20);
      dirLight.castShadow = true;
      dirLight.shadow.mapSize.width = 1024;
      dirLight.shadow.mapSize.height = 1024;
      scene.add(dirLight);

      const fireLight = new THREE.PointLight(0xff3b00, 1.5, 35);
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
      console.warn("WebGL initialization failed, falling back to 2.5D Canvas:", err);
      isThreeAvailable = false;
      initCanvasFallback();
    }
  }

  function createTextSprite(text, color = '#00f2fe', fontSize = 28) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = `bold ${fontSize}px "SF Mono", Consolas, monospace`;
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

    // X Axis labels along the South edge (Z = +10.2)
    for (let x = 1; x <= 9; x++) {
      const wx = (x - 5) * 2;
      const sprite = createTextSprite(`X:${x}`, '#00f2fe', 24);
      sprite.position.set(wx, 0.15, 10.2);
      labelGroup.add(sprite);
    }

    // Y Axis labels along the West edge (X = -10.2)
    for (let y = 1; y <= 9; y++) {
      const wz = -(y - 5) * 2;
      const sprite = createTextSprite(`Y:${y}`, '#f59e0b', 24);
      sprite.position.set(-10.2, 0.15, wz);
      labelGroup.add(sprite);
    }

    // Axis titles
    const labelX = createTextSprite('X (WEST → EAST)', '#00f2fe', 18);
    labelX.scale.set(4.5, 1.1, 1.0);
    labelX.position.set(0, 0.15, 11.6);
    labelGroup.add(labelX);

    const labelY = createTextSprite('Y (SOUTH → NORTH)', '#f59e0b', 18);
    labelY.scale.set(4.5, 1.1, 1.0);
    labelY.position.set(-11.8, 0.15, 0);
    labelGroup.add(labelY);
  }

  function buildGroundPlane() {
    // 9x9 grid representation (centered at 0, 0)
    // Step size = 2 units per cell. Grid span = 9 * 2 = 18 units.
    const size = 18;
    const divisions = 9;

    const gridHelper = new THREE.GridHelper(size, divisions, 0x00f2fe, 0x142a42);
    gridHelper.position.y = -0.01;
    gridHelperGroup.add(gridHelper);

    // Glowing base plane
    const planeGeo = new THREE.PlaneGeometry(size + 2, size + 2);
    const planeMat = new THREE.MeshBasicMaterial({
      color: 0x071322,
      depthWrite: false
    });
    const planeMesh = new THREE.Mesh(planeGeo, planeMat);
    planeMesh.rotation.x = -Math.PI / 2;
    planeMesh.position.y = -0.05;
    gridHelperGroup.add(planeMesh);
  }

  // Convert (X, Y) where X, Y in [1..9] to 3D world coordinates (xPos, zPos)
  function cellToWorld(x, y) {
    // X goes from 1 (West) to 9 (East) -> world X from -8 to +8 (step = 2)
    const xPos = (x - 5) * 2;
    // Y goes from 1 (South) to 9 (North) -> world Z from +8 (South) to -8 (North)
    const zPos = -(y - 5) * 2;
    return { x: xPos, z: zPos };
  }

  // Color interpolation for fire impact score (0.0 to 1.0)
  function getScoreColor(score) {
    const s = Math.max(0, Math.min(1, score));
    // Color stops: Low (0.0: Teal/Forest) -> Mid (0.4: Amber) -> High (0.8+: Flame Red)
    if (s < 0.35) {
      // 0x00a86b to 0xf59e0b
      const t = s / 0.35;
      const r = Math.round(16 + t * (245 - 16));
      const g = Math.round(168 + t * (158 - 168));
      const b = Math.round(107 + t * (11 - 107));
      return (r << 16) | (g << 8) | b;
    } else if (s < 0.70) {
      // 0xf59e0b to 0xff5500
      const t = (s - 0.35) / 0.35;
      const r = Math.round(245 + t * (255 - 245));
      const g = Math.round(158 + t * (85 - 158));
      const b = Math.round(11 + t * (0 - 11));
      return (r << 16) | (g << 8) | b;
    } else {
      // 0xff5500 to 0xff0033
      const t = (s - 0.70) / 0.30;
      const r = 255;
      const g = Math.round(85 * (1 - t));
      const b = Math.round(51 * t);
      return (r << 16) | (g << 8) | b;
    }
  }

  function render3DGrid() {
    if (!isThreeAvailable || !pillarGroup) {
      renderCanvasFallback();
      return;
    }

    // Clear existing pillars and crews
    while (pillarGroup.children.length > 0) {
      const obj = pillarGroup.children[0];
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
      pillarGroup.remove(obj);
    }

    while (crewGroup.children.length > 0) {
      const obj = crewGroup.children[0];
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
      crewGroup.remove(obj);
    }

    // Build 81 cell pillars
    currentGridData.forEach(cell => {
      const { x: cx, y: cy } = cell;
      const { x: wx, z: wz } = cellToWorld(cx, cy);

      // Height scaled with fire impact (minimum 0.3, max 6.0)
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

      // If cell has response crew assignments (1 to 4)
      if (cell.crews_assigned > 0) {
        renderCrewMarkersForCell(cell, wx, wz, height);
      }
    });
  }

  function renderCrewMarkersForCell(cell, wx, wz, pillarHeight) {
    const crewCount = Math.min(4, cell.crews_assigned);
    const topY = pillarHeight;

    // Arrange up to 4 crews in a 2x2 offset pattern on the top face
    const offsets = [
      { dx: 0, dz: 0 },                       // 1 crew: centered
      { dx: -0.4, dz: 0 }, { dx: 0.4, dz: 0 }, // 2 crews
      { dx: -0.4, dz: -0.4 }, { dx: 0.4, dz: -0.4 }, { dx: 0, dz: 0.4 }, // 3 crews
      { dx: -0.4, dz: -0.4 }, { dx: 0.4, dz: -0.4 }, { dx: -0.4, dz: 0.4 }, { dx: 0.4, dz: 0.4 } // 4 crews
    ];

    let cellOffsets;
    if (crewCount === 1) cellOffsets = [offsets[0]];
    else if (crewCount === 2) cellOffsets = [offsets[1], offsets[2]];
    else if (crewCount === 3) cellOffsets = [offsets[3], offsets[4], offsets[5]];
    else cellOffsets = [offsets[6], offsets[7], offsets[8], offsets[9]];

    // Top Glowing Ring on Cell
    const ringGeo = new THREE.RingGeometry(0.7, 0.82, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f2fe,
      side: THREE.DoubleSide
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.set(wx, topY + 0.02, wz);
    crewGroup.add(ringMesh);

    // Place Crew Pin Markers
    cellOffsets.forEach((pos, idx) => {
      const pinX = wx + pos.dx;
      const pinZ = wz + pos.dz;

      // Cone Tactical Pin
      const pinGeo = new THREE.ConeGeometry(0.22, 0.7, 8);
      const pinMat = new THREE.MeshStandardMaterial({
        color: 0x00f2fe,
        emissive: 0x00f2fe,
        emissiveIntensity: 0.8,
        metalness: 0.9,
        roughness: 0.1
      });
      const pinMesh = new THREE.Mesh(pinGeo, pinMat);
      // Invert cone so tip points at the cell top
      pinMesh.rotation.x = Math.PI;
      pinMesh.position.set(pinX, topY + 0.5, pinZ);
      crewGroup.add(pinMesh);

      // Floating Glow Sphere above pin
      const sphereGeo = new THREE.SphereGeometry(0.14, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({
        color: 0xffffff
      });
      const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
      sphereMesh.position.set(pinX, topY + 0.95, pinZ);
      crewGroup.add(sphereMesh);
    });
  }

  function setupViewControls() {
    document.getElementById('view-iso').addEventListener('click', () => {
      setCameraPreset(16, 22, 24);
      setActiveViewBtn('view-iso');
    });

    document.getElementById('view-top').addEventListener('click', () => {
      setCameraPreset(0, 32, 0.01);
      setActiveViewBtn('view-top');
    });

    document.getElementById('view-low').addEventListener('click', () => {
      setCameraPreset(0, 5, 26);
      setActiveViewBtn('view-low');
    });

    const btnRotate = document.getElementById('view-rotate');
    btnRotate.addEventListener('click', () => {
      isAutoRotating = !isAutoRotating;
      if (controls) controls.autoRotate = isAutoRotating;
      btnRotate.textContent = isAutoRotating ? 'Auto-Rotate: On' : 'Auto-Rotate: Off';
      btnRotate.classList.toggle('active', isAutoRotating);
    });

    // Layer toggles
    document.getElementById('toggle-pillars').addEventListener('click', function () {
      this.classList.toggle('active');
      if (pillarGroup) pillarGroup.visible = this.classList.contains('active');
      if (!isThreeAvailable) renderCanvasFallback();
    });

    document.getElementById('toggle-crews').addEventListener('click', function () {
      this.classList.toggle('active');
      if (crewGroup) crewGroup.visible = this.classList.contains('active');
      if (!isThreeAvailable) renderCanvasFallback();
    });

    document.getElementById('toggle-grid-coords').addEventListener('click', function () {
      this.classList.toggle('active');
      const isVis = this.classList.contains('active');
      if (gridHelperGroup) gridHelperGroup.visible = isVis;
      if (labelGroup) labelGroup.visible = isVis;
      if (!isThreeAvailable) renderCanvasFallback();
    });
  }

  function setActiveViewBtn(id) {
    ['view-iso', 'view-top', 'view-low'].forEach(bId => {
      document.getElementById(bId).classList.toggle('active', bId === id);
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
        hoveredMesh.material.emissive.setHex(0x00f2fe);
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
  // Tooltip & Tactical Inspector
  // =========================================================================
  function showTooltip(cell, clientX, clientY) {
    if (!cell || !tooltip) return;

    document.getElementById('tt-cell-coord').textContent = `Cell (X=${cell.x}, Y=${cell.y})`;
    const crewBadge = document.getElementById('tt-cell-crews');
    crewBadge.textContent = `${cell.crews_assigned} Crew${cell.crews_assigned === 1 ? '' : 's'}`;
    crewBadge.className = cell.crews_assigned > 0 ? 'badge badge-cyan' : 'badge badge-emerald';

    document.getElementById('tt-cell-impact').textContent = (cell.avg_impact || 0).toFixed(3);
    document.getElementById('tt-cell-obs').textContent = cell.observation_count;
    document.getElementById('tt-cell-max-impact').textContent = (cell.max_impact || 0).toFixed(3);
    document.getElementById('tt-cell-quota').textContent = `${cell.crews_assigned} / 4 max`;

    const weatherSnippet = document.getElementById('tt-weather-snippet');
    if (cell.sample_weather) {
      const w = cell.sample_weather;
      weatherSnippet.textContent = `Temp: ${w.temp.toFixed(1)}°C | Wind: ${w.wind.toFixed(1)} km/h | RH: ${w.RH.toFixed(0)}%`;
    } else {
      weatherSnippet.textContent = "No observations recorded in this sector.";
    }

    tooltip.style.display = 'block';
  }

  function hideTooltip() {
    if (tooltip) tooltip.style.display = 'none';
  }

  function selectCell(cell) {
    selectedCell = cell;
    inspectorCoords.textContent = `Sector (X=${cell.x}, Y=${cell.y})`;

    let html = `
      <div style="margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
        <span>Average Risk: <b style="color:var(--accent-cyan)">${(cell.avg_impact || 0).toFixed(4)}</b></span>
        <span class="badge ${cell.crews_assigned > 0 ? 'badge-cyan' : 'badge-emerald'}">
          Quota: ${cell.crews_assigned}/4 Crews
        </span>
      </div>
      <div style="font-size:0.72rem; color:var(--text-secondary); margin-bottom:12px;">
        <div>Total Observations: <b>${cell.observation_count}</b></div>
        <div>Peak Observed Risk: <b>${(cell.max_impact || 0).toFixed(4)}</b></div>
      </div>
    `;

    if (cell.crew_details && cell.crew_details.length > 0) {
      html += `<div style="font-weight:700; color:var(--accent-cyan); margin-bottom:6px;">Assigned Responses (${cell.crew_details.length}):</div>`;
      html += `<ul style="list-style:none; padding-left:0; display:flex; flex-direction:column; gap:6px;">`;
      cell.crew_details.forEach(cr => {
        html += `
          <li style="background:rgba(0,242,254,0.08); border:1px solid rgba(0,242,254,0.2); padding:6px 8px; border-radius:4px; font-family:var(--font-mono);">
            <div style="display:flex; justify-content:space-between;">
              <span style="color:var(--accent-cyan); font-weight:bold;"><i class="nf nf-fa-bullseye"></i> Rank #${cr.priority_rank}</span>
              <span>Score: <b>${cr.impact_score}</b></span>
            </div>
            <div style="color:var(--text-muted); font-size:0.68rem; margin-top:2px;">
              ${cr.month} ${cr.day} | <i class="nf nf-weather-thermometer"></i> ${cr.temp}°C | <i class="nf nf-weather-windy"></i> ${cr.wind} km/h | <i class="nf nf-weather-humidity"></i> ${cr.RH}%
            </div>
          </li>
        `;
      });
      html += `</ul>`;
    } else {
      html += `<p style="color:var(--text-muted); font-style:italic;">No crews deployed to this cell in current portfolio.</p>`;
    }

    inspectorBody.innerHTML = html;
  }

  // =========================================================================
  // Canvas Isometric Fallback (Graceful offline/non-WebGL mode)
  // =========================================================================
  let fallbackCtx = null;

  function initCanvasFallback() {
    fallbackCtx = webglCanvas.getContext('2d');
    if (!fallbackCtx) return;

    function resizeFallbackCanvas() {
      webglCanvas.width = canvasWrapper.clientWidth || 800;
      webglCanvas.height = canvasWrapper.clientHeight || 520;
      renderCanvasFallback();
    }
    resizeFallbackCanvas();
    window.addEventListener('resize', resizeFallbackCanvas);

    webglCanvas.addEventListener('mousemove', onCanvasFallbackMouseMove);
    webglCanvas.addEventListener('click', onCanvasFallbackClick);
    renderCanvasFallback();
  }

  function getFallbackCellCoords(x, y) {
    const cw = webglCanvas.width;
    const ch = webglCanvas.height;
    const originX = cw / 2;
    const originY = ch * 0.42;
    const tileW = Math.max(18, Math.min(30, cw / 26));
    const tileH = tileW * 0.52;

    // Isometric projection: X increases to lower-right, Y increases to upper-right
    const screenX = originX + (x - 5) * tileW - (y - 5) * tileW;
    const screenY = originY + (x - 5) * tileH + (y - 5) * tileH;
    return { screenX, screenY, tileW, tileH };
  }

  function renderCanvasFallback() {
    if (!fallbackCtx) return;
    const ctx = fallbackCtx;
    const cw = webglCanvas.width;
    const ch = webglCanvas.height;

    ctx.fillStyle = '#040810';
    ctx.fillRect(0, 0, cw, ch);

    // Title banner
    ctx.fillStyle = 'rgba(0, 242, 254, 0.85)';
    ctx.font = 'bold 12px "SF Mono", Consolas, monospace';
    ctx.fillText('MONTESINHO 9×9 SPATIAL SECTOR GRID (2.5D ISOMETRIC VIEW)', 18, 26);
    ctx.fillStyle = '#64748b';
    ctx.font = '11px "SF Mono", Consolas, monospace';
    ctx.fillText('Autonomous Dispatch Mode • Click cell to inspect', 18, 44);

    const coordsVisible = document.getElementById('toggle-grid-coords') ?
      document.getElementById('toggle-grid-coords').classList.contains('active') : true;
    const pillarsVisible = document.getElementById('toggle-pillars') ?
      document.getElementById('toggle-pillars').classList.contains('active') : true;
    const crewsVisible = document.getElementById('toggle-crews') ?
      document.getElementById('toggle-crews').classList.contains('active') : true;

    // Draw cells back-to-front (sorted by X + Y ascending)
    const sortedCells = [...currentGridData].sort((a, b) => (a.x + a.y) - (b.x + b.y));

    sortedCells.forEach(cell => {
      const { screenX, screenY, tileW, tileH } = getFallbackCellCoords(cell.x, cell.y);
      const baseScore = cell.max_impact || cell.avg_impact || 0;
      const height = pillarsVisible ? Math.max(4, baseScore * 50 + (cell.observation_count > 0 ? 6 : 2)) : 0;
      const hexColor = '#' + getScoreColor(baseScore).toString(16).padStart(6, '0');

      // Top face rhombus offset upward by height
      const topY = screenY - height;

      // Draw sides if height > 0
      if (height > 0) {
        // Left side
        ctx.beginPath();
        ctx.moveTo(screenX - tileW, topY);
        ctx.lineTo(screenX, topY + tileH);
        ctx.lineTo(screenX, screenY + tileH);
        ctx.lineTo(screenX - tileW, screenY);
        ctx.closePath();
        ctx.fillStyle = 'rgba(10, 25, 45, 0.7)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 242, 254, 0.2)';
        ctx.stroke();

        // Right side
        ctx.beginPath();
        ctx.moveTo(screenX, topY + tileH);
        ctx.lineTo(screenX + tileW, topY);
        ctx.lineTo(screenX + tileW, screenY);
        ctx.lineTo(screenX, screenY + tileH);
        ctx.closePath();
        ctx.fillStyle = 'rgba(15, 35, 60, 0.85)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 242, 254, 0.2)';
        ctx.stroke();
      }

      // Top Face Rhombus
      ctx.beginPath();
      ctx.moveTo(screenX, topY - tileH);
      ctx.lineTo(screenX + tileW, topY);
      ctx.lineTo(screenX, topY + tileH);
      ctx.lineTo(screenX - tileW, topY);
      ctx.closePath();
      ctx.fillStyle = hexColor + (cell.observation_count > 0 ? 'e6' : '77');
      ctx.fill();
      ctx.strokeStyle = cell.crews_assigned > 0 ? '#00f2fe' : 'rgba(0, 242, 254, 0.35)';
      ctx.lineWidth = cell.crews_assigned > 0 ? 2 : 1;
      ctx.stroke();

      // Coordinates text
      if (coordsVisible) {
        ctx.fillStyle = '#ffffff';
        ctx.font = '9px "SF Mono", Consolas, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${cell.x},${cell.y}`, screenX, topY);
      }

      // Crew Pin Indicator
      if (crewsVisible && cell.crews_assigned > 0) {
        ctx.beginPath();
        ctx.arc(screenX, topY - 8, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#00f2fe';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#040810';
        ctx.font = 'bold 9px "SF Mono", Consolas, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${cell.crews_assigned}`, screenX, topY - 8);
      }
    });
  }

  function getFallbackCellAtPos(clientX, clientY) {
    const rect = webglCanvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;

    let closest = null;
    let minD = Infinity;

    currentGridData.forEach(cell => {
      const { screenX, screenY, tileW } = getFallbackCellCoords(cell.x, cell.y);
      const baseScore = cell.max_impact || cell.avg_impact || 0;
      const height = Math.max(4, baseScore * 50 + (cell.observation_count > 0 ? 6 : 2));
      const topY = screenY - height;
      const d = Math.hypot(mx - screenX, my - topY);
      if (d < tileW * 1.1 && d < minD) {
        minD = d;
        closest = cell;
      }
    });
    return closest;
  }

  function onCanvasFallbackMouseMove(e) {
    if (isThreeAvailable) return;
    const cell = getFallbackCellAtPos(e.clientX, e.clientY);
    if (cell) {
      showTooltip(cell, e.clientX, e.clientY);
    } else {
      hideTooltip();
    }
  }

  function onCanvasFallbackClick(e) {
    if (isThreeAvailable) return;
    const cell = getFallbackCellAtPos(e.clientX, e.clientY);
    if (cell) {
      selectCell(cell);
    }
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
        hudSimStatus.textContent = `Active: ${data.simulation.preset}`;
        hudSimStatus.style.color = 'var(--accent-fire)';
      } else {
        hudSimStatus.textContent = 'Optimal Baseline Dispatch';
        hudSimStatus.style.color = 'var(--accent-emerald)';
      }
    } catch (err) {
      console.error("Failed to load grid data:", err);
    }
  }

  function updateMetricsHUD() {
    const m = currentMetrics;
    if (metricCrews) metricCrews.textContent = m.total_crews_selected || 25;
    if (metricMaxCell) {
      metricMaxCell.textContent = m.max_per_cell_observed || 4;
      metricMaxCell.className = (m.max_per_cell_observed <= 4) ? 'metric-val emerald' : 'metric-val fire';
    }
    if (metricUniqueCells) metricUniqueCells.textContent = m.unique_cells_covered || '--';
    if (metricEvalRows) metricEvalRows.textContent = m.total_eval_rows || '--';

    if (badgeConstraint) {
      const ok = m.constraint_satisfied;
      badgeConstraint.textContent = ok ? 'MAX 4 PER CELL: OK' : 'CONSTRAINT VIOLATED';
      badgeConstraint.className = ok ? 'badge badge-emerald' : 'badge badge-fire';
    }

    if (constraintIndicator) {
      const ok = m.constraint_satisfied;
      constraintIndicator.textContent = ok ? 'MAX ≤ 4 OK' : 'VIOLATION (>4)';
      constraintIndicator.className = ok ? 'badge badge-emerald' : 'badge badge-fire';
    }

    if (hudDataset) hudDataset.textContent = m.eval_source || 'forestfires.csv';

    // Ground truth metrics (if available)
    if (m.has_ground_truth) {
      if (metricNdcg) metricNdcg.textContent = m.ndcg_at_25 !== undefined ? m.ndcg_at_25.toFixed(4) : '--';
      if (metricSpearman) metricSpearman.textContent = m.spearman_corr !== undefined ? m.spearman_corr.toFixed(4) : '--';
      if (metricRecall) metricRecall.textContent = m.high_impact_recall !== undefined ? (m.high_impact_recall * 100).toFixed(1) + '%' : '--';
    }
  }

  function renderPortfolioTable() {
    if (!portfolioTbody) return;
    portfolioTbody.innerHTML = '';

    if (currentCrews.length === 0) {
      portfolioTbody.innerHTML = `<tr><td colspan="10" style="text-align:center;">No responses selected.</td></tr>`;
      return;
    }

    // Tally cell assignments
    const cellTally = {};
    currentCrews.forEach(c => {
      const key = `${c.x},${c.y}`;
      cellTally[key] = (cellTally[key] || 0) + 1;
    });

    currentCrews.forEach(crew => {
      const countInCell = cellTally[`${crew.x},${crew.y}`];
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="rank-badge"><i class="nf nf-fa-hashtag" style="font-size:0.8em;opacity:0.8;"></i>${crew.priority_rank}</span></td>
        <td style="color:var(--accent-cyan); font-weight:bold; font-family:var(--font-mono);">${crew.impact_score.toFixed(4)}</td>
        <td><b style="font-family:var(--font-mono);">(${crew.x}, ${crew.y})</b></td>
        <td style="font-family:var(--font-mono);">${crew.month} / ${crew.day}</td>
        <td style="font-family:var(--font-mono);">${crew.temp}°C</td>
        <td style="font-family:var(--font-mono);">${crew.wind}</td>
        <td style="font-family:var(--font-mono);">${crew.RH}%</td>
        <td style="font-family:var(--font-mono);">${crew.FFMC}</td>
        <td style="font-family:var(--font-mono);">${crew.ISI}</td>
        <td><span class="badge ${countInCell <= 4 ? 'badge-emerald' : 'badge-fire'}"><i class="nf ${countInCell <= 4 ? 'nf-fa-circle_check' : 'nf-fa-triangle_exclamation'}"></i> ${countInCell}/4</span></td>
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

    // Render 9x9 matrix: rows Y from 9 down to 1, columns X from 1 to 9
    for (let y = 9; y >= 1; y--) {
      for (let x = 1; x <= 9; x++) {
        const cell = currentGridData.find(c => c.x === x && c.y === y) || {
          x: x, y: y, avg_impact: 0, max_impact: 0, observation_count: 0, crews_assigned: 0
        };

        const cellEl = document.createElement('div');
        cellEl.className = 'matrix-cell';

        // Color cell background based on score
        const s = cell.max_impact || cell.avg_impact || 0;
        if (s > 0) {
          const hex = getScoreColor(s).toString(16).padStart(6, '0');
          cellEl.style.backgroundColor = `#${hex}33`;
          cellEl.style.borderColor = `#${hex}88`;
        }

        cellEl.innerHTML = `
          <span>${x},${y}</span>
          ${cell.crews_assigned > 0 ? `<div class="cell-crews">${cell.crews_assigned}</div>` : ''}
        `;

        cellEl.addEventListener('mouseenter', (e) => showTooltip(cell, e.clientX, e.clientY));
        cellEl.addEventListener('mouseleave', hideTooltip);
        cellEl.addEventListener('click', () => selectCell(cell));

        matrixGrid.appendChild(cellEl);
      }
    }
  }

  async function fetchAssets() {
    if (!assetsList) return;
    try {
      const res = await fetch('/api/assets');
      if (!res.ok) return;
      const data = await res.json();

      assetsList.innerHTML = '';
      data.available_slots.forEach(slot => {
        const item = document.createElement('div');
        item.style.cssText = "background:rgba(15,26,44,0.8); border:1px solid rgba(0,242,254,0.15); padding:10px; border-radius:6px; display:flex; align-items:center; gap:10px;";
        item.innerHTML = `
          <div style="width:36px; height:36px; background:rgba(0,242,254,0.1); border-radius:4px; display:flex; align-items:center; justify-content:center; overflow:hidden;">
            ${/\.(svg|png|jpg|jpeg|gif|webp)$/i.test(slot.filename) ? `<img src="${slot.slot_path}" style="max-width:28px; max-height:28px;" alt="${slot.filename}">` : `<span style="color:var(--accent-cyan); font-size:0.6rem;">ASSET</span>`}
          </div>
          <div>
            <div style="color:var(--text-primary); font-weight:bold;">${slot.filename}</div>
            <div style="color:var(--text-muted); font-size:0.65rem;">${slot.slot_path}</div>
          </div>
        `;
        assetsList.appendChild(item);
      });
    } catch (e) {
      console.warn("Assets API failed:", e);
    }
  }

  // =========================================================================
  // Sliders & What-If Simulation
  // =========================================================================
  function setupSliders() {
    sliderTemp.addEventListener('input', () => {
      const val = parseInt(sliderTemp.value, 10);
      valTemp.textContent = `${val >= 0 ? '+' : ''}${val}.0°C`;
    });

    sliderRh.addEventListener('input', () => {
      const val = parseInt(sliderRh.value, 10);
      valRh.textContent = `${val >= 0 ? '+' : ''}${val}%`;
    });

    sliderWind.addEventListener('input', () => {
      const val = parseInt(sliderWind.value, 10);
      valWind.textContent = `${val >= 0 ? '+' : ''}${val}.0 km/h`;
    });

    document.getElementById('btn-run-sim').addEventListener('click', runSimulation);
  }

  function setupPresetButtons() {
    document.getElementById('preset-heatwave').addEventListener('click', () => {
      sliderTemp.value = 8;
      sliderRh.value = -20;
      sliderWind.value = 6;
      sliderTemp.dispatchEvent(new Event('input'));
      sliderRh.dispatchEvent(new Event('input'));
      sliderWind.dispatchEvent(new Event('input'));
      runSimulationPreset("Summer Heatwave & Drought");
    });

    document.getElementById('preset-gale').addEventListener('click', () => {
      sliderTemp.value = 3;
      sliderRh.value = -10;
      sliderWind.value = 18;
      sliderTemp.dispatchEvent(new Event('input'));
      sliderRh.dispatchEvent(new Event('input'));
      sliderWind.dispatchEvent(new Event('input'));
      runSimulationPreset("High Wind Gale");
    });

    document.getElementById('preset-rain').addEventListener('click', () => {
      sliderTemp.value = -6;
      sliderRh.value = 25;
      sliderWind.value = -4;
      sliderTemp.dispatchEvent(new Event('input'));
      sliderRh.dispatchEvent(new Event('input'));
      sliderWind.dispatchEvent(new Event('input'));
      runSimulationPreset("Autumn Rain");
    });

    document.getElementById('preset-normal').addEventListener('click', () => {
      sliderTemp.value = 0;
      sliderRh.value = 0;
      sliderWind.value = 0;
      sliderTemp.dispatchEvent(new Event('input'));
      sliderRh.dispatchEvent(new Event('input'));
      sliderWind.dispatchEvent(new Event('input'));
      runSimulationPreset("Baseline Weather (Unshifted)");
    });
  }

  async function runSimulation() {
    const tempDelta = parseFloat(sliderTemp.value);
    const rhDelta = parseFloat(sliderRh.value);
    const windDelta = parseFloat(sliderWind.value);

    await executeSimulationPayload({
      temp_delta: tempDelta,
      rh_delta: rhDelta,
      wind_delta: windDelta,
      scenario_preset: `Custom Shift (T:${tempDelta > 0 ? '+' : ''}${tempDelta}°C, RH:${rhDelta > 0 ? '+' : ''}${rhDelta}%, W:${windDelta > 0 ? '+' : ''}${windDelta}km/h)`
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
    try {
      const btn = document.getElementById('btn-run-sim');
      btn.textContent = 'Simulating...';
      btn.disabled = true;

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

      hudSimStatus.textContent = `Active: ${payload.scenario_preset}`;
      hudSimStatus.style.color = 'var(--accent-fire)';
    } catch (err) {
      alert(`Simulation error: ${err.message}`);
    } finally {
      const btn = document.getElementById('btn-run-sim');
      btn.textContent = 'Simulate & Re-Allocate Crews';
      btn.disabled = false;
    }
  }

  async function resetBaseline() {
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

      hudSimStatus.textContent = 'Optimal Baseline Dispatch';
      hudSimStatus.style.color = 'var(--accent-emerald)';
    } catch (e) {
      alert(`Reset error: ${e.message}`);
    }
  }

  // =========================================================================
  // Upload CSV Handling
  // =========================================================================
  function setupUpload() {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('csv-file-input');
    const statusBox = document.getElementById('upload-status');

    dropzone.addEventListener('click', () => fileInput.click());

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
      statusBox.style.color = 'var(--accent-cyan)';
      statusBox.textContent = `Uploading ${file.name} and executing Hurdle-Rank inference...`;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || 'Upload error');
        }

        statusBox.style.color = 'var(--accent-emerald)';
        statusBox.textContent = `✓ Uploaded ${file.name}! 25 crews optimized.`;

        currentGridData = data.grid || [];
        currentCrews = data.crews || [];
        currentMetrics = data.metrics || {};

        render3DGrid();
        renderPortfolioTable();
        renderMatrixGrid();
        updateMetricsHUD();
      } catch (err) {
        statusBox.style.color = 'var(--accent-red)';
        statusBox.textContent = `Error: ${err.message}`;
      }
    }
  }

  // =========================================================================
  // Export & Tab Controls
  // =========================================================================
  function setupExportButtons() {
    document.getElementById('btn-reset-baseline').addEventListener('click', resetBaseline);
  }

  function setupTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        tab.classList.add('active');
        const target = tab.getAttribute('data-tab');
        document.getElementById(target).classList.add('active');
      });
    });
  }

  // Kickstart on DOM content loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
