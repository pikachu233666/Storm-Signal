import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const stage = document.querySelector(".earth-stage");
const canvas = document.querySelector(".earth-canvas");
const hero = document.querySelector(".hero");
const intro = document.querySelector(".intro");
const textNavigation = document.querySelector(".text-navigation");
const starNavigation = document.querySelector(".star-navigation");
const startButton = document.querySelector(".start-button");
const status = document.querySelector(".motion-status");
const countryPanel = document.querySelector(".country-panel");
const countryPanelClose = document.querySelector(".country-panel-close");
const selectedRegionName = document.querySelector("[data-selected-region-name]");
const selectedRegionSummary = document.querySelector("[data-selected-region-summary]");
const regionDetail = document.querySelector(".region-detail");
const detailCountryName = document.querySelector("[data-detail-country]");
const detailContinent = document.querySelector("[data-detail-continent]");
const detailCountryCode = document.querySelector("[data-detail-code]");
const detailPopulation = document.querySelector("[data-detail-population]");
const detailBoundaryNote = document.querySelector(".region-boundary-note");
const apiMode = document.querySelector("[data-api-mode]");
const forecastModeSelect = document.querySelector("[data-forecast-mode]");
const observationTime = document.querySelector("[data-observation-time]");
const API_BASE = String(window.STORM_SIGNAL_API_URL || "").replace(/\/$/, "");
const globalSymh = document.querySelector("[data-global-symh]");
const globalCurrent = document.querySelector("[data-global-current]");
const globalActual = document.querySelector("[data-global-actual]");
const globalTarget = document.querySelector("[data-global-target]");
const globalHorizon = document.querySelector("[data-global-horizon]");
const globalNote = document.querySelector("[data-global-note]");
const globalModelSelect = document.querySelector("[data-global-model]");
const replayIndexInput = document.querySelector("[data-replay-index]");
const sampleIndexOutput = document.querySelector("[data-sample-index]");
const replayPlayButton = document.querySelector("[data-replay-play]");
const forecastViewButtons = [...document.querySelectorAll("[data-forecast-view]")];
const forecastPanels = [...document.querySelectorAll("[data-forecast-panel]")];
const regionalScoreCanvas = document.querySelector(".regional-score-grid");
const regionalScoreContext = regionalScoreCanvas.getContext("2d");
const selectedModelScore = document.querySelector("[data-model-score]");
const selectedModelScoreMax = document.querySelector("[data-model-score-max]");
const modelScoreNote = document.querySelector("[data-model-score-note]");
const tecOverlayControl = document.querySelector("[data-tec-overlay]");
const tecMappingStatus = document.querySelector("[data-tec-mapping-status]");
const gridScoreRange = document.querySelector("[data-grid-score-range]");
const gridScoreNote = document.querySelector("[data-grid-score-note]");
const forecastNote = document.querySelector("[data-forecast-note]");
const countryConnector = document.querySelector(".country-connector");
const countryConnectorLine = document.querySelector(".country-connector-line");
const countryConnectorHalo = document.querySelector(".country-connector-halo");
const countryConnectorNode = document.querySelector(".country-connector-node");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let apiSnapshotPromise = null;
let apiSnapshotKey = "";
let apiRequestId = 0;
let apiDebounceTimer = null;
let hasUserSelectedSample = false;
let replayPlaying = false;
let replayTimer = null;
let liveRefreshTimer = null;
const TEC_GRID_ROWS = 72;
const TEC_GRID_COLUMNS = 72;
const tecGrid = {
  rows: TEC_GRID_ROWS,
  columns: TEC_GRID_COLUMNS,
  values: new Float32Array(TEC_GRID_ROWS * TEC_GRID_COLUMNS).fill(Number.NaN),
  units: "TEC Disturbance Model Score",
  mapping: null,
  mappingVerified: false
};
const countryFeatures = [];
const countryRaycaster = new THREE.Raycaster();
const countryPointer = new THREE.Vector2();
const tecOverlay = new THREE.Group();
tecOverlay.name = "TEC heatmap overlay";
tecOverlay.visible = false;
let countryBoundaryLines = null;
let selectedCountryOutline = null;
let selectedCountryAnchor = null;
let selectedCountryFeature = null;
let tecOverlayRequested = false;
let tecOverlayReady = false;
let lastCountryTap = null;
let detailedBoundaryPromise = null;
let detailedCountryFeatures = null;
let admin1BoundaryFeatures = null;
let detailRequestId = 0;
const detailBoundaryGroup = new THREE.Group();
detailBoundaryGroup.name = "Detailed country and admin-1 boundaries";
detailBoundaryGroup.visible = false;

function setEarthTarget(centerX, centerY, targetSize) {
  const layoutCenterX = stage.offsetLeft + stage.offsetWidth / 2;
  const layoutCenterY = stage.offsetTop + stage.offsetHeight / 2;
  stage.style.setProperty("--earth-shift-x", `${centerX - layoutCenterX}px`);
  stage.style.setProperty("--earth-shift-y", `${centerY - layoutCenterY}px`);
  stage.style.setProperty("--earth-focus-scale", String(targetSize / stage.offsetWidth));
}

function setActiveNavigationView(activeView) {
  for (const link of [...textNavigation.querySelectorAll("a"), ...starNavigation.querySelectorAll("a")]) {
    const isCurrent = link.dataset.view === activeView;
    link.classList.toggle("is-active", isCurrent);
    if (isCurrent) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

function setAlertNavigationEnabled(enabled) {
  for (const navigation of [textNavigation, starNavigation]) {
    const alertLink = navigation.querySelector('a[href="#alert"]');
    if (!alertLink) continue;
    alertLink.dataset.view = "alert";
    alertLink.removeAttribute("aria-disabled");
  }
}

function setCenteredEarthTarget() {
  const minViewport = Math.min(window.innerWidth, window.innerHeight);
  const isMobile = window.matchMedia("(max-width: 700px)").matches;
  const isTablet = window.matchMedia("(min-width: 701px) and (max-width: 900px)").matches;
  const targetSize = isMobile
    ? Math.min(minViewport * 0.94, 780)
    : isTablet
      ? Math.min(stage.offsetWidth * 0.94, minViewport * 0.94)
      : Math.min(minViewport * 0.82, 980);
  setEarthTarget(window.innerWidth / 2, window.innerHeight * (isMobile ? 0.54 : 0.53), targetSize);
}

function positionEarthBesidePopup(clientX, clientY, bounds, anchorOverride = null) {
  const isMobile = window.matchMedia("(max-width: 700px)").matches;
  const selectedOnLeft = anchorOverride
    ? anchorOverride.panelSide === "left"
    : clientX < bounds.left + bounds.width / 2;
  const panelSide = selectedOnLeft ? "left" : "right";
  const targetSize = isMobile
    ? window.innerWidth * 2.15
    : Math.min(window.innerWidth, window.innerHeight) * 1.5;

  countryPanel.classList.toggle("is-left", panelSide === "left");
  selectedCountryAnchor = anchorOverride
    ? {...anchorOverride, panelSide}
    : {
        u: THREE.MathUtils.clamp((clientX - bounds.left) / bounds.width, 0, 1),
        v: THREE.MathUtils.clamp((clientY - bounds.top) / bounds.height, 0, 1),
        panelSide
      };
  const focusX = window.innerWidth * (isMobile ? 0.5 : panelSide === "left" ? 0.66 : 0.34);
  const focusY = window.innerHeight * (isMobile ? 0.37 : 0.53);
  const centerX = focusX - (selectedCountryAnchor.u - 0.5) * targetSize;
  const centerY = focusY - (selectedCountryAnchor.v - 0.5) * targetSize;
  setEarthTarget(centerX, centerY, targetSize);
  countryConnector.classList.add("is-visible");
  updateCountryConnector();
}

function updateCountryConnector() {
  if (!selectedCountryAnchor || !countryPanel.classList.contains("is-open")) return;
  const panelBounds = countryPanel.getBoundingClientRect();
  const isMobile = window.matchMedia("(max-width: 700px)").matches;
  let pointX;
  let pointY;
  if (selectedCountryAnchor.localPoint) {
    globe.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    const worldPoint = globe.localToWorld(selectedCountryAnchor.localPoint.clone().multiplyScalar(2.735));
    const projectedPoint = worldPoint.project(camera);
    const earthBounds = stage.getBoundingClientRect();
    pointX = earthBounds.left + ((projectedPoint.x + 1) / 2) * earthBounds.width;
    pointY = earthBounds.top + ((1 - projectedPoint.y) / 2) * earthBounds.height;
  } else {
    const earthBounds = stage.getBoundingClientRect();
    pointX = earthBounds.left + selectedCountryAnchor.u * earthBounds.width;
    pointY = earthBounds.top + selectedCountryAnchor.v * earthBounds.height;
  }
  const panelX = isMobile
    ? panelBounds.left + panelBounds.width / 2
    : selectedCountryAnchor.panelSide === "left" ? panelBounds.right : panelBounds.left;
  const panelY = isMobile ? panelBounds.top : panelBounds.top + 52;
  const controlX = (pointX + panelX) / 2;
  countryConnector.setAttribute("viewBox", `0 0 ${window.innerWidth} ${window.innerHeight}`);
  countryConnectorLine.setAttribute("d", `M ${pointX} ${pointY} C ${controlX} ${pointY}, ${controlX} ${panelY}, ${panelX} ${panelY}`);
  countryConnectorHalo.setAttribute("cx", String(pointX));
  countryConnectorHalo.setAttribute("cy", String(pointY));
  countryConnectorNode.setAttribute("cx", String(pointX));
  countryConnectorNode.setAttribute("cy", String(pointY));
}

function setEarthView(isExploring) {
  if (hero.classList.contains("is-exploring") === isExploring) return;
  if (isExploring) {
    setCenteredEarthTarget();
    hero.classList.add("is-exploring");
  } else {
    hero.classList.remove("is-exploring");
    stage.style.removeProperty("--earth-shift-x");
    stage.style.removeProperty("--earth-shift-y");
    stage.style.removeProperty("--earth-focus-scale");
    closeCountryPanel();
  }
  if (countryBoundaryLines) countryBoundaryLines.visible = isExploring;
  countryRiskBoundaries.visible = isExploring;
  tecOverlay.visible = isExploring && tecOverlayReady && tecOverlayRequested;
  if (isExploring) {
    stage.focus({preventScroll: true});
  } else {
    textNavigation.inert = false;
    textNavigation.setAttribute("aria-hidden", "false");
    textNavigation.querySelector('[data-view="home"]').focus({preventScroll: true});
  }
  stage.setAttribute("aria-pressed", String(isExploring));
  stage.setAttribute("aria-label", isExploring ? "Tap Earth to return to the forecast page" : "Tap Earth to open immersive view");
  intro.inert = isExploring;
  intro.setAttribute("aria-hidden", String(isExploring));
  startButton.inert = isExploring;
  countryPanel.inert = !isExploring || !countryPanel.classList.contains("is-open");
  countryPanel.setAttribute("aria-hidden", String(countryPanel.inert));
  textNavigation.inert = isExploring;
  textNavigation.setAttribute("aria-hidden", String(isExploring));
  starNavigation.inert = !isExploring;
  starNavigation.setAttribute("aria-hidden", String(!isExploring));
  status.textContent = isExploring ? "Immersive Earth view opened." : "Returned to the forecast page.";

  setAlertNavigationEnabled(false);
  setActiveNavigationView(isExploring ? "forecast" : "home");
}

function handleNavigation(event) {
  const link = event.target.closest("a[data-view]");
  if (!link) return;
  event.preventDefault();
  if (link.dataset.view === "home") {
    closeAlertPage();
    setEarthView(false);
    closeAboutPage();
  } else if (link.dataset.view === "alert") {
    openAlertPage();
    closeAboutPage();
  }
  else if (link.dataset.view === "forecast") {
    closeAlertPage();
    if (hero.classList.contains("is-region-detail")) closeCountryPanel();
    setEarthView(true);
    closeAboutPage();
  }
  else if (link.dataset.view === "about") {
    openAboutPage();
  }
}

textNavigation.addEventListener("click", handleNavigation);
starNavigation.addEventListener("click", handleNavigation);

const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
const deviceMemory = navigator.deviceMemory || 4;
const supports8KTextures = renderer.capabilities.maxTextureSize >= 8192 && deviceMemory >= 8;
const maxPixelRatio = supports8KTextures && deviceMemory >= 16 ? 1.5 : 1;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatio));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.z = 8.8;
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.045;
controls.enablePan = false;
controls.enableZoom = true;
controls.minDistance = 6.8;
controls.maxDistance = 11;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.65;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.7;
controls.update();

const globe = new THREE.Group();
scene.add(globe);
globe.add(tecOverlay);
globe.add(detailBoundaryGroup);

const earthGeometry = new THREE.SphereGeometry(2.7, 64, 64);
const fallbackMaterial = new THREE.MeshPhongMaterial({
  color: 0x0b1930,
  emissive: 0x000000,
  shininess: 28,
  specular: 0x4c8dd4
});
const earth = new THREE.Mesh(earthGeometry, fallbackMaterial);
globe.add(earth);

const atmosphere = new THREE.Mesh(
  new THREE.SphereGeometry(2.77, 48, 48),
  new THREE.MeshBasicMaterial({
    color: 0x48b4ff,
    side: THREE.BackSide,
    transparent: true,
    opacity: 0.3
  })
);
globe.add(atmosphere);

scene.add(new THREE.AmbientLight(0x829dcc, 0.22));
const keyLight = new THREE.DirectionalLight(0x9ecfff, 1.75);
keyLight.position.set(4, 3.8, 5);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x167cff, 0.8);
rimLight.position.set(4, -1.2, -4);
scene.add(rimLight);

const textureLoader = new THREE.TextureLoader();
let cloudLayer;
let lowResolutionReady = false;
let highResolutionUpgradeStarted = false;
const loadTexture = (url) => new Promise((resolve, reject) => {
  textureLoader.load(url, resolve, undefined, reject);
});

Promise.all([
  loadTexture("assets/2k_earth_daymap.jpg"),
  loadTexture("assets/2k_earth_nightmap.jpg"),
  loadTexture("assets/2k_earth_clouds.jpg")
]).then(([surfaceTexture, cityLightsTexture, cloudsTexture]) => {
  surfaceTexture.colorSpace = THREE.SRGBColorSpace;
  cityLightsTexture.colorSpace = THREE.SRGBColorSpace;
  cloudsTexture.colorSpace = THREE.SRGBColorSpace;
  const anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 4);
  for (const texture of [surfaceTexture, cityLightsTexture, cloudsTexture]) {
    texture.anisotropy = anisotropy;
  }
  earth.material = new THREE.MeshPhongMaterial({
    map: surfaceTexture,
    color: 0x587ba3,
    emissive: 0xffd39a,
    emissiveMap: cityLightsTexture,
    emissiveIntensity: 1.05,
    shininess: 28,
    specular: 0x5b8abd
  });
  cloudLayer = new THREE.Mesh(
    new THREE.SphereGeometry(2.725, 48, 48),
    new THREE.MeshPhongMaterial({
      map: cloudsTexture,
      color: 0xffffff,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    })
  );
  globe.add(cloudLayer);
  stage.querySelector(".earth-fallback")?.remove();
  lowResolutionReady = true;
  upgradeEarthTextures();
}).catch((error) => console.warn("Earth textures could not be loaded.", error));

function upgradeEarthTextures() {
  if (!supports8KTextures || !lowResolutionReady || highResolutionUpgradeStarted) return;
  highResolutionUpgradeStarted = true;

  Promise.all([
    loadTexture("assets/8k_earth_daymap.jpg"),
    loadTexture("assets/8k_earth_nightmap.jpg"),
    loadTexture("assets/8k_earth_clouds.jpg"),
    loadTexture("https://threejs.org/examples/textures/planets/earth_normal_2048.jpg"),
    loadTexture("https://threejs.org/examples/textures/planets/earth_specular_2048.jpg")
  ]).then(([surfaceTexture, cityLightsTexture, cloudsTexture, normalTexture, specularTexture]) => {
    surfaceTexture.colorSpace = THREE.SRGBColorSpace;
    cityLightsTexture.colorSpace = THREE.SRGBColorSpace;
    cloudsTexture.colorSpace = THREE.SRGBColorSpace;
    const anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 4);
    for (const texture of [surfaceTexture, cityLightsTexture, cloudsTexture, normalTexture, specularTexture]) {
      texture.anisotropy = anisotropy;
    }
    earth.material.map = surfaceTexture;
    earth.material.emissiveMap = cityLightsTexture;
    earth.material.normalMap = normalTexture;
    earth.material.specularMap = specularTexture;
    earth.material.needsUpdate = true;
    cloudLayer.material.map = cloudsTexture;
    cloudLayer.material.needsUpdate = true;
  }).catch((error) => console.warn("High-resolution Earth textures could not be loaded.", error));
}

function geoToGlobe(longitude, latitude, radius) {
  const phi = (longitude + 180) * Math.PI / 180;
  const theta = (90 - latitude) * Math.PI / 180;
  const sinTheta = Math.sin(theta);
  return [
    -radius * sinTheta * Math.cos(phi),
    radius * Math.cos(theta),
    radius * sinTheta * Math.sin(phi)
  ];
}

function getFeaturePolygons(feature) {
  if (feature.geometry.type === "Polygon") return [feature.geometry.coordinates];
  if (feature.geometry.type === "MultiPolygon") return feature.geometry.coordinates;
  return [];
}

function getFeatureLinePositions(feature, radius) {
  const positions = [];
  for (const polygon of getFeaturePolygons(feature)) {
    for (const ring of polygon) {
      for (let index = 0; index < ring.length - 1; index++) {
        const start = ring[index];
        const end = ring[index + 1];
        if (Math.abs(end[0] - start[0]) > 180) continue;
        positions.push(...geoToGlobe(start[0], start[1], radius));
        positions.push(...geoToGlobe(end[0], end[1], radius));
      }
    }
  }
  return positions;
}

function drawCountryBoundaries(features) {
  const positions = [];
  for (const feature of features) positions.push(...getFeatureLinePositions(feature, 2.718));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: 0x9bdcff,
    transparent: true,
    opacity: 0.62,
    depthWrite: false
  });
  countryBoundaryLines = new THREE.LineSegments(geometry, material);
  countryBoundaryLines.name = "GeoJSON country boundaries";
  countryBoundaryLines.frustumCulled = false;
  countryBoundaryLines.renderOrder = 2;
  countryBoundaryLines.visible = hero.classList.contains("is-exploring");
  globe.add(countryBoundaryLines);
}

function normalizeLongitude(longitude, reference) {
  return longitude + 360 * Math.round((reference - longitude) / 360);
}

function pointInRing(longitude, latitude, ring) {
  let inside = false;
  let previous = ring[ring.length - 1];
  let previousLongitude = previous[0];
  const pointLongitude = normalizeLongitude(longitude, previousLongitude);

  for (const current of ring) {
    const currentLongitude = normalizeLongitude(current[0], previousLongitude);
    const crossesLatitude = (current[1] > latitude) !== (previous[1] > latitude);
    if (crossesLatitude) {
      const crossingLongitude = ((previousLongitude - currentLongitude) * (latitude - current[1])) /
        (previous[1] - current[1]) + currentLongitude;
      if (pointLongitude < crossingLongitude) inside = !inside;
    }
    previous = current;
    previousLongitude = currentLongitude;
  }
  return inside;
}

function featureContainsPoint(feature, longitude, latitude) {
  for (const polygon of getFeaturePolygons(feature)) {
    if (!polygon.length || !pointInRing(longitude, latitude, polygon[0])) continue;
    if (!polygon.slice(1).some((hole) => pointInRing(longitude, latitude, hole))) return true;
  }
  return false;
}

function findCountryAt(clientX, clientY) {
  if (!countryFeatures.length) return null;
  const bounds = canvas.getBoundingClientRect();
  countryPointer.x = ((clientX - bounds.left) / bounds.width) * 2 - 1;
  countryPointer.y = -((clientY - bounds.top) / bounds.height) * 2 + 1;
  countryRaycaster.setFromCamera(countryPointer, camera);
  const hit = countryRaycaster.intersectObject(earth, false)[0];
  if (!hit) return null;

  const localPoint = globe.worldToLocal(hit.point.clone()).normalize();
  const latitude = Math.asin(THREE.MathUtils.clamp(localPoint.y, -1, 1)) * 180 / Math.PI;
  let longitude = Math.atan2(localPoint.z, -localPoint.x) * 180 / Math.PI - 180;
  longitude = ((longitude + 180) % 360 + 360) % 360 - 180;
  const feature = countryFeatures.find((country) => featureContainsPoint(country, longitude, latitude));
  return feature ? {feature, longitude, latitude, localPoint: localPoint.clone()} : null;
}

function closeCountryPanel() {
  if (countryPanel.contains(document.activeElement)) {
    stage.focus({preventScroll: true});
  }
  countryPanel.classList.remove("is-open");
  countryPanel.classList.remove("is-left");
  countryPanel.classList.remove("is-detail");
  countryPanel.inert = true;
  countryPanel.setAttribute("aria-hidden", "true");
  selectedRegionName.textContent = "";
  regionDetail.inert = true;
  regionDetail.setAttribute("aria-hidden", "true");
  hero.classList.remove("is-region-detail");
  setAlertNavigationEnabled(false);
  setActiveNavigationView(hero.classList.contains("is-exploring") ? "forecast" : "home");
  countryConnector.classList.remove("is-visible");
  selectedCountryAnchor = null;
  selectedCountryFeature = null;
  lastCountryTap = null;
  detailRequestId++;
  if (hero.classList.contains("is-exploring")) setCenteredEarthTarget();
  detailBoundaryGroup.visible = false;
  for (const boundary of detailBoundaryGroup.children) {
    boundary.geometry.dispose();
    boundary.material.dispose();
  }
  detailBoundaryGroup.clear();
  if (countryBoundaryLines) countryBoundaryLines.visible = hero.classList.contains("is-exploring");
  if (selectedCountryOutline) {
    globe.remove(selectedCountryOutline);
    selectedCountryOutline.geometry.dispose();
    selectedCountryOutline.material.dispose();
    selectedCountryOutline = null;
  }
}

function showCountry(feature, clientX, clientY, anchorOverride = null) {
  const properties = feature.properties;
  const bounds = stage.getBoundingClientRect();
  closeCountryPanel();
  selectedCountryFeature = feature;
  updateCountryTecSummary();
  positionEarthBesidePopup(clientX, clientY, bounds, anchorOverride);
  selectedRegionName.textContent = properties.ADMIN || properties.NAME || "Unknown country";
  countryPanel.classList.add("is-open");
  countryPanel.inert = false;
  countryPanel.setAttribute("aria-hidden", "false");

  const positions = getFeatureLinePositions(feature, 2.735);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  selectedCountryOutline = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: 0xd5f2ff, transparent: true, opacity: 0.98, depthWrite: false })
  );
  selectedCountryOutline.frustumCulled = false;
  selectedCountryOutline.renderOrder = 3;
  globe.add(selectedCountryOutline);
  updateCountryConnector();
  status.textContent = `${properties.ADMIN || properties.NAME || "Country"} selected.`;
}

function getCountryCode(feature) {
  const properties = feature?.properties || {};
  return properties.ADM0_A3 || properties.ISO_A3 || properties.SOV_A3 || "";
}

function isSamePlaceTap(previousTap, hit, time) {
  if (!previousTap || time - previousTap.time > 650) return false;
  if (previousTap.countryCode !== getCountryCode(hit.feature)) return false;
  let longitudeDelta = Math.abs(previousTap.longitude - hit.longitude);
  if (longitudeDelta > 180) longitudeDelta = 360 - longitudeDelta;
  const latitudeDelta = Math.abs(previousTap.latitude - hit.latitude);
  const averageLatitude = (previousTap.latitude + hit.latitude) * 0.5 * Math.PI / 180;
  const distanceKm = Math.hypot(latitudeDelta, longitudeDelta * Math.cos(averageLatitude)) * 111.32;
  return distanceKm <= 250;
}

function getAdmin1LinePositions(feature, radius) {
  if (!feature.geometry) return [];
  const lines = feature.geometry.type === "LineString"
    ? [feature.geometry.coordinates]
    : feature.geometry.type === "MultiLineString"
      ? feature.geometry.coordinates
      : [];
  const positions = [];
  for (const line of lines) {
    for (let index = 0; index < line.length - 1; index++) {
      const start = line[index];
      const end = line[index + 1];
      if (Math.abs(end[0] - start[0]) > 180) continue;
      positions.push(...geoToGlobe(start[0], start[1], radius));
      positions.push(...geoToGlobe(end[0], end[1], radius));
    }
  }
  return positions;
}

async function loadDetailedBoundaries(feature, requestId) {
  if (!detailedBoundaryPromise) {
    detailedBoundaryPromise = Promise.all([
      fetch("assets/countries-50m.geojson").then((response) => {
        if (!response.ok) throw new Error(`50m country boundaries request failed: ${response.status}`);
        return response.json();
      }),
      fetch("assets/admin1-lines-50m.geojson").then((response) => {
        if (!response.ok) throw new Error(`Admin-1 boundaries request failed: ${response.status}`);
        return response.json();
      })
    ]).then(([countries, admin1]) => {
      detailedCountryFeatures = countries.features;
      admin1BoundaryFeatures = admin1.features;
    }).catch((error) => {
      detailedBoundaryPromise = null;
      throw error;
    });
  }

  try {
    await detailedBoundaryPromise;
    if (requestId !== detailRequestId || !hero.classList.contains("is-region-detail")) return;
    const countryCode = getCountryCode(feature);
    const detailedCountry = detailedCountryFeatures.find((candidate) => getCountryCode(candidate) === countryCode);
    const admin1Lines = admin1BoundaryFeatures.filter((candidate) => candidate.properties.ADM0_A3 === countryCode);
    const positions = detailedCountry ? getFeatureLinePositions(detailedCountry, 2.719) : [];
    for (const boundary of admin1Lines) positions.push(...getAdmin1LinePositions(boundary, 2.723));

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color: 0x76d0ff,
      transparent: true,
      opacity: 0.82,
      depthWrite: false
    }));
    lines.name = "1:50m country and admin-1 boundaries";
    lines.frustumCulled = false;
    lines.renderOrder = 4;
    detailBoundaryGroup.clear();
    detailBoundaryGroup.add(lines);
    detailBoundaryGroup.visible = true;
    detailBoundaryNote.textContent = admin1Lines.length
      ? `1:50m detail / ${admin1Lines.length} admin-1 boundaries`
      : "1:50m detail / no finer boundaries available";
  } catch (error) {
    if (requestId === detailRequestId) detailBoundaryNote.textContent = "Detailed boundaries unavailable";
    console.warn("Detailed regional boundaries could not be loaded.", error);
  }
}

function setForecastView(view) {
  for (const button of forecastViewButtons) {
    const active = button.dataset.forecastView === view;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  }
  for (const panel of forecastPanels) {
    const active = panel.dataset.forecastPanel === view;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  }
}

function getRiskColor(score) {
  if (score >= 0.8) return "#EF4444";
  if (score >= 0.6) return "#F97316";
  if (score >= 0.4) return "#FACC15";
  if (score >= 0.2) return "#22D3EE";
  return "#22C55E";
}

function getCountryRiskColor(score) {
  if (!Number.isFinite(score)) return 0x9bdcff;

  if (score >= 0.8) return 0xef4444; // High risk: red
  if (score >= 0.6) return 0xf97316; // Elevated: orange
  if (score >= 0.4) return 0xfacc15; // Moderate: yellow

  return 0x9bdcff; // Low: original blue
}

const countryRiskBoundaries = new THREE.Group();
countryRiskBoundaries.name = "Country risk warning borders";
globe.add(countryRiskBoundaries);

function updateCountryRiskBorders() {
  for (const child of countryRiskBoundaries.children) {
    child.geometry.dispose();
    child.material.dispose();
  }
  countryRiskBoundaries.clear();

  if (!tecOverlayReady || !tecGrid.mapping) return;

  for (const feature of countryFeatures) {
    const summary = getCountryTecSummary(feature);

    if (!summary || summary.maximum < 0.6) continue;

    const positions = getFeatureLinePositions(feature, 2.742);
    if (!positions.length) continue;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3)
    );

    const material = new THREE.LineBasicMaterial({
      color: getCountryRiskColor(summary.maximum),
      transparent: true,
      opacity: 0.95,
      depthWrite: false
    });

    const outline = new THREE.LineSegments(geometry, material);
    outline.renderOrder = 5;
    outline.frustumCulled = false;

    countryRiskBoundaries.add(outline);
  }

  countryRiskBoundaries.visible =
    hero.classList.contains("is-exploring");
}

function drawRegionalScoreGrid(scores) {
  const rows = scores.length;
  const columns = rows ? scores[0].length : 0;
  regionalScoreContext.clearRect(0, 0, regionalScoreCanvas.width, regionalScoreCanvas.height);
  if (!rows || !columns) return;
  const values = scores.flat().filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!values.length) return;
  const cellWidth = regionalScoreCanvas.width / columns;
  const cellHeight = regionalScoreCanvas.height / rows;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const value = scores[row][column];
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      regionalScoreContext.fillStyle = getRiskColor(value);
      regionalScoreContext.fillRect(column * cellWidth, row * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
    }
  }
}

function loadApiSnapshot() {
  const model = globalModelSelect.value;
  const isLive = forecastModeSelect.value === "live";
  const index = Number(replayIndexInput.value) || 0;
  const requestKey = `${isLive ? "live" : "replay"}:${model}:${isLive ? Math.floor(Date.now() / 5000) : index}`;
  if (apiSnapshotPromise && apiSnapshotKey === requestKey) return apiSnapshotPromise;
  apiSnapshotKey = requestKey;
  const requestId = ++apiRequestId;
  apiMode.textContent = isLive ? "Updating Live Forecast…" : "Loading Historical Replay…";

  const getJson = async (path) => {
    const response = await fetch(`${API_BASE}/api/v1/${path}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`API request failed: ${response.status}`);
    return response.json();
  };

  const framePath = isLive
    ? `live/latest?model=${encodeURIComponent(model)}`
    : `replay/current?index=${index}&model=${encodeURIComponent(model)}`;

  apiSnapshotPromise = Promise.all([getJson("status"), getJson("grid/metadata"), getJson(framePath)])
    .then(([apiStatus, gridMetadata, frame]) => {
      if (requestId !== apiRequestId) return null;
      const globalForecast = frame.global_forecast;
      const regionalForecast = frame.regional_forecast;
      if (!globalForecast || !regionalForecast) throw new Error("Missing forecast data in API frame");
      const sampleCount = Number(frame.sample_count) || Number(regionalForecast.sample_count) || 1640;
      const activeIndex = Number(frame.sample_index) || 0;
      replayIndexInput.max = String(sampleCount - 1);
      replayIndexInput.value = String(activeIndex);
      sampleIndexOutput.textContent = String(activeIndex + 1);
      apiMode.textContent = `${isLive ? "Live Forecast" : "Historical Replay"} / ${globalForecast.model || model}`;
      observationTime.textContent = `Observation UTC: ${frame.forecast_time_utc || globalForecast.forecast_time_utc || "--"} · Historical input`;

      const mappingVerified = apiStatus.geographic_mapping_status === "verified" &&
        gridMetadata.mapping_status === "verified" && gridMetadata.coordinates_available === true;
      const gridShapeValid = gridMetadata.rows === TEC_GRID_ROWS &&
        gridMetadata.columns === TEC_GRID_COLUMNS &&
        Array.isArray(regionalForecast.scores) && regionalForecast.scores.length === TEC_GRID_ROWS &&
        regionalForecast.scores.every(row => Array.isArray(row) && row.length === TEC_GRID_COLUMNS);
      const formatSymH = (value) => {
        if (value === null || value === undefined) return "--";
        const number = Number(value);
        return Number.isFinite(number) ? `${number > 0 ? "+" : ""}${number.toFixed(2)} nT` : "--";
      };
      globalSymh.textContent = formatSymH(globalForecast.predicted_sym_h_nT);
      globalCurrent.textContent = formatSymH(globalForecast.current_sym_h_nT);
      globalActual.textContent = formatSymH(globalForecast.actual_future_sym_h_nT);
      globalTarget.textContent = globalForecast.target_time_utc || "--";
      globalHorizon.textContent = Number.isFinite(Number(globalForecast.horizon_minutes))
        ? `${globalForecast.horizon_minutes} min` : "--";
      globalNote.textContent = `${isLive ? "Auto-updating forecast" : "Historical replay"} / sample ${activeIndex + 1} of ${sampleCount} / historical observations`;
      const values = gridShapeValid ? regionalForecast.scores.flat().filter(value => typeof value === "number" && Number.isFinite(value)) : [];
      gridScoreRange.textContent = values.length ? `${Math.min(...values).toFixed(3)} – ${Math.max(...values).toFixed(3)}` : "--";
      gridScoreNote.textContent = gridShapeValid
        ? `TEC Disturbance Model Score / ${regionalForecast.probability_status || "uncalibrated"} / raw index grid / not a probability`
        : "72 x 72 grid metadata unavailable";
      if (gridShapeValid) {
        drawRegionalScoreGrid(regionalForecast.scores);
        setTecGrid({ rows: gridMetadata.rows, columns: gridMetadata.columns,
          values: regionalForecast.scores, mapping: tecGrid.mapping,
          units: regionalForecast.display_label || "TEC Disturbance Model Score" });
      } else {
        tecOverlayReady = false;
        tecOverlay.visible = false;
        tecOverlayControl.disabled = true;
        tecMappingStatus.textContent = "Invalid regional grid; Earth overlay unavailable.";
        updateCountryTecSummary();
      }
      forecastNote.textContent = `${isLive ? "Live Forecast" : "Historical Replay"} / ${regionalForecast.horizon_minutes || 120} minute target / historical input`;
      return { apiStatus, gridMetadata, globalForecast, regionalForecast, gridShapeValid, mappingVerified };
    }).catch(error => {
      if (requestId !== apiRequestId) return null;
      apiSnapshotPromise = null;
      apiSnapshotKey = "";
      apiMode.textContent = "Forecast API unavailable";
      observationTime.textContent = `Connection failed: ${error.message}`;
      tecOverlayReady = false;
      tecOverlay.visible = false;
      tecOverlayControl.disabled = true;
      tecMappingStatus.textContent = "Regional data unavailable.";
      selectedModelScore.textContent = "--";
      selectedModelScoreMax.textContent = "--";
      selectedRegionSummary.textContent = "TEC Disturbance Model Score unavailable";
      modelScoreNote.textContent = "Forecast data could not be loaded.";
      console.warn("Forecast API could not be loaded.", error);
      return null;
    });
  return apiSnapshotPromise;
}

function updateForecastMode() {
  const isLive = forecastModeSelect.value === "live";
  stopReplay();
  replayIndexInput.disabled = isLive;
  replayPlayButton.disabled = isLive;
  replayIndexInput.closest("label").hidden = isLive;
  replayPlayButton.hidden = isLive;
  apiSnapshotKey = "";
  clearInterval(liveRefreshTimer);
  liveRefreshTimer = null;
  void loadApiSnapshot();
  if (isLive) liveRefreshTimer = setInterval(() => {
    apiSnapshotKey = "";
    void loadApiSnapshot();
  }, 5000);
}

function showRegionDetail(feature, clientX, clientY, anchorOverride = null) {
  showCountry(feature, clientX, clientY, anchorOverride);
  hero.classList.add("is-region-detail");
  setAlertNavigationEnabled(true);
  setActiveNavigationView("alert");
  countryPanel.classList.add("is-detail");
  regionDetail.inert = false;
  regionDetail.setAttribute("aria-hidden", "false");

  const properties = feature.properties;
  detailCountryName.textContent = properties.ADMIN || properties.NAME || "Unknown country";
  detailContinent.textContent = properties.CONTINENT || "--";
  const countryCode = properties.ISO_A3 && properties.ISO_A3 !== "-99" ? properties.ISO_A3 : getCountryCode(feature);
  detailCountryCode.textContent = countryCode || "--";
  const population = Number(properties.POP_EST);
  detailPopulation.textContent = Number.isFinite(population) && population > 0
    ? new Intl.NumberFormat().format(population)
    : "--";
  detailBoundaryNote.textContent = "Loading 1:50m detail boundaries";

  const mobile = window.matchMedia("(max-width: 700px)").matches;
  const tablet = window.matchMedia("(min-width: 701px) and (max-width: 900px)").matches;
  const singleClickSize = mobile
    ? Math.min(window.innerWidth * 0.62, window.innerHeight * 0.38, 300)
    : tablet
      ? Math.min(window.innerHeight * 0.68, window.innerWidth * 0.46, 620)
      : Math.min(window.innerHeight * 0.68, window.innerWidth * 0.44, 720);
  const targetSize = Math.min(singleClickSize * 1.16, mobile ? 340 : tablet ? 680 : 820);
  const panelSide = selectedCountryAnchor?.panelSide === "left" ? "right" : "left";
  if (selectedCountryAnchor) selectedCountryAnchor.panelSide = panelSide;
  countryPanel.classList.toggle("is-left", panelSide === "left");
  const margin = targetSize / 2 + 20;
  const desiredCenterX = window.innerWidth * (panelSide === "left" ? 0.66 : 0.34);
  const centerX = THREE.MathUtils.clamp(desiredCenterX, margin, window.innerWidth - margin);
  setEarthTarget(centerX, window.innerHeight * (mobile ? 0.42 : 0.53), targetSize);
  if (countryBoundaryLines) countryBoundaryLines.visible = false;

  const requestId = ++detailRequestId;
  void loadDetailedBoundaries(feature, requestId);
  status.textContent = `${detailCountryName.textContent} detailed view.`;
}
// Experimental geographic mapping for visualization only.
// NOT verified IonoBench coordinates.
function createExperimentalTecMapping() {
  const latitudeStep = 180 / TEC_GRID_ROWS;
  const longitudeStep = 360 / TEC_GRID_COLUMNS;

  return {
    verified: false,
    experimental: true,

    cellBounds(row, column) {
      return {
        west: -180 + column * longitudeStep,
        east: -180 + (column + 1) * longitudeStep,
        north: 90 - row * latitudeStep,
        south: 90 - (row + 1) * latitudeStep
      };
    }
  };
}
function setTecGrid({ rows, columns, values, mapping, units = "TEC Disturbance Model Score" }) {
  if (rows !== TEC_GRID_ROWS || columns !== TEC_GRID_COLUMNS) {
    throw new RangeError(`TEC grid must be ${TEC_GRID_ROWS} x ${TEC_GRID_COLUMNS}.`);
  }
  if (Array.isArray(values) && Array.isArray(values[0]) &&
      (values.length !== rows || values.some((row) => !Array.isArray(row) || row.length !== columns))) {
    throw new RangeError("TEC grid must contain exactly 72 rows of 72 values.");
  }
  const flattenedValues = Array.isArray(values) && Array.isArray(values[0]) ? values.flat() : Array.from(values || []);
  if (flattenedValues.length !== TEC_GRID_ROWS * TEC_GRID_COLUMNS) {
    throw new RangeError("TEC grid value count does not match its 72 x 72 dimensions.");
  }

  tecOverlayReady = false;
  tecOverlay.visible = false;
  tecOverlayControl.disabled = true;
  tecOverlayControl.checked = tecOverlayRequested;
  tecGrid.values = Float32Array.from(flattenedValues, (value) => typeof value === "number" && Number.isFinite(value) ? value : Number.NaN);
  tecGrid.mapping =
    mapping && typeof mapping.cellBounds === "function"
      ? mapping
      : createExperimentalTecMapping();

  tecGrid.mappingVerified =
    tecGrid.mapping.verified === true;

  tecGrid.units = units;

  const finiteValues = Array.from(tecGrid.values).filter(Number.isFinite);
  if (!finiteValues.length) {
    disposeTecOverlay();
    tecMappingStatus.textContent = "No valid TEC Disturbance Model Score values.";
    updateCountryTecSummary();
    return false;
  }

  const positions = [];
  const colors = [];
  const indices = [];
  const color = new THREE.Color();

  for (let row = 0; row < TEC_GRID_ROWS; row++) {
    for (let column = 0; column < TEC_GRID_COLUMNS; column++) {
      const value = tecGrid.values[row * TEC_GRID_COLUMNS + column];
      if (!Number.isFinite(value)) continue;
      const cellBounds = tecGrid.mapping.cellBounds(row, column);
      validateTecCellBounds(cellBounds);
      const base = positions.length / 3;
      const corners = [
        [cellBounds.west, cellBounds.south],
        [cellBounds.east, cellBounds.south],
        [cellBounds.east, cellBounds.north],
        [cellBounds.west, cellBounds.north]
      ];
      for (const [longitude, latitude] of corners) {
        positions.push(...geoToGlobe(longitude, latitude, 2.724));
        color.set(getRiskColor(value));
        colors.push(color.r, color.g, color.b);
      }
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }

  disposeTecOverlay();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  const overlayMesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.56,
    side: THREE.DoubleSide,
    depthWrite: false
  }));
  overlayMesh.renderOrder = 2;
  tecOverlay.add(overlayMesh);
  tecOverlayReady = true;
  tecOverlayControl.disabled = false;
  tecOverlayControl.checked = tecOverlayRequested;
  tecMappingStatus.textContent = "Verified mapping.";
  tecOverlay.visible =
    hero.classList.contains("is-exploring") &&
    tecOverlayRequested;

  updateCountryTecSummary();
  updateCountryRiskBorders();
  updateAlertPage();

  return true;
}

function disposeTecOverlay() {
  for (const child of tecOverlay.children) {
    child.geometry.dispose();
    child.material.dispose();
  }
  tecOverlay.clear();
}

function setTecOverlayVisible(visible) {
  tecOverlayRequested = Boolean(visible);
  tecOverlayControl.checked = tecOverlayRequested && tecOverlayReady;
  tecOverlay.visible = hero.classList.contains("is-exploring") && tecOverlayReady && tecOverlayRequested;
  return tecOverlay.visible;
}

function validateTecCellBounds(bounds) {
  if (!bounds || ![bounds.west, bounds.south, bounds.east, bounds.north].every(Number.isFinite) ||
      bounds.west < -180 || bounds.west > 180 || bounds.east < -180 || bounds.east > 180 ||
      bounds.south < -90 || bounds.north > 90 || bounds.south >= bounds.north || bounds.west === bounds.east) {
    throw new TypeError("cellBounds must provide valid geographic west, south, east and north edges.");
  }
}

function getCountryTecSummary(feature) {
  if (
  !feature ||
  !tecOverlayReady ||
  !tecGrid.mapping ||
  typeof tecGrid.mapping.cellBounds !== "function"
) {
  return null;
}
  let sum = 0;
  let maximum = -Infinity;
  let count = 0;
  for (let row = 0; row < TEC_GRID_ROWS; row++) {
    for (let column = 0; column < TEC_GRID_COLUMNS; column++) {
      const score = tecGrid.values[row * TEC_GRID_COLUMNS + column];
      if (!Number.isFinite(score)) continue;
      const bounds = tecGrid.mapping.cellBounds(row, column);
      validateTecCellBounds(bounds);
      const east = bounds.east < bounds.west ? bounds.east + 360 : bounds.east;
      const longitude = ((bounds.west + (east - bounds.west) / 2 + 180) % 360 + 360) % 360 - 180;
      const latitude = (bounds.south + bounds.north) / 2;
      if (!featureContainsPoint(feature, longitude, latitude)) continue;
      sum += score;
      maximum = Math.max(maximum, score);
      count++;
    }
  }
  return count ? {mean: sum / count, maximum, count} : null;
}

function updateCountryTecSummary() {
  const summary = getCountryTecSummary(selectedCountryFeature);

  selectedModelScore.textContent = summary
    ? summary.mean.toFixed(3)
    : "--";

  selectedModelScoreMax.textContent = summary
    ? summary.maximum.toFixed(3)
    : "--";

  selectedRegionSummary.textContent = summary
    ? `Experimental TEC Model Score: mean ${summary.mean.toFixed(3)}, max ${summary.maximum.toFixed(3)}`
    : "TEC Disturbance Model Score unavailable";

  modelScoreNote.textContent = summary
    ? `${summary.count} grid cell centers assigned to this country using an experimental, unverified coordinate mapping. These are uncalibrated model scores, not probabilities or verified geographic measurements.`
    : !tecOverlayReady
      ? "Regional TEC grid is not ready."
      : "No grid cell centers assigned to this country under the experimental mapping.";
}

function setTecMapping(mapping) {
  if (mapping !== null && (mapping?.verified !== true || typeof mapping.cellBounds !== "function")) {
    throw new TypeError("TEC mapping must be explicitly verified and provide cellBounds(row, column), or be null.");
  }
  return setTecGrid({ rows: TEC_GRID_ROWS, columns: TEC_GRID_COLUMNS, values: tecGrid.values, mapping, units: tecGrid.units });
}

tecOverlayControl.addEventListener("change", () => setTecOverlayVisible(tecOverlayControl.checked));

window.StormSignalData = Object.freeze({
  tecGridShape: Object.freeze({ rows: TEC_GRID_ROWS, columns: TEC_GRID_COLUMNS }),
  getRiskColor,
  setTecGrid,
  setTecMapping,
  getCountryTecSummary: (countryCode) => getCountryTecSummary(countryFeatures.find((feature) => getCountryCode(feature) === countryCode)),
  setTecOverlayVisible
});

fetch("assets/countries-110m.geojson")
  .then((response) => {
    if (!response.ok) throw new Error(`Country GeoJSON request failed: ${response.status}`);
    return response.json();
  })
  .then((geojson) => {
    countryFeatures.push(...geojson.features);
    updateAlertPage();
    drawCountryBoundaries(countryFeatures);
    updateCountryRiskBorders();
    stage.dataset.countryCount = String(countryFeatures.length);
  })
  .catch((error) => console.warn("Country boundaries could not be loaded.", error));

countryPanelClose.addEventListener("click", closeCountryPanel);
hero.addEventListener("pointerdown", (event) => {
  if (!countryPanel.classList.contains("is-open")) return;
  const target = event.target;
  const isInteractiveTarget = countryPanel.contains(target) ||
    canvas.contains(target) ||
    textNavigation.contains(target) ||
    starNavigation.contains(target) ||
    startButton.contains(target);
  if (isInteractiveTarget) return;
  const previousTap = lastCountryTap;
  const time = performance.now();
  const sameScreenPlace = previousTap &&
    time - previousTap.time <= 650 &&
    Math.hypot(event.clientX - previousTap.x, event.clientY - previousTap.y) <= 36;
  if (sameScreenPlace) {
    lastCountryTap = null;
    showRegionDetail(previousTap.feature, event.clientX, event.clientY, previousTap.anchor, previousTap.localPoint);
    return;
  }
  closeCountryPanel();
});

forecastViewButtons.forEach((button) => {
  button.addEventListener("click", () => setForecastView(button.dataset.forecastView));
});

forecastModeSelect.addEventListener("change", updateForecastMode);

globalModelSelect.addEventListener("change", () => {
  stopReplay();
  apiSnapshotKey = "";
  void loadApiSnapshot();
});

replayIndexInput.addEventListener("input", () => {
  stopReplay();
  hasUserSelectedSample = true;
  sampleIndexOutput.textContent = String(Number(replayIndexInput.value) + 1);
  clearTimeout(apiDebounceTimer);
  apiDebounceTimer = setTimeout(() => {
    apiSnapshotKey = "";
    void loadApiSnapshot();
  }, 180);
});

replayIndexInput.addEventListener("change", () => {
  stopReplay();
  hasUserSelectedSample = true;
  clearTimeout(apiDebounceTimer);
  apiSnapshotKey = "";
  void loadApiSnapshot();
});

function stopReplay() {
  replayPlaying = false;
  clearTimeout(replayTimer);
  replayTimer = null;
  replayPlayButton.textContent = "Play replay";
  replayPlayButton.setAttribute("aria-pressed", "false");
}

async function advanceReplay() {
  if (!replayPlaying) return;
  const maximum = Number(replayIndexInput.max);
  const current = Number(replayIndexInput.value);
  const next = current >= maximum ? 0 : current + 1;
  replayIndexInput.value = String(next);
  sampleIndexOutput.textContent = String(next + 1);
  const snapshot = await loadApiSnapshot();
  if (!replayPlaying) return;
  if (!snapshot || next >= Number(replayIndexInput.max)) {
    stopReplay();
    return;
  }
  replayTimer = setTimeout(advanceReplay, 1500);
}

replayPlayButton.addEventListener("click", () => {
  if (replayPlaying) {
    stopReplay();
    return;
  }
  hasUserSelectedSample = true;
  replayPlaying = true;
  replayPlayButton.textContent = "Pause replay";
  replayPlayButton.setAttribute("aria-pressed", "true");
  void advanceReplay();
});

globe.rotation.y = -0.4;
globe.rotation.z = -0.1;

let earthPointerStart = null;
canvas.addEventListener("pointerdown", (event) => {
  stage.classList.add("is-dragging");
  if (!event.isPrimary || (event.pointerType === "mouse" && event.button !== 0)) {
    earthPointerStart = null;
    return;
  }
  earthPointerStart = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
});

canvas.addEventListener("pointermove", (event) => {
  if (!earthPointerStart || event.pointerId !== earthPointerStart.id) return;
  if (Math.hypot(event.clientX - earthPointerStart.x, event.clientY - earthPointerStart.y) > 8) {
    earthPointerStart.moved = true;
  }
});

window.addEventListener("pointerup", (event) => {
  stage.classList.remove("is-dragging");
  if (earthPointerStart && event.pointerId === earthPointerStart.id && !earthPointerStart.moved) {
    if (hero.classList.contains("is-exploring")) {
      const time = performance.now();
      const previousTap = lastCountryTap;
      const sameScreenPlace = previousTap &&
        time - previousTap.time <= 650 &&
        Math.hypot(event.clientX - previousTap.x, event.clientY - previousTap.y) <= 36;
      const hit = findCountryAt(event.clientX, event.clientY);
      if (sameScreenPlace) {
        lastCountryTap = null;
        showRegionDetail(previousTap.feature, event.clientX, event.clientY, previousTap.anchor);
      } else if (!hit) {
        lastCountryTap = null;
        closeCountryPanel();
      } else if (isSamePlaceTap(previousTap, hit, time)) {
        const anchor = previousTap.anchor;
        lastCountryTap = null;
        showRegionDetail(hit.feature, event.clientX, event.clientY, anchor);
      } else {
        const bounds = stage.getBoundingClientRect();
        const anchor = {
          u: THREE.MathUtils.clamp((event.clientX - bounds.left) / bounds.width, 0, 1),
          v: THREE.MathUtils.clamp((event.clientY - bounds.top) / bounds.height, 0, 1),
          panelSide: event.clientX < bounds.left + bounds.width / 2 ? "left" : "right",
          localPoint: hit.localPoint.clone()
        };
        showCountry(hit.feature, event.clientX, event.clientY, anchor);
        lastCountryTap = {
          feature: hit.feature,
          countryCode: getCountryCode(hit.feature),
          longitude: hit.longitude,
          latitude: hit.latitude,
          localPoint: hit.localPoint.clone(),
          x: event.clientX,
          y: event.clientY,
          anchor,
          time
        };
      }
    } else {
      setEarthView(true);
    }
  }
  earthPointerStart = null;
});

window.addEventListener("pointercancel", () => {
  stage.classList.remove("is-dragging");
  earthPointerStart = null;
});

stage.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    setEarthView(!hero.classList.contains("is-exploring"));
  }
});

function resizeScene() {
  const bounds = stage.getBoundingClientRect();
  const width = Math.max(bounds.width, 1);
  const height = Math.max(bounds.height, 1);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  updateCountryConnector();
}

const resizeObserver = new ResizeObserver(resizeScene);
resizeObserver.observe(stage);
resizeScene();

const renderInterval = 1000 / 60;
let previousTime = 0;
let lastRenderTime = 0;
function render(time) {
  if (document.hidden) return;
  if (time - lastRenderTime < renderInterval) {
    requestAnimationFrame(render);
    return;
  }
  const delta = Math.min((time - previousTime) / 1000 || 0, 0.05);
  previousTime = time;
  lastRenderTime = time;
  const motionEnabled = !hero.classList.contains("is-paused") && !reducedMotion.matches && !selectedCountryOutline;
  controls.autoRotate = motionEnabled;
  if (motionEnabled) {
    if (cloudLayer) cloudLayer.rotation.y += delta * 0.005;
  }
  controls.update();
  updateCountryConnector();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
updateForecastMode();

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    previousTime = 0;
    lastRenderTime = 0;
    requestAnimationFrame(render);
  }
});

startButton.addEventListener("click", () => {
  setEarthView(true);
});

const alertPage = document.querySelector(".alert-page");
const alertGrid = document.querySelector("[data-alert-grid]");

const ALERT_THRESHOLD = 0.6;
const CRITICAL_THRESHOLD = 0.8;

function updateAlertPage() {
  if (!tecOverlayReady || !tecGrid.mapping || !countryFeatures.length) {
    alertGrid.textContent = "Waiting for regional forecast data...";
    return;
  }

  const alerts = [];

  for (const feature of countryFeatures) {
    const summary = getCountryTecSummary(feature);

    if (!summary || summary.maximum < ALERT_THRESHOLD) continue;

    const name = feature.properties.ADMIN ||
      feature.properties.NAME || "Unknown";

    alerts.push({
      name,
      code: getCountryCode(feature),
      mean: summary.mean,
      max: summary.maximum,
      count: summary.count,
      level: summary.maximum >= CRITICAL_THRESHOLD
        ? "CRITICAL" : "ELEVATED"
    });
  }

  alerts.sort((a, b) => b.max - a.max);

  document.querySelector("[data-alert-count]").textContent =
    alerts.length;

  document.querySelector("[data-alert-critical]").textContent =
    alerts.filter(a => a.level === "CRITICAL").length;

  document.querySelector("[data-alert-highest]").textContent =
    alerts.length ? alerts[0].max.toFixed(3) : "--";

  alertGrid.replaceChildren();

  if (!alerts.length) {
    const empty = document.createElement("p");
    empty.textContent = "No countries exceed the selected alert threshold.";
    alertGrid.appendChild(empty);
    return;
  }

  for (const alert of alerts) {
    const tile = document.createElement("article");
    tile.className = "alert-tile";

    const color = alert.level === "CRITICAL"
      ? "#ef4444" : "#f97316";

    tile.style.setProperty("--risk-color", color);
    tile.style.setProperty(
      "--score-width",
      `${Math.min(100, Math.max(0, alert.max * 100))}%`
    );

    // Text fields use textContent to avoid injecting country names as HTML.
    tile.innerHTML = `
      <div class="alert-tile-top">
        <div>
          <h3 class="alert-country-name"></h3>
          <p class="alert-country-code"></p>
        </div>
        <span class="alert-level"></span>
      </div>
      <div class="alert-score">
        <span class="alert-max"></span>
        <small>Max Model Score</small>
      </div>
      <div class="alert-bar">
        <div class="alert-bar-fill"></div>
      </div>
      <div class="alert-tile-footer">
        <span class="alert-mean"></span>
        <span class="alert-cells"></span>
      </div>
    `;

    tile.querySelector(".alert-country-name").textContent = alert.name;
    tile.querySelector(".alert-country-code").textContent = alert.code;
    tile.querySelector(".alert-level").textContent = alert.level;
    tile.querySelector(".alert-max").textContent = alert.max.toFixed(3);
    tile.querySelector(".alert-mean").textContent =
      `Mean: ${alert.mean.toFixed(3)}`;
    tile.querySelector(".alert-cells").textContent =
      `${alert.count} grid cells`;

    alertGrid.appendChild(tile);
  }
}

function openAlertPage() {
  updateAlertPage();
  alertPage.hidden = false;
  setActiveNavigationView("alert");
}

function closeAlertPage() {
  alertPage.hidden = true;
  setActiveNavigationView(
    hero.classList.contains("is-exploring") ? "forecast" : "home"
  );
}

document.querySelector("[data-alert-back]")
  .addEventListener("click", closeAlertPage);

const aboutPage = document.querySelector("#about-page");

function openAboutPage() {
  aboutPage.hidden = false;

  const alertPage = document.querySelector("#alert-page");
  if (alertPage) alertPage.hidden = true;

  setActiveNavigationView("about");
}

function closeAboutPage() {
  aboutPage.hidden = true;
}