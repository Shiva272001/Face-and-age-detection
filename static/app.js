/**
 * OmniVision AI — Frontend Application Controller
 * Single Frame Camera with Multi-Camera (Cam 0 / Cam 1) Switching,
 * Real-Time YOLOv8 Object Detection, OpenCV Face Recognition & Gemini Assistant
 */

(() => {
  'use strict';

  // ─── Application State ────────────────────────────────────────────────────────
  const state = {
    apiKey: localStorage.getItem('omnivision_gemini_key') || '',
    modelName: localStorage.getItem('omnivision_gemini_model') || 'gemini-2.5-flash',
    source: 'webcam', // 'webcam' | 'upload'
    currentCameraIndex: 0,
    webcamStream: null,
    videoDevices: [],
    isStreaming: false,
    streamTimer: null,
    streamIntervalMs: 600,
    isProcessingFrame: false,
    activeProfile: null,
    detectedObjects: [],
    detectedFaces: [],
    chatHistory: [],
    ttsEnabled: false,
    lastFrameTime: performance.now(),
    fps: 0,
    enrollStream: null,
    enrollSnapshotB64: null,
    profiles: [],
    highlightMode: 'normal', // 'normal' | 'minimal' | 'wireframe'
    lastDetections: null     // { objects, faces, imgSize }
  };

  // ─── DOM Selectors ────────────────────────────────────────────────────────────
  const els = {
    // Header
    geminiInd: document.getElementById('gemini-ind'),
    geminiStatusText: document.getElementById('gemini-status-text'),
    profileCountBadge: document.getElementById('profile-count-badge'),
    btnOpenProfiles: document.getElementById('btn-open-profiles'),
    btnEnrollProfile: document.getElementById('btn-enroll-profile'),
    btnOpenSettings: document.getElementById('btn-open-settings'),

    // Presets & Tabs
    tabWebcam: document.getElementById('tab-webcam'),
    tabUpload: document.getElementById('tab-upload'),
    sampleOffice: document.getElementById('sample-office'),
    sampleAlice: document.getElementById('sample-alice'),
    sampleBob: document.getElementById('sample-bob'),

    // Single Frame Viewport
    viewportContainer: document.getElementById('viewport-container'),
    webcamVideo: document.getElementById('webcam-video'),
    viewportImage: document.getElementById('viewport-image'),
    overlayCanvas: document.getElementById('overlay-canvas'),
    cameraSelect: document.getElementById('camera-select'),
    scanLaser: document.getElementById('scan-laser'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('file-input'),

    // Telemetry
    telemetryFps: document.getElementById('telemetry-fps'),
    telemetryLatency: document.getElementById('telemetry-latency'),
    telemetryObjCount: document.getElementById('telemetry-obj-count'),
    telemetryFaceCount: document.getElementById('telemetry-face-count'),

    // Controls
    btnToggleCam: document.getElementById('btn-toggle-cam'),
    camBtnText: document.getElementById('cam-btn-text'),
    btnSwitchCam: document.getElementById('btn-switch-cam'),
    btnSnapshotAnalyze: document.getElementById('btn-snapshot-analyze'),
    btnToggleStream: document.getElementById('btn-toggle-stream'),
    streamIndicator: document.getElementById('stream-indicator'),
    streamBtnText: document.getElementById('stream-btn-text'),
    confSlider: document.getElementById('conf-slider'),
    confVal: document.getElementById('conf-val'),
    filterMode: document.getElementById('filter-mode'),
    highlightMode: document.getElementById('highlight-mode'),

    // Tray
    detectionSummaryText: document.getElementById('detection-summary-text'),
    detectionChips: document.getElementById('detection-chips'),

    // Profile Card
    profileCardImg: document.getElementById('profile-card-img'),
    profileMatchBadge: document.getElementById('profile-match-badge'),
    profileCardName: document.getElementById('profile-card-name'),
    profileCardRole: document.getElementById('profile-card-role'),
    profileCardDept: document.getElementById('profile-card-dept'),
    btnAutoGreet: document.getElementById('btn-auto-greet'),

    // Chat
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    btnSendChat: document.getElementById('btn-send-chat'),
    btnToggleTts: document.getElementById('btn-toggle-tts'),
    promptPills: document.querySelectorAll('.prompt-pill'),

    // Modals
    modalSettings: document.getElementById('modal-settings'),
    inputApiKey: document.getElementById('input-api-key'),
    btnToggleKeyVisibility: document.getElementById('btn-toggle-key-visibility'),
    selectGeminiModel: document.getElementById('select-gemini-model'),
    keyTestFeedback: document.getElementById('key-test-feedback'),
    btnTestKey: document.getElementById('btn-test-key'),
    btnSaveKey: document.getElementById('btn-save-key'),

    modalEnroll: document.getElementById('modal-enroll'),
    enrollVideo: document.getElementById('enroll-video'),
    enrollPreview: document.getElementById('enroll-preview'),
    enrollCanvas: document.getElementById('enroll-canvas'),
    btnEnrollSnap: document.getElementById('btn-enroll-snap'),
    btnEnrollRetake: document.getElementById('btn-enroll-retake'),
    btnEnrollUploadFile: document.getElementById('btn-enroll-upload-file'),
    enrollFileInput: document.getElementById('enroll-file-input'),
    enrollName: document.getElementById('enroll-name'),
    enrollAge: document.getElementById('enroll-age'),
    enrollJob: document.getElementById('enroll-job'),
    enrollDept: document.getElementById('enroll-dept'),
    enrollNotes: document.getElementById('enroll-notes'),
    enrollFeedback: document.getElementById('enroll-feedback'),
    btnSubmitEnroll: document.getElementById('btn-submit-enroll'),

    modalProfiles: document.getElementById('modal-profiles'),
    profilesGrid: document.getElementById('profiles-grid')
  };

  // ─── Initialization ───────────────────────────────────────────────────────────
  async function init() {
    setupEventListeners();
    updateApiStatusDisplay();
    await checkSystemStatus();
    await loadProfilesDirectory();

    // Discover cameras and launch single frame view on Camera 0
    await discoverCameras();
    await startCamera(state.currentCameraIndex);
  }

  // ─── System Telemetry & Status ────────────────────────────────────────────────
  async function checkSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (data.server_api_key_configured && !state.apiKey) {
          state.apiKey = 'SERVER_CONFIGURED';
          updateApiStatusDisplay();
        }
      }
    } catch (e) {
      console.warn('Backend status check warning:', e);
    }
  }

  function updateApiStatusDisplay() {
    if (state.apiKey) {
      els.geminiInd.className = 'status-indicator active';
      els.geminiStatusText.textContent = `Gemini: Active (${state.modelName.replace('gemini-', '')})`;
    } else {
      els.geminiInd.className = 'status-indicator warning';
      els.geminiStatusText.textContent = 'Gemini: Key Needed';
    }
  }

  // ─── Camera Management (Built-in & External USB Camera Support) ─────────────
  let isDiscoveringCameras = false;

  async function discoverCameras() {
    if (isDiscoveringCameras) return;
    isDiscoveringCameras = true;
    try {
      // 1. Fetch hardware camera indices and labels from backend OpenCV scan
      let serverCameras = [];
      try {
        const sRes = await fetch('/api/cameras');
        if (sRes.ok) {
          const sData = await sRes.json();
          serverCameras = sData.cameras || [];
        }
      } catch (err) {
        console.warn('Backend cameras probe warning:', err);
      }

      // 2. Query browser media devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      state.videoDevices = devices.filter(d => d.kind === 'videoinput');

      if (els.cameraSelect) {
        const currentVal = els.cameraSelect.value;
        els.cameraSelect.innerHTML = '';

        if (state.videoDevices.length > 0) {
          state.videoDevices.forEach((dev, idx) => {
            const opt = document.createElement('option');
            opt.value = idx.toString();

            let label = dev.label || '';
            if (!label || label.trim() === '') {
              if (serverCameras[idx] && serverCameras[idx].label) {
                label = serverCameras[idx].label;
              } else if (idx === 0) {
                label = 'Built-in Webcam (Cam 0)';
              } else {
                label = `External Camera (Cam ${idx})`;
              }
            } else {
              // Add helpful tag if it's external
              if (idx > 0 && !label.toLowerCase().includes('external')) {
                label = `${label} (External)`;
              }
            }

            const cleanLabel = label.length > 28 ? label.slice(0, 28) + '...' : label;
            opt.textContent = `📷 ${cleanLabel}`;
            if (idx === state.currentCameraIndex) opt.selected = true;
            els.cameraSelect.appendChild(opt);
          });
        } else if (serverCameras.length > 0) {
          serverCameras.forEach((cam) => {
            const opt = document.createElement('option');
            opt.value = cam.index.toString();
            opt.textContent = `📷 ${cam.label}`;
            if (cam.index === state.currentCameraIndex) opt.selected = true;
            els.cameraSelect.appendChild(opt);
          });
        } else {
          els.cameraSelect.innerHTML = `
            <option value="0" selected>📷 Cam 0: Built-in Camera</option>
            <option value="1">📷 Cam 1: External Camera</option>
          `;
        }
      }
    } catch (e) {
      console.warn('Camera enumeration error:', e);
    } finally {
      isDiscoveringCameras = false;
    }
  }

  async function startCamera(camIndex = 0) {
    stopWebcam();

    if (state.source !== 'webcam') return;

    state.currentCameraIndex = camIndex;

    els.webcamVideo.style.display = 'block';
    els.viewportImage.style.display = 'none';
    els.dropzone.style.display = 'none';
    els.camBtnText.textContent = 'Stop Camera';

    let videoConstraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 }
    };

    if (state.videoDevices && state.videoDevices[camIndex] && state.videoDevices[camIndex].deviceId) {
      videoConstraints.deviceId = { exact: state.videoDevices[camIndex].deviceId };
    } else if (camIndex === 1) {
      videoConstraints.facingMode = { ideal: 'environment' };
    } else {
      videoConstraints.facingMode = { ideal: 'user' };
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false
      });
      state.webcamStream = stream;
      els.webcamVideo.srcObject = stream;
      els.webcamVideo.onloadedmetadata = () => {
        resizeCanvas();
      };

      // Now that camera permissions are active, update labels with full device names
      setTimeout(discoverCameras, 300);

    } catch (err) {
      console.warn(`Camera ${camIndex} access error with strict constraints, attempting relaxed constraints:`, err);
      // Fallback try with relaxed resolution for external cameras
      try {
        let relaxedConstraints = { video: true, audio: false };
        if (state.videoDevices && state.videoDevices[camIndex] && state.videoDevices[camIndex].deviceId) {
          relaxedConstraints = {
            video: { deviceId: state.videoDevices[camIndex].deviceId },
            audio: false
          };
        }
        const streamFallback = await navigator.mediaDevices.getUserMedia(relaxedConstraints);
        state.webcamStream = streamFallback;
        els.webcamVideo.srcObject = streamFallback;
        els.camBtnText.textContent = 'Stop Camera';
        els.webcamVideo.onloadedmetadata = () => {
          resizeCanvas();
        };
        setTimeout(discoverCameras, 300);
      } catch (err2) {
        console.error('All camera attempts failed:', err2);
        els.camBtnText.textContent = 'Camera Blocked';
        loadPresetImage('/static/samples/office_desk.jpg');
      }
    }
  }

  function stopWebcam() {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(t => t.stop());
      state.webcamStream = null;
    }
    els.webcamVideo.srcObject = null;
    els.camBtnText.textContent = 'Start Camera';
    stopContinuousStream();
  }

  function toggleCamera() {
    if (state.webcamStream) {
      stopWebcam();
    } else {
      switchSource('webcam');
      startCamera(state.currentCameraIndex);
    }
  }

  function switchSource(sourceType) {
    state.source = sourceType;
    if (sourceType === 'webcam') {
      els.tabWebcam.classList.add('active');
      els.tabUpload.classList.remove('active');
      els.dropzone.style.display = 'none';
      els.viewportImage.style.display = 'none';
      els.webcamVideo.style.display = 'block';
      if (!state.webcamStream) startCamera(state.currentCameraIndex);
    } else {
      els.tabUpload.classList.add('active');
      els.tabWebcam.classList.remove('active');
      stopContinuousStream();
      stopWebcam();
      els.webcamVideo.style.display = 'none';
      if (!els.viewportImage.src || els.viewportImage.src.endsWith('/')) {
        els.dropzone.style.display = 'flex';
        els.viewportImage.style.display = 'none';
      } else {
        els.dropzone.style.display = 'none';
        els.viewportImage.style.display = 'block';
      }
    }
    clearCanvas();
  }

  function loadPresetImage(url) {
    switchSource('upload');
    els.viewportImage.crossOrigin = 'anonymous';
    els.viewportImage.src = url;
    els.viewportImage.onload = () => {
      els.dropzone.style.display = 'none';
      els.viewportImage.style.display = 'block';
      resizeCanvas();
      setTimeout(captureAndAnalyze, 100);
    };
  }

  // ─── Single Frame Capture & Analysis ─────────────────────────────────────────
  function getCurrentFrameBase64() {
    const hiddenCanvas = document.createElement('canvas');
    let srcW = 0, srcH = 0;

    if (state.source === 'webcam' && els.webcamVideo.videoWidth > 0) {
      srcW = els.webcamVideo.videoWidth;
      srcH = els.webcamVideo.videoHeight;
      hiddenCanvas.width = srcW;
      hiddenCanvas.height = srcH;
      const ctx = hiddenCanvas.getContext('2d');
      ctx.drawImage(els.webcamVideo, 0, 0, srcW, srcH);
    } else if (state.source === 'upload' && els.viewportImage.naturalWidth > 0) {
      srcW = els.viewportImage.naturalWidth;
      srcH = els.viewportImage.naturalHeight;
      hiddenCanvas.width = srcW;
      hiddenCanvas.height = srcH;
      const ctx = hiddenCanvas.getContext('2d');
      ctx.drawImage(els.viewportImage, 0, 0, srcW, srcH);
    } else {
      return null;
    }

    return hiddenCanvas.toDataURL('image/jpeg', 0.85);
  }

  async function captureAndAnalyze() {
    if (state.isProcessingFrame) return;

    const base64Img = getCurrentFrameBase64();
    if (!base64Img) return;

    state.isProcessingFrame = true;
    els.scanLaser.classList.add('active');

    const filterVal = els.filterMode.value;
    const detectObjects = filterVal === 'both' || filterVal === 'objects';
    const detectFaces = filterVal === 'both' || filterVal === 'faces';
    const confThresh = parseFloat(els.confSlider.value);

    const t0 = performance.now();

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: base64Img,
          conf_thresh: confThresh,
          detect_objects: detectObjects,
          detect_faces: detectFaces,
          draw_annotated: false
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const result = await response.json();
      const elapsed = Math.round(performance.now() - t0);

      // Update telemetry
      els.telemetryLatency.textContent = `${elapsed} ms`;
      els.telemetryObjCount.textContent = result.objects.length;
      els.telemetryFaceCount.textContent = result.faces.length;

      // Calculate FPS
      const now = performance.now();
      state.fps = Math.round(1000 / (now - state.lastFrameTime));
      state.lastFrameTime = now;
      els.telemetryFps.textContent = state.fps;

      state.detectedObjects = result.objects;
      state.detectedFaces = result.faces;

      // Render crisp neon overlays on the single canvas
      renderCanvasOverlays(result.objects, result.faces, result.image_size);

      // Update Detected Items Inventory Tray
      updateDetectionChips(result.objects, result.faces);

      // Update active profile in AI context panel
      if (result.active_profile) {
        updateActiveProfile(result.active_profile);
      }

    } catch (err) {
      console.error('Detection analysis failed:', err);
    } finally {
      state.isProcessingFrame = false;
      if (!state.isStreaming) {
        els.scanLaser.classList.remove('active');
      }
    }
  }

  // ─── Continuous Stream Scan ───────────────────────────────────────────────────
  function toggleContinuousStream() {
    if (state.isStreaming) {
      stopContinuousStream();
    } else {
      startContinuousStream();
    }
  }

  function startContinuousStream() {
    if (state.source !== 'webcam') {
      switchSource('webcam');
    }
    state.isStreaming = true;
    els.btnToggleStream.classList.add('btn-neon-violet');
    els.streamIndicator.style.display = 'block';
    els.streamBtnText.textContent = '■ Pause Scan';
    els.scanLaser.classList.add('active');

    runStreamLoop();
  }

  function stopContinuousStream() {
    state.isStreaming = false;
    els.btnToggleStream.classList.remove('btn-neon-violet');
    els.streamIndicator.style.display = 'none';
    els.streamBtnText.textContent = '▶ Continuous Scan';
    els.scanLaser.classList.remove('active');
    if (state.streamTimer) {
      clearTimeout(state.streamTimer);
      state.streamTimer = null;
    }
  }

  async function runStreamLoop() {
    if (!state.isStreaming) return;
    await captureAndAnalyze();
    if (state.isStreaming) {
      state.streamTimer = setTimeout(runStreamLoop, state.streamIntervalMs);
    }
  }

  // ─── Visual Theme & Category Palettes for Bounding Box Highlights ────────────
  const CATEGORY_STYLES = {
    // People & Faces
    person: {
      border: '#8b5cf6',
      fillNormal: 'rgba(139, 92, 246, 0.08)',
      fillMinimal: 'rgba(139, 92, 246, 0.03)',
      badgeBg: '#8b5cf6',
      badgeText: '#ffffff',
      icon: '👤'
    },
    face_matched: {
      border: '#00f2fe',
      fillNormal: 'rgba(0, 242, 254, 0.10)',
      fillMinimal: 'rgba(0, 242, 254, 0.04)',
      badgeBg: '#00f2fe',
      badgeText: '#040d1a',
      icon: '👤'
    },
    face_guest: {
      border: '#f59e0b',
      fillNormal: 'rgba(245, 158, 11, 0.10)',
      fillMinimal: 'rgba(245, 158, 11, 0.04)',
      badgeBg: '#f59e0b',
      badgeText: '#040d1a',
      icon: '👤'
    },
    // Electronics & Devices
    laptop: {
      border: '#0284c7',
      fillNormal: 'rgba(2, 132, 199, 0.08)',
      fillMinimal: 'rgba(2, 132, 199, 0.03)',
      badgeBg: '#0284c7',
      badgeText: '#ffffff',
      icon: '💻'
    },
    'cell phone': {
      border: '#0284c7',
      fillNormal: 'rgba(2, 132, 199, 0.08)',
      fillMinimal: 'rgba(2, 132, 199, 0.03)',
      badgeBg: '#0284c7',
      badgeText: '#ffffff',
      icon: '📱'
    },
    mouse: {
      border: '#0284c7',
      fillNormal: 'rgba(2, 132, 199, 0.08)',
      fillMinimal: 'rgba(2, 132, 199, 0.03)',
      badgeBg: '#0284c7',
      badgeText: '#ffffff',
      icon: '🖱️'
    },
    keyboard: {
      border: '#0284c7',
      fillNormal: 'rgba(2, 132, 199, 0.08)',
      fillMinimal: 'rgba(2, 132, 199, 0.03)',
      badgeBg: '#0284c7',
      badgeText: '#ffffff',
      icon: '⌨️'
    },
    tv: {
      border: '#0284c7',
      fillNormal: 'rgba(2, 132, 199, 0.08)',
      fillMinimal: 'rgba(2, 132, 199, 0.03)',
      badgeBg: '#0284c7',
      badgeText: '#ffffff',
      icon: '🖥️'
    },
    // Furniture
    chair: {
      border: '#10b981',
      fillNormal: 'rgba(16, 185, 129, 0.08)',
      fillMinimal: 'rgba(16, 185, 129, 0.03)',
      badgeBg: '#10b981',
      badgeText: '#022114',
      icon: '🪑'
    },
    couch: {
      border: '#10b981',
      fillNormal: 'rgba(16, 185, 129, 0.08)',
      fillMinimal: 'rgba(16, 185, 129, 0.03)',
      badgeBg: '#10b981',
      badgeText: '#022114',
      icon: '🛋️'
    },
    'dining table': {
      border: '#10b981',
      fillNormal: 'rgba(16, 185, 129, 0.08)',
      fillMinimal: 'rgba(16, 185, 129, 0.03)',
      badgeBg: '#10b981',
      badgeText: '#022114',
      icon: '🪵'
    },
    // Nature & Items
    'potted plant': {
      border: '#65a30d',
      fillNormal: 'rgba(101, 163, 13, 0.08)',
      fillMinimal: 'rgba(101, 163, 13, 0.03)',
      badgeBg: '#65a30d',
      badgeText: '#ffffff',
      icon: '🪴'
    },
    cup: {
      border: '#ea580c',
      fillNormal: 'rgba(234, 88, 12, 0.08)',
      fillMinimal: 'rgba(234, 88, 12, 0.03)',
      badgeBg: '#ea580c',
      badgeText: '#ffffff',
      icon: '☕'
    },
    bottle: {
      border: '#ea580c',
      fillNormal: 'rgba(234, 88, 12, 0.08)',
      fillMinimal: 'rgba(234, 88, 12, 0.03)',
      badgeBg: '#ea580c',
      badgeText: '#ffffff',
      icon: '🍾'
    },
    book: {
      border: '#d97706',
      fillNormal: 'rgba(217, 119, 6, 0.08)',
      fillMinimal: 'rgba(217, 119, 6, 0.03)',
      badgeBg: '#d97706',
      badgeText: '#ffffff',
      icon: '📖'
    },
    default: {
      border: '#10b981',
      fillNormal: 'rgba(16, 185, 129, 0.08)',
      fillMinimal: 'rgba(16, 185, 129, 0.03)',
      badgeBg: '#10b981',
      badgeText: '#022114',
      icon: '🎯'
    }
  };

  // ─── Canvas Overlay Renderer with Normal Highlight Engine ─────────────────────
  function resizeCanvas() {
    const canvas = els.overlayCanvas;
    const rect = els.viewportContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function clearCanvas() {
    const canvas = els.overlayCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function renderCanvasOverlays(objects, faces, imgSize) {
    state.lastDetections = { objects, faces, imgSize };

    const canvas = els.overlayCanvas;
    const rect = els.viewportContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Keep internal canvas resolution matched to viewport
    const targetW = Math.round(rect.width * dpr);
    const targetH = Math.round(rect.height * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    }

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (!imgSize || imgSize.width === 0 || imgSize.height === 0) return;

    // Compute aspect-ratio fitted coordinate scales
    const scaleX = rect.width / imgSize.width;
    const scaleY = rect.height / imgSize.height;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (rect.width - imgSize.width * scale) / 2;
    const offsetY = (rect.height - imgSize.height * scale) / 2;

    const mode = state.highlightMode; // 'normal' | 'minimal' | 'wireframe'

    // 1. Draw YOLO Object Bounding Boxes with Normal Highlights
    objects.forEach(obj => {
      const [bx, by, bw, bh] = obj.bbox;
      const x = offsetX + bx * scale;
      const y = offsetY + by * scale;
      const w = bw * scale;
      const h = bh * scale;

      const cls = obj.class.toLowerCase();
      const style = CATEGORY_STYLES[cls] || CATEGORY_STYLES.default;
      const label = `${style.icon} ${obj.class.toUpperCase()} ${Math.round(obj.confidence * 100)}%`;

      drawHighlightBox(ctx, x, y, w, h, style, mode, label, null, false);
    });

    // 2. Draw Face Recognition Bounding Boxes with Clean Highlights
    faces.forEach(face => {
      const [fbx, fby, fbw, fbh] = face.bbox;
      const x = offsetX + fbx * scale;
      const y = offsetY + fby * scale;
      const w = fbw * scale;
      const h = fbh * scale;

      const isMatched = face.matched;
      const style = isMatched ? CATEGORY_STYLES.face_matched : CATEGORY_STYLES.face_guest;
      const confStr = isMatched ? ` (${Math.round(face.confidence * 100)}%)` : '';
      const label = `👤 ${face.name}${confStr}`;
      
      const attrParts = [];
      if (face.gender && face.gender !== 'Unknown') attrParts.push(face.gender);
      if (face.age && face.age !== 'Unknown') attrParts.push(`Age: ${face.age}`);
      if (face.emotion && face.emotion !== 'Unknown') attrParts.push(face.emotion);
      const subLabel = attrParts.length > 0 ? attrParts.join(' • ') : (isMatched && face.job ? `${face.job} • ${face.department || 'IT'}` : 'Visitor');

      drawHighlightBox(ctx, x, y, w, h, style, mode, label, subLabel, true);
    });
  }

  function drawHighlightBox(ctx, x, y, w, h, style, mode, label, subLabel, isFace) {
    ctx.save();

    // 1. Normal Highlight Fill (Subtle, clean translucent tint)
    if (mode === 'normal' || mode === 'vivid') {
      ctx.fillStyle = style.fillNormal;
      drawRoundedRect(ctx, x, y, w, h, 4);
      ctx.fill();
    } else if (mode === 'minimal' || mode === 'subtle') {
      ctx.fillStyle = style.fillMinimal;
      drawRoundedRect(ctx, x, y, w, h, 4);
      ctx.fill();
    }

    // 2. Crisp, Clean Bounding Box (No blur cloud)
    ctx.strokeStyle = style.border;
    ctx.lineWidth = isFace ? 2 : 1.75;
    ctx.shadowBlur = 0;
    drawRoundedRect(ctx, x, y, w, h, 4);
    ctx.stroke();

    // 3. Compact Clean Label Tag
    ctx.font = '600 11px "JetBrains Mono", monospace';
    const textMetrics = ctx.measureText(label);
    const tagH = 18;
    const tagW = textMetrics.width + 12;
    const tagY = y - tagH >= 0 ? y - tagH : y;
    const tagX = Math.max(0, x);

    // Tag background
    ctx.fillStyle = style.badgeBg;
    drawRoundedRect(ctx, tagX, tagY, tagW, tagH, 3);
    ctx.fill();

    // Tag text
    ctx.fillStyle = style.badgeText;
    ctx.fillText(label, tagX + 6, tagY + 13);

    // 4. Subtle role sub-label for identified face
    if (subLabel && isFace && y + h + 22 < ctx.canvas.height) {
      ctx.font = '500 10px "Plus Jakarta Sans", sans-serif';
      const subMetrics = ctx.measureText(subLabel);
      const subW = subMetrics.width + 10;
      const subH = 16;
      const subX = Math.max(0, x);
      const subY = y + h + 3;

      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.strokeStyle = style.border;
      ctx.lineWidth = 1;
      drawRoundedRect(ctx, subX, subY, subW, subH, 3);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(subLabel, subX + 5, subY + 12);
    }

    ctx.restore();
  }

  function drawRoundedRect(ctx, x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ─── Detection Inventory Tray ─────────────────────────────────────────────────
  function updateDetectionChips(objects, faces) {
    const container = els.detectionChips;
    container.innerHTML = '';

    const total = objects.length + faces.length;
    if (total === 0) {
      els.detectionSummaryText.textContent = 'Scanning scene...';
      container.innerHTML = '<span class="detected-badge object">No items in frame</span>';
      return;
    }

    els.detectionSummaryText.textContent = `${total} detections (${faces.length} faces, ${objects.length} objects)`;

    // Faces first
    faces.forEach(f => {
      const chip = document.createElement('span');
      chip.className = 'detected-badge face';
      const ageStr = f.age && f.age !== 'Unknown' ? ` • Age: ${f.age}` : '';
      const emoStr = f.emotion ? ` • ${f.emotion}` : '';
      chip.textContent = `👤 ${f.name}${ageStr}${emoStr}`;
      chip.title = `Role: ${f.job || 'Visitor'} | Gender: ${f.gender || 'N/A'}`;
      chip.onclick = () => {
        sendChatMessage(`Tell me about ${f.name} (detected age ${f.age || 'N/A'}, emotion ${f.emotion || 'Neutral'}).`);
      };
      container.appendChild(chip);
    });

    // Object counts
    const counts = {};
    objects.forEach(o => {
      counts[o.class] = (counts[o.class] || 0) + 1;
    });

    Object.entries(counts).forEach(([cls, cnt]) => {
      const chip = document.createElement('span');
      chip.className = 'detected-badge object';
      chip.textContent = `${cnt}x ${cls}`;
      chip.title = `Click to ask AI about ${cls}`;
      chip.onclick = () => {
        sendChatMessage(`I notice a ${cls} in front of me. Can you give me insights or tips regarding it?`);
      };
      container.appendChild(chip);
    });
  }

  // ─── Active Profile Card Synchronization ──────────────────────────────────────
  function updateActiveProfile(profile) {
    state.activeProfile = profile;
    els.profileCardName.textContent = profile.name;
    const ageTag = profile.age && profile.age !== 'Unknown' ? `Age: ${profile.age}` : '';
    const emoTag = profile.emotion ? `Mood: ${profile.emotion}` : '';
    const details = [profile.job, profile.department, ageTag, emoTag].filter(Boolean).join(' • ');
    els.profileCardRole.textContent = details || `${profile.job} • ${profile.department}`;
    els.profileCardDept.textContent = profile.notes || 'Identified Profile Context';
    els.profileMatchBadge.textContent = `${Math.round(profile.confidence * 100)}%`;

    if (profile.image) {
      els.profileCardImg.src = profile.image.startsWith('/') ? profile.image : `/${profile.image}`;
    } else if (profile.name && profile.name.toLowerCase().includes('alice')) {
      els.profileCardImg.src = '/static/profiles/alice.jpg';
    } else if (profile.name && profile.name.toLowerCase().includes('bob')) {
      els.profileCardImg.src = '/static/profiles/bob.jpg';
    }
  }

  // ─── Conversational AI Assistant ──────────────────────────────────────────────
  async function sendChatMessage(customMsg = null) {
    const text = customMsg || els.chatInput.value.trim();
    if (!text) return;

    if (!customMsg) {
      els.chatInput.value = '';
    }

    appendChatMessage('user', text);
    state.chatHistory.push({ role: 'user', content: text });

    const typingId = showTypingIndicator();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          api_key: state.apiKey === 'SERVER_CONFIGURED' ? null : state.apiKey,
          model_name: state.modelName,
          profile: state.activeProfile,
          detected_objects: state.detectedObjects.map(o => o.class),
          history: state.chatHistory.slice(-6)
        })
      });

      removeTypingIndicator(typingId);

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Chat request failed');
      }

      const data = await response.json();
      appendChatMessage('assistant', data.reply);
      state.chatHistory.push({ role: 'assistant', content: data.reply });

      if (state.ttsEnabled) {
        speakText(data.reply);
      }

    } catch (err) {
      removeTypingIndicator(typingId);
      appendChatMessage('assistant', `❌ **Error:** ${err.message}`);
    }
  }

  async function triggerAutoGreet() {
    if (!state.activeProfile) {
      appendChatMessage('assistant', 'No face is currently identified in front of the camera.');
      return;
    }

    const typingId = showTypingIndicator();

    try {
      const res = await fetch('/api/greet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: state.apiKey === 'SERVER_CONFIGURED' ? null : state.apiKey,
          model_name: state.modelName,
          profile: state.activeProfile,
          detected_objects: state.detectedObjects.map(o => o.class)
        })
      });

      removeTypingIndicator(typingId);

      if (res.ok) {
        const data = await res.json();
        appendChatMessage('assistant', data.greeting);
        if (state.ttsEnabled) {
          speakText(data.greeting);
        }
      }
    } catch (e) {
      removeTypingIndicator(typingId);
      console.warn('Greet error:', e);
    }
  }

  function appendChatMessage(role, text) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar';
    avatar.textContent = role === 'user' ? 'YOU' : 'AI';

    const content = document.createElement('div');
    content.className = 'bubble-content';
    content.innerHTML = renderMarkdown(text);

    const time = document.createElement('div');
    time.className = 'bubble-time';
    const d = new Date();
    time.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    content.appendChild(time);

    bubble.appendChild(avatar);
    bubble.appendChild(content);

    els.chatMessages.appendChild(bubble);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  }

  function showTypingIndicator() {
    const id = `typing-${Date.now()}`;
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble assistant';
    bubble.id = id;

    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar';
    avatar.textContent = 'AI';

    const content = document.createElement('div');
    content.className = 'bubble-content';
    content.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';

    bubble.appendChild(avatar);
    bubble.appendChild(content);
    els.chatMessages.appendChild(bubble);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    return id;
  }

  function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function renderMarkdown(md) {
    let html = md
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
      .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 4px;border-radius:4px;">$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>');

    return `<p>${html}</p>`;
  }

  function toggleTts() {
    state.ttsEnabled = !state.ttsEnabled;
    if (state.ttsEnabled) {
      els.btnToggleTts.classList.add('btn-neon-violet');
      speakText('Voice narration activated.');
    } else {
      els.btnToggleTts.classList.remove('btn-neon-violet');
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
  }

  function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const cleanText = text.replace(/[*_#`[\]]/g, '').trim();
    const utter = new SpeechSynthesisUtterance(cleanText);
    utter.rate = 1.05;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
  }

  // ─── Profile Enrollment (Webcam or File) ──────────────────────────────────────
  async function openEnrollModal() {
    openModal('modal-enroll');
    els.enrollFeedback.style.display = 'none';
    els.enrollVideo.style.display = 'block';
    els.enrollPreview.style.display = 'none';
    els.btnEnrollSnap.style.display = 'inline-flex';
    els.btnEnrollRetake.style.display = 'none';

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      });
      state.enrollStream = stream;
      els.enrollVideo.srcObject = stream;
    } catch (e) {
      console.warn('Enroll webcam unavailable:', e);
      els.enrollVideo.style.display = 'none';
      els.enrollPreview.style.display = 'block';
      els.enrollPreview.src = '/static/profiles/alice.jpg';
    }
  }

  function closeEnrollModal() {
    closeModal('modal-enroll');
    if (state.enrollStream) {
      state.enrollStream.getTracks().forEach(t => t.stop());
      state.enrollStream = null;
    }
  }

  function snapEnrollPhoto() {
    if (!state.enrollStream) return;
    const canvas = els.enrollCanvas;
    canvas.width = els.enrollVideo.videoWidth;
    canvas.height = els.enrollVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(els.enrollVideo, 0, 0);

    state.enrollSnapshotB64 = canvas.toDataURL('image/jpeg', 0.9);
    els.enrollPreview.src = state.enrollSnapshotB64;
    els.enrollPreview.style.display = 'block';
    els.enrollVideo.style.display = 'none';
    els.btnEnrollSnap.style.display = 'none';
    els.btnEnrollRetake.style.display = 'inline-flex';
  }

  function retakeEnrollPhoto() {
    state.enrollSnapshotB64 = null;
    els.enrollPreview.style.display = 'none';
    els.enrollVideo.style.display = 'block';
    els.btnEnrollSnap.style.display = 'inline-flex';
    els.btnEnrollRetake.style.display = 'none';
  }

  async function submitEnrollProfile() {
    const name = els.enrollName.value.trim();
    if (!name) {
      showEnrollFeedback('Please provide a full name.', false);
      return;
    }

    if (!state.enrollSnapshotB64) {
      showEnrollFeedback('Please snap a photo or upload an image first.', false);
      return;
    }

    showEnrollFeedback('Encoding face landmarks & saving profile...', true);

    try {
      const res = await fetch('/api/profiles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          age: els.enrollAge.value.trim() || 'Unknown',
          job: els.enrollJob.value.trim() || 'Team Member',
          department: els.enrollDept.value.trim() || 'General',
          notes: els.enrollNotes.value.trim() || '',
          image_base64: state.enrollSnapshotB64
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Enrollment failed');
      }

      showEnrollFeedback(`✅ ${data.message}`, true);
      await loadProfilesDirectory();

      setTimeout(() => {
        closeEnrollModal();
        els.enrollName.value = '';
        els.enrollAge.value = '';
        els.enrollJob.value = '';
        els.enrollDept.value = '';
        els.enrollNotes.value = '';
        state.enrollSnapshotB64 = null;
      }, 1200);

    } catch (err) {
      showEnrollFeedback(`❌ ${err.message}`, false);
    }
  }

  function showEnrollFeedback(msg, isSuccess) {
    els.enrollFeedback.style.display = 'block';
    els.enrollFeedback.style.background = isSuccess ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)';
    els.enrollFeedback.style.color = isSuccess ? '#10b981' : '#f43f5e';
    els.enrollFeedback.textContent = msg;
  }

  // ─── Profiles Directory Viewer ────────────────────────────────────────────────
  async function loadProfilesDirectory() {
    try {
      const res = await fetch('/api/profiles');
      if (res.ok) {
        const data = await res.json();
        state.profiles = data.profiles;
        els.profileCountBadge.textContent = state.profiles.length;
        renderProfilesGrid();
      }
    } catch (e) {
      console.warn('Failed to load profiles:', e);
    }
  }

  function renderProfilesGrid() {
    const grid = els.profilesGrid;
    grid.innerHTML = '';

    state.profiles.forEach(p => {
      const card = document.createElement('div');
      card.className = 'profile-card-item';

      const img = document.createElement('img');
      img.className = 'profile-card-img';
      img.src = `/${p.image}`;
      img.alt = p.name;
      img.onerror = () => { img.src = '/static/profiles/alice.jpg'; };

      const name = document.createElement('div');
      name.className = 'profile-card-name';
      name.textContent = p.name;

      const role = document.createElement('div');
      role.className = 'profile-card-role';
      role.textContent = `${p.job} • ${p.department}`;

      const delBtn = document.createElement('button');
      delBtn.className = 'profile-card-del-btn';
      delBtn.textContent = '✕ Remove';
      delBtn.onclick = async (e) => {
        e.stopPropagation();
        if (confirm(`Remove profile "${p.name}"?`)) {
          await deleteProfile(p.id);
        }
      };

      card.appendChild(img);
      card.appendChild(name);
      card.appendChild(role);
      card.appendChild(delBtn);

      card.onclick = () => {
        updateActiveProfile({ ...p, confidence: 1.0 });
        closeModal('modal-profiles');
        sendChatMessage(`Hi! I am ${p.name}. Can you review my role and welcome me?`);
      };

      grid.appendChild(card);
    });
  }

  async function deleteProfile(profileId) {
    try {
      const res = await fetch(`/api/profiles/${profileId}`, { method: 'DELETE' });
      if (res.ok) {
        await loadProfilesDirectory();
      }
    } catch (e) {
      alert('Delete failed: ' + e.message);
    }
  }

  // ─── Settings Modal & API Key Testing ─────────────────────────────────────────
  async function testGeminiApiKey() {
    const key = els.inputApiKey.value.trim();
    if (!key) {
      showKeyFeedback('Please enter an API key to test.', false);
      return;
    }

    showKeyFeedback('Testing connection to Google GenAI API...', true);

    try {
      const res = await fetch('/api/validate-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: key,
          model_name: els.selectGeminiModel.value
        })
      });

      const data = await res.json();
      if (res.ok && data.valid) {
        showKeyFeedback(`✅ ${data.message} (${data.model})`, true);
      } else {
        showKeyFeedback(`❌ ${data.error || 'Key validation failed.'}`, false);
      }
    } catch (err) {
      showKeyFeedback(`❌ Network error: ${err.message}`, false);
    }
  }

  function saveGeminiSettings() {
    const key = els.inputApiKey.value.trim();
    const model = els.selectGeminiModel.value;

    state.apiKey = key;
    state.modelName = model;

    localStorage.setItem('omnivision_gemini_key', key);
    localStorage.setItem('omnivision_gemini_model', model);

    updateApiStatusDisplay();
    closeModal('modal-settings');

    appendChatMessage('assistant', `🔑 **Gemini API Key Configured!** Model: \`${model}\`. Vision assistant is ready.`);
  }

  function showKeyFeedback(msg, isSuccess) {
    els.keyTestFeedback.style.display = 'block';
    els.keyTestFeedback.style.background = isSuccess ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)';
    els.keyTestFeedback.style.color = isSuccess ? '#10b981' : '#f43f5e';
    els.keyTestFeedback.textContent = msg;
  }

  function openModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('open');
  }

  function closeModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('open');
  }

  // ─── Setup Event Listeners ────────────────────────────────────────────────────
  function setupEventListeners() {
    window.addEventListener('resize', () => {
      resizeCanvas();
      if (state.lastDetections) {
        renderCanvasOverlays(state.lastDetections.objects, state.lastDetections.faces, state.lastDetections.imgSize);
      }
    });

    // Source Tabs
    els.tabWebcam.addEventListener('click', () => switchSource('webcam'));
    els.tabUpload.addEventListener('click', () => switchSource('upload'));

    // Sample Presets
    els.sampleOffice.addEventListener('click', () => loadPresetImage('/static/samples/office_desk.jpg'));
    els.sampleAlice.addEventListener('click', () => loadPresetImage('/static/samples/alice_test.jpg'));
    els.sampleBob.addEventListener('click', () => loadPresetImage('/static/samples/bob_test.jpg'));

    // Camera Switcher (Built-in Cam 0 & External Cam 1)
    if (els.cameraSelect) {
      els.cameraSelect.addEventListener('change', async (e) => {
        const camIdx = parseInt(e.target.value) || 0;
        await startCamera(camIdx);
        if (state.isStreaming) {
          stopContinuousStream();
          startContinuousStream();
        }
      });
    }

    if (els.btnSwitchCam) {
      els.btnSwitchCam.addEventListener('click', async () => {
        const total = (state.videoDevices && state.videoDevices.length > 0) ? state.videoDevices.length : 2;
        const nextIdx = (state.currentCameraIndex + 1) % total;
        if (els.cameraSelect) els.cameraSelect.value = nextIdx.toString();
        await startCamera(nextIdx);
        if (state.isStreaming) {
          stopContinuousStream();
          startContinuousStream();
        }
      });
    }

    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', async () => {
        console.log('Camera hardware device change detected, refreshing list...');
        await discoverCameras();
      });
    }

    // Controls
    els.btnToggleCam.addEventListener('click', toggleCamera);
    els.btnSnapshotAnalyze.addEventListener('click', captureAndAnalyze);
    els.btnToggleStream.addEventListener('click', toggleContinuousStream);

    els.confSlider.addEventListener('input', (e) => {
      els.confVal.textContent = `${Math.round(e.target.value * 100)}%`;
      if (!state.isStreaming) captureAndAnalyze();
    });

    els.filterMode.addEventListener('change', () => {
      if (!state.isStreaming) captureAndAnalyze();
    });

    if (els.highlightMode) {
      els.highlightMode.addEventListener('change', (e) => {
        state.highlightMode = e.target.value;
        if (state.lastDetections) {
          renderCanvasOverlays(state.lastDetections.objects, state.lastDetections.faces, state.lastDetections.imgSize);
        }
      });
    }

    // Dropzone & File Upload
    els.dropzone.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
          loadPresetImage(evt.target.result);
        };
        reader.readAsDataURL(file);
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      els.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        els.dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(eventName => {
      els.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        els.dropzone.classList.remove('dragover');
      });
    });
    els.dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const file = dt.files[0];
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (evt) => {
          loadPresetImage(evt.target.result);
        };
        reader.readAsDataURL(file);
      }
    });

    // Chat
    els.btnSendChat.addEventListener('click', () => sendChatMessage());
    els.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendChatMessage();
    });
    els.btnToggleTts.addEventListener('click', toggleTts);
    els.btnAutoGreet.addEventListener('click', triggerAutoGreet);

    els.promptPills.forEach(pill => {
      pill.addEventListener('click', () => {
        const prompt = pill.getAttribute('data-prompt');
        sendChatMessage(prompt);
      });
    });

    // Modals
    els.btnOpenSettings.addEventListener('click', () => {
      els.inputApiKey.value = state.apiKey === 'SERVER_CONFIGURED' ? '' : state.apiKey;
      els.selectGeminiModel.value = state.modelName;
      openModal('modal-settings');
    });

    els.btnToggleKeyVisibility.addEventListener('click', () => {
      const type = els.inputApiKey.type === 'password' ? 'text' : 'password';
      els.inputApiKey.type = type;
      els.btnToggleKeyVisibility.textContent = type === 'password' ? '👁' : '🔒';
    });

    els.btnTestKey.addEventListener('click', testGeminiApiKey);
    els.btnSaveKey.addEventListener('click', saveGeminiSettings);

    els.btnOpenProfiles.addEventListener('click', () => {
      renderProfilesGrid();
      openModal('modal-profiles');
    });

    els.btnEnrollProfile.addEventListener('click', openEnrollModal);
    els.btnEnrollSnap.addEventListener('click', snapEnrollPhoto);
    els.btnEnrollRetake.addEventListener('click', retakeEnrollPhoto);
    els.btnSubmitEnroll.addEventListener('click', submitEnrollProfile);

    els.btnEnrollUploadFile.addEventListener('click', () => els.enrollFileInput.click());
    els.enrollFileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
          state.enrollSnapshotB64 = evt.target.result;
          els.enrollPreview.src = evt.target.result;
          els.enrollPreview.style.display = 'block';
          els.enrollVideo.style.display = 'none';
          els.btnEnrollSnap.style.display = 'none';
          els.btnEnrollRetake.style.display = 'inline-flex';
        };
        reader.readAsDataURL(file);
      }
    });

    document.querySelectorAll('.modal-close-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const modalId = btn.getAttribute('data-close') || btn.closest('.modal-backdrop').id;
        if (modalId === 'modal-enroll') closeEnrollModal();
        else closeModal(modalId);
      });
    });

    document.querySelectorAll('.modal-backdrop').forEach(modal => {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          if (modal.id === 'modal-enroll') closeEnrollModal();
          else closeModal(modal.id);
        }
      });
    });
  }

  // ─── Start Application ────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);
})();
