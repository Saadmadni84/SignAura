// web/app.js

// Wait for SignAuraUI to be available
let video, overlay, ctx;
let cameraStream = null; // Store camera stream for stop functionality

// Debouncing for stable text display
let lastStableText = "";
let currentCandidateText = "";
let candidateTextCount = 0;
const STABILITY_THRESHOLD = 8; // Text must be same for 8 frames (~270ms at 30fps) before updating

function initElements() {
  if (window.SignAuraUI && window.SignAuraUI.elems) {
    video = window.SignAuraUI.elems.video;
    overlay = window.SignAuraUI.elems.overlay;
    ctx = overlay.getContext('2d');
    return true;
  }
  return false;
}

// start webcam with error handling
async function startCamera() {
  try {
    console.log('startCamera() called');
    console.log('video element:', video);
    console.log('overlay element:', overlay);
    
    if (!video) {
      console.error('Video element not initialized!');
      alert('Video element not found. Please refresh the page.');
      return;
    }
    
    console.log('Requesting camera access...');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraStream = stream; // Store for later stop
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    console.log('Camera started successfully:', overlay.width, 'x', overlay.height);
    
    // Update UI status
    if (window.SignAuraUI) {
      window.SignAuraUI.setCameraStatus('Live');
      window.SignAuraUI.setCameraRunning(true);
    }
  } catch (err) {
    console.error('Camera error:', err);
    const errMsg = 'Camera Error: ' + err.message + '. Please allow camera access and use HTTPS or localhost.';
    if (window.SignAuraUI) {
      window.SignAuraUI.setTranslation(errMsg);
      window.SignAuraUI.setCameraStatus('Error');
      window.SignAuraUI.setCameraRunning(false);
    }
    alert(errMsg);
  }
}

// stop webcam
function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    video.srcObject = null;
    console.log('Camera stopped');
    
    // Update UI status
    if (window.SignAuraUI) {
      window.SignAuraUI.setCameraStatus('Stopped');
      window.SignAuraUI.setCameraRunning(false);
      window.SignAuraUI.setTranslation('Camera stopped');
    }
    
    // Clear canvas
    if (ctx) {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }
  }
}

// record current sample with selected label
function recordSample() {
  if (!window.SignAuraUI) return;
  
  const labelSelect = window.SignAuraUI.elems.labelSelect;
  const label = labelSelect.value;
  
  // Get current smoothed features
  if (featureHistory.length === 0) {
    alert('No features detected yet! Please wait for camera to start.');
    return;
  }
  
  const currentFeatures = featureHistory[featureHistory.length - 1];
  
  // Create CSV row
  const row = [
    label,
    currentFeatures.eyeRatio.toFixed(4),
    currentFeatures.browRatio.toFixed(4),
    currentFeatures.mouthOpen.toFixed(4),
    currentFeatures.roll.toFixed(2),
    currentFeatures.nod.toFixed(4),
    currentFeatures.torsoLean.toFixed(4),
    currentFeatures.browAsymmetry.toFixed(4),
    currentFeatures.smileMetric.toFixed(4),
    currentFeatures.gazeMetric.toFixed(4)
  ].join(',');
  
  // Check if we already have dataset in localStorage
  let dataset = localStorage.getItem('isl-nmf-dataset') || '';
  
  // Add header if first entry
  if (!dataset) {
    dataset = 'label,eyeRatio,browRatio,mouthOpen,roll,nod,torsoLean,browAsymmetry,smileMetric,gazeMetric\n';
  }
  
  dataset += row + '\n';
  localStorage.setItem('isl-nmf-dataset', dataset);
  
  // Provide visual feedback with pulse
  window.SignAuraUI.setRecording(true);
  setTimeout(() => {
    window.SignAuraUI.setRecording(false);
  }, 800);
  
  // Provide feedback
  const sampleCount = dataset.split('\n').length - 2; // -2 for header and empty last line
  console.log(`Sample recorded: ${label} (Total: ${sampleCount})`);
  
  // Show brief confirmation in translation box
  const currentText = window.SignAuraUI.elems.nmfText.textContent;
  window.SignAuraUI.setTranslation(`✓ Recorded: ${label}`, { confidence: 1 });
  setTimeout(() => {
    window.SignAuraUI.setTranslation(currentText);
  }, 1500);
}

// export dataset
function exportDataset() {
  const dataset = localStorage.getItem('isl-nmf-dataset');
  
  if (!dataset) {
    alert('No dataset saved yet! Use "Record Sample" to collect data first.');
    return;
  }
  
  const blob = new Blob([dataset], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `isl-nmf-dataset-${new Date().toISOString().slice(0,10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  const sampleCount = dataset.split('\n').length - 2;
  console.log(`Dataset exported (${sampleCount} samples)`);
}

// helper: distance between points
function dist(a,b) { return Math.hypot(a.x-b.x, a.y-b.y); }

// Temporal smoothing: rolling window for feature history
const featureHistory = [];
const SMOOTHING_WINDOW = 5; // average over last 5 frames

function smoothFeatures(rawFeatures) {
  if (!rawFeatures) return null;
  
  featureHistory.push(rawFeatures);
  if (featureHistory.length > SMOOTHING_WINDOW) {
    featureHistory.shift();
  }
  
  // Average numeric features
  const avgFeatures = {
    eyeRatio: 0,
    browRatio: 0,
    mouthOpen: 0,
    roll: 0,
    nod: 0,
    torsoLean: 0,
    browAsymmetry: 0,
    smileMetric: 0,
    gazeMetric: 0
  };
  
  featureHistory.forEach(f => {
    avgFeatures.eyeRatio += f.eyeRatio;
    avgFeatures.browRatio += f.browRatio;
    avgFeatures.mouthOpen += f.mouthOpen;
    avgFeatures.roll += f.roll;
    avgFeatures.nod += f.nod;
    avgFeatures.torsoLean += f.torsoLean;
    avgFeatures.browAsymmetry += f.browAsymmetry;
    avgFeatures.smileMetric += f.smileMetric;
    avgFeatures.gazeMetric += f.gazeMetric;
  });
  
  const count = featureHistory.length;
  avgFeatures.eyeRatio /= count;
  avgFeatures.browRatio /= count;
  avgFeatures.mouthOpen /= count;
  avgFeatures.roll /= count;
  avgFeatures.nod /= count;
  avgFeatures.torsoLean /= count;
  avgFeatures.browAsymmetry /= count;
  avgFeatures.smileMetric /= count;
  avgFeatures.gazeMetric /= count;
  
  return avgFeatures;
}

// feature extractor: takes faceLandmarks (468) and poseLandmarks
function extractFeatures(faceLandmarks, poseLandmarks) {
  if (!faceLandmarks || faceLandmarks.length === 0) return null;

  // indices for eyes, eyebrows, lips (MediaPipe layout)
  const leftEyeTop = faceLandmarks[105]; // approximate
  const leftEyeBottom = faceLandmarks[159];
  const rightEyeTop = faceLandmarks[386];
  const rightEyeBottom = faceLandmarks[145];
  const leftBrow = faceLandmarks[70];
  const rightBrow = faceLandmarks[300];
  const noseTip = faceLandmarks[1];
  const chin = faceLandmarks[152];
  const upperLip = faceLandmarks[13];
  const lowerLip = faceLandmarks[14];

  // eyeblink measure — normalized by interocular distance
  const eyeOpenness = ((dist(leftEyeTop,leftEyeBottom) + dist(rightEyeTop,rightEyeBottom))/2);
  const interOcular = dist(faceLandmarks[33], faceLandmarks[263]) + 1e-6;
  const eyeRatio = eyeOpenness / interOcular;

  // eyebrow distance to eye (raise detection)
  // Use vertical distance from eye center to brow for more stable raise detection
  const leftEyeCenterY = (leftEyeTop.y + leftEyeBottom.y) / 2;
  const rightEyeCenterY = (rightEyeTop.y + rightEyeBottom.y) / 2;
  const leftBrowY = leftBrow.y;
  const rightBrowY = rightBrow.y;
  // positive when brow is above eye center (remember: y increases downward in image coords)
  const browEyeDist = ((leftEyeCenterY - leftBrowY) + (rightEyeCenterY - rightBrowY)) / 2;
  const browRatio = browEyeDist / interOcular;

  // mouth open
  const mouthOpen = dist(upperLip, lowerLip) / interOcular;

  // head tilt (roll): use nose to left/right eye difference
  const leftEye = faceLandmarks[33];
  const rightEye = faceLandmarks[263];
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const roll = Math.atan2(dy, dx) * 180 / Math.PI; // degrees

  // head nod detection (based on nose-chin vertical diff)
  // Use chin - nose so value is positive when chin drops (nodding down)
  const nod = (chin.y - noseTip.y); // positive when chin below nose

  // torso lean (if pose available)
  let torsoLean = 0;
  if (poseLandmarks && poseLandmarks.length>0) {
    const leftShoulder = poseLandmarks[11];
    const rightShoulder = poseLandmarks[12];
    const shoulderY = (leftShoulder.y + rightShoulder.y)/2;
    const midHip = poseLandmarks[23] && poseLandmarks[24] ? ((poseLandmarks[23].y+poseLandmarks[24].y)/2) : null;
    if (midHip) torsoLean = shoulderY - midHip; // positive means leaning forward/back depending on coordinate system
  }

  // PHASE C: Additional NMF features
  
  // Eyebrow asymmetry (brow furrow / concern)
  const leftBrowDist = dist(leftBrow, leftEyeTop);
  const rightBrowDist = dist(rightBrow, rightEyeTop);
  const browAsymmetry = Math.abs(leftBrowDist - rightBrowDist) / interOcular;
  
  // Smile detection: use mouth width (distance between corners) normalized by inter-ocular distance
  const mouthCornerLeft = faceLandmarks[61];
  const mouthCornerRight = faceLandmarks[291];
  const mouthWidth = dist(mouthCornerLeft, mouthCornerRight) / interOcular;
  const smileMetric = mouthWidth;
  
  // Eye gaze direction (horizontal)
  const leftPupil = faceLandmarks[468]; // iris center (if refineLandmarks enabled)
  const rightPupil = faceLandmarks[473];
  const leftIris = leftPupil || leftEye;
  const rightIris = rightPupil || rightEye;
  const gazeLeft = (leftIris.x - leftEye.x) + (rightIris.x - rightEye.x);
  const gazeMetric = gazeLeft / interOcular; // negative = looking left, positive = looking right

  return {
    eyeRatio, browRatio, mouthOpen, roll, nod, torsoLean,
    browAsymmetry, smileMetric, gazeMetric
  };
}

// Simple rule-based mapper: convert features into short text
// Uses hysteresis thresholds to prevent flicker
const detectionState = {
  eyesClosed: false,
  mouthOpen: false,
  browsRaised: false,
  headTiltRight: false,
  headTiltLeft: false,
  headNod: false,
  bodyLean: false,
  browFurrow: false,
  smiling: false,
  gazingLeft: false,
  gazingRight: false
};

// Transcript history for export
const transcript = [];
let lastDetection = "";
let lastDetectionTime = Date.now();

function mapToText(f) {
  if (!f) return "";
  const texts = [];
  const timestamp = new Date().toLocaleTimeString();

  // Hysteresis: different thresholds for on/off transitions
  // Eyes closed/blink
  if (!detectionState.eyesClosed && f.eyeRatio < 0.02) {
    detectionState.eyesClosed = true;
  } else if (detectionState.eyesClosed && f.eyeRatio > 0.03) {
    detectionState.eyesClosed = false;
  }
  if (detectionState.eyesClosed) texts.push("eyes closed/blink");

  // Mouth open
  if (!detectionState.mouthOpen && f.mouthOpen > 0.08) {
    detectionState.mouthOpen = true;
  } else if (detectionState.mouthOpen && f.mouthOpen < 0.06) {
    detectionState.mouthOpen = false;
  }
  if (detectionState.mouthOpen) texts.push("mouth open (maybe surprise/talking)");

  // Eyebrows raised — use stricter vertical threshold to avoid constant firing
  // These thresholds assume browRatio is measured in units of inter-ocular distance
  if (!detectionState.browsRaised && f.browRatio > 0.12) {
    detectionState.browsRaised = true;
  } else if (detectionState.browsRaised && f.browRatio < 0.10) {
    detectionState.browsRaised = false;
  }
  if (detectionState.browsRaised) texts.push("eyebrows raised (surprise/ask)");

  // Head tilt right
  if (!detectionState.headTiltRight && f.roll > 10) {
    detectionState.headTiltRight = true;
    detectionState.headTiltLeft = false;
  } else if (detectionState.headTiltRight && f.roll < 8) {
    detectionState.headTiltRight = false;
  }
  if (detectionState.headTiltRight) texts.push("head tilted right");

  // Head tilt left
  if (!detectionState.headTiltLeft && f.roll < -10) {
    detectionState.headTiltLeft = true;
    detectionState.headTiltRight = false;
  } else if (detectionState.headTiltLeft && f.roll > -8) {
    detectionState.headTiltLeft = false;
  }
  if (detectionState.headTiltLeft) texts.push("head tilted left");

  // Head nod — detect when chin drops below nose (nod down). Using positive nod value.
  if (!detectionState.headNod && f.nod > 0.02) {
    detectionState.headNod = true;
  } else if (detectionState.headNod && f.nod < 0.01) {
    detectionState.headNod = false;
  }
  if (detectionState.headNod) texts.push("head down / nod");

  // Body lean — increase threshold to reduce false positives
  if (!detectionState.bodyLean && Math.abs(f.torsoLean) > 0.08) {
    detectionState.bodyLean = true;
  } else if (detectionState.bodyLean && Math.abs(f.torsoLean) < 0.05) {
    detectionState.bodyLean = false;
  }
  if (detectionState.bodyLean) texts.push("body leaning");

  // PHASE C: New features
  
  // Brow furrow (asymmetry / concern) — use a larger threshold to avoid small offsets
  if (!detectionState.browFurrow && f.browAsymmetry > 0.06) {
    detectionState.browFurrow = true;
  } else if (detectionState.browFurrow && f.browAsymmetry < 0.04) {
    detectionState.browFurrow = false;
  }
  if (detectionState.browFurrow) texts.push("brow furrow (concern)");

  // Smile — use mouth width metric (normalized). Increase threshold to avoid neutral mouth being detected.
  if (!detectionState.smiling && f.smileMetric > 0.30) {
    detectionState.smiling = true;
  } else if (detectionState.smiling && f.smileMetric < 0.28) {
    detectionState.smiling = false;
  }
  if (detectionState.smiling) texts.push("smiling");

  // Gaze direction
  if (!detectionState.gazingLeft && f.gazeMetric < -0.02) {
    detectionState.gazingLeft = true;
    detectionState.gazingRight = false;
  } else if (detectionState.gazingLeft && f.gazeMetric > -0.01) {
    detectionState.gazingLeft = false;
  }
  if (detectionState.gazingLeft) texts.push("looking left");

  if (!detectionState.gazingRight && f.gazeMetric > 0.02) {
    detectionState.gazingRight = true;
    detectionState.gazingLeft = false;
  } else if (detectionState.gazingRight && f.gazeMetric < 0.01) {
    detectionState.gazingRight = false;
  }
  if (detectionState.gazingRight) texts.push("looking right");

  const result = texts.length ? texts.join(", ") : "neutral";
  
  // Add to transcript if detection changed and held for >500ms
  const now = Date.now();
  if (result !== lastDetection && (now - lastDetectionTime) > 500) {
    if (window.SignAuraUI && window.SignAuraUI.state) {
      // Use SignAuraUI's internal transcript management
      window.SignAuraUI.state.transcript.push({ time: now, text: result });
    }
    lastDetection = result;
    lastDetectionTime = now;
  }
  
  return result;
}

// draw simple landmarks and show features
function drawResults(faceLandmarks, poseLandmarks, features) {
  ctx.clearRect(0,0,overlay.width,overlay.height);
  ctx.fillStyle = 'rgba(0,255,0,0.6)';
  if (faceLandmarks) {
    for (let i=0;i<faceLandmarks.length;i+=4) { // draw fewer points for speed
      const p = faceLandmarks[i];
      ctx.beginPath();
      ctx.arc(p.x*overlay.width, p.y*overlay.height, 2, 0, Math.PI*2);
      ctx.fill();
    }
  }
  if (poseLandmarks) {
    ctx.fillStyle = 'rgba(255,0,0,0.6)';
    [11,12,23,24].forEach(idx => {
      const p = poseLandmarks[idx];
      if (p) ctx.fillRect(p.x*overlay.width-3, p.y*overlay.height-3, 6,6);
    });
  }
  
  // Update UI with features and translation
  if (window.SignAuraUI && features) {
    window.SignAuraUI.updateFeatures({
      eye: features.eyeRatio,
      brow: features.browRatio,
      mouth: features.mouthOpen,
      roll: features.roll,
      nod: features.nod,
      torso: features.torsoLean
    });
    
    // Update progress bars
    updateProgressBars(features);
    
    // Debounced text update - only update when text is stable
    const translationText = mapToText(features);
    
    if (translationText === currentCandidateText) {
      candidateTextCount++;
      // Only update UI if text has been stable for enough frames
      if (candidateTextCount >= STABILITY_THRESHOLD && translationText !== lastStableText) {
        window.SignAuraUI.setTranslation(translationText);
        lastStableText = translationText;
      }
    } else {
      // New text detected, reset counter
      currentCandidateText = translationText;
      candidateTextCount = 1;
    }
  }
}

// Update progress bars with glow effect for high values
function updateProgressBars(features) {
  if (!features) return;
  
  // Eye openness (0.0 - 0.15 typical range, normalize to 0-100%)
  const eyePercent = Math.min(100, (features.eyeRatio / 0.15) * 100);
  const barEye = document.getElementById('barEye');
  const barEyeValue = document.getElementById('barEyeValue');
  if (barEye && barEyeValue) {
    barEye.style.width = eyePercent + '%';
    barEyeValue.textContent = Math.round(eyePercent) + '%';
    barEye.setAttribute('data-high', eyePercent > 70 ? 'true' : 'false');
  }
  
  // Brow raise (0.0 - 0.15 typical range, normalize to 0-100%)
  const browPercent = Math.min(100, (features.browRatio / 0.15) * 100);
  const barBrow = document.getElementById('barBrow');
  const barBrowValue = document.getElementById('barBrowValue');
  if (barBrow && barBrowValue) {
    barBrow.style.width = browPercent + '%';
    barBrowValue.textContent = Math.round(browPercent) + '%';
    barBrow.setAttribute('data-high', browPercent > 70 ? 'true' : 'false');
  }
  
  // Mouth open (0.0 - 0.15 typical range, normalize to 0-100%)
  const mouthPercent = Math.min(100, (features.mouthOpen / 0.15) * 100);
  const barMouth = document.getElementById('barMouth');
  const barMouthValue = document.getElementById('barMouthValue');
  if (barMouth && barMouthValue) {
    barMouth.style.width = mouthPercent + '%';
    barMouthValue.textContent = Math.round(mouthPercent) + '%';
    barMouth.setAttribute('data-high', mouthPercent > 50 ? 'true' : 'false');
  }
  
  // Head roll (-30 to +30 degrees typical, map to 0-100% where 50% = neutral)
  const rollNormalized = 50 + (features.roll / 30) * 50;
  const rollPercent = Math.max(0, Math.min(100, rollNormalized));
  const barRoll = document.getElementById('barRoll');
  const barRollValue = document.getElementById('barRollValue');
  if (barRoll && barRollValue) {
    barRoll.style.width = rollPercent + '%';
    barRollValue.textContent = Math.round(features.roll) + '°';
    barRoll.setAttribute('data-high', Math.abs(features.roll) > 15 ? 'true' : 'false');
  }
  
  // Nod (0.0 - 0.1 typical range, normalize to 0-100%)
  const nodPercent = Math.min(100, Math.abs(features.nod / 0.1) * 100);
  const barNod = document.getElementById('barNod');
  const barNodValue = document.getElementById('barNodValue');
  if (barNod && barNodValue) {
    barNod.style.width = nodPercent + '%';
    barNodValue.textContent = Math.round(nodPercent) + '%';
    barNod.setAttribute('data-high', nodPercent > 40 ? 'true' : 'false');
  }
  
  // Torso lean (0.0 - 0.15 typical range, normalize to 0-100%)
  const torsoPercent = Math.min(100, Math.abs(features.torsoLean / 0.15) * 100);
  const barTorso = document.getElementById('barTorso');
  const barTorsoValue = document.getElementById('barTorsoValue');
  if (barTorso && barTorsoValue) {
    barTorso.style.width = torsoPercent + '%';
    barTorsoValue.textContent = Math.round(torsoPercent) + '%';
    barTorso.setAttribute('data-high', torsoPercent > 50 ? 'true' : 'false');
  }
}

// Setup MediaPipe - wrapped in function to be called after video element is ready
let faceMesh, pose, lastFace = null, lastPose = null;

function initMediaPipe() {
  console.log('Initializing MediaPipe...');
  console.log('FaceMesh available:', typeof FaceMesh !== 'undefined');
  console.log('Pose available:', typeof Pose !== 'undefined');

  faceMesh = new FaceMesh({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`});
  faceMesh.setOptions({maxNumFaces:1, refineLandmarks:true, minDetectionConfidence:0.5, minTrackingConfidence:0.5});
  pose = new Pose({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`});
  pose.setOptions({modelComplexity:0, minDetectionConfidence:0.5});

  console.log('MediaPipe models initialized');

  // run both and combine results
  faceMesh.onResults((res) => {
    lastFace = res.multiFaceLandmarks && res.multiFaceLandmarks[0] ? res.multiFaceLandmarks[0] : null;
    const rawFeatures = extractFeatures(lastFace, lastPose);
    const smoothedFeatures = smoothFeatures(rawFeatures);
    drawResults(lastFace, lastPose, smoothedFeatures);
  });
  pose.onResults((res) => {
    lastPose = res.poseLandmarks || null;
    const rawFeatures = extractFeatures(lastFace, lastPose);
    const smoothedFeatures = smoothFeatures(rawFeatures);
    drawResults(lastFace, lastPose, smoothedFeatures);
  });

  // feed video into both
  async function loop() {
    if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
      await faceMesh.send({image: video});
      await pose.send({image: video});
    }
    requestAnimationFrame(loop);
  }

  video.addEventListener('playing', () => { 
    console.log('Video playing, starting detection loop...');
    loop().catch(err => {
      console.error('Loop error:', err);
      if (window.SignAuraUI) {
        window.SignAuraUI.setTranslation('Detection Error: ' + err.message);
      }
    });
  });
}


// startMediaPipe camera warmup after video plays
// Notes:
// The extractFeatures uses normalized ratios to be roughly scale-invariant.
// The rule thresholds (0.02, 0.08, 0.06, etc.) are starting values. You'll refine them quickly with short labelled samples.

// ========== PHASE C: UI FUNCTIONS ==========

// Export transcript - now uses SignAuraUI
function exportTranscript() {
  if (window.SignAuraUI) {
    window.SignAuraUI.exportTranscript();
  }
}

// Clear transcript - now uses SignAuraUI
function clearTranscript() {
  if (window.SignAuraUI) {
    window.SignAuraUI.clearTranscript();
  }
}

// Toggle big font mode - replaced by dark mode in new UI
function toggleBigFont() {
  if (window.SignAuraUI) {
    window.SignAuraUI.toggleDarkMode();
  }
}

// Download dataset from localStorage (legacy function for compatibility)
function downloadDataset() {
  exportDataset(); // Use the new exportDataset function
}

// Initialize immediately when script loads
console.log('🚀 app.js loaded!');

// Use a short delay to let SignAuraUI initialize first
setTimeout(() => {
  console.log('🚀 app.js: Starting initialization...');
  console.log('🔍 app.js: SignAuraUI available?', typeof window.SignAuraUI !== 'undefined');
  console.log('🔍 app.js: SignAuraUI.elems?', window.SignAuraUI?.elems);
  
  // Set initial status
  const statusEl = document.getElementById('cameraStatus');
  if (statusEl) {
    statusEl.textContent = 'Loading UI...';
  }
  
  // Wait for SignAuraUI to be available
  let attempts = 0;
  const initUI = setInterval(() => {
    attempts++;
    console.log(`🔄 Attempt ${attempts}: Checking for SignAuraUI...`);
    
    if (initElements()) {
      clearInterval(initUI);
      console.log('✅ app.js: UI elements initialized!');
      console.log('Video element:', video);
      console.log('Overlay element:', overlay);
      
      // Initialize MediaPipe now that video element is ready
      initMediaPipe();
      
      // Wire up control panel buttons
      if (window.SignAuraUI && window.SignAuraUI.elems) {
        const elems = window.SignAuraUI.elems;
        
        console.log('🔌 Wiring up buttons...');
        console.log('btnStart element:', elems.btnStart);
        
        // Camera controls
        elems.btnStart.addEventListener('click', () => {
          console.log('🟢 Start button clicked!');
          startCamera();
        });
        
        elems.btnStop.addEventListener('click', () => {
          console.log('🔴 Stop button clicked!');
          stopCamera();
        });
        
        elems.btnRecord.addEventListener('click', () => {
          console.log('⏺️ Record button clicked!');
          recordSample();
        });
        
        // Dataset export
        elems.btnExport.addEventListener('click', () => {
          console.log('💾 Export button clicked!');
          exportDataset();
        });
        
        // Keep old downloadDataBtn for compatibility
        if (elems.downloadDataBtn) {
          elems.downloadDataBtn.addEventListener('click', downloadDataset);
        }
        
        console.log('✅ All buttons wired up successfully');
      }
      
      // Don't auto-start camera - wait for user to click Start button
      console.log('✅ Ready. Click "Start Camera" to begin.');
      if (window.SignAuraUI) {
        window.SignAuraUI.setCameraStatus('Ready');
        window.SignAuraUI.setTranslation('👆 Click "Start Camera" to begin');
      }
    }
  }, 100);
  
  // Timeout after 10 seconds
  setTimeout(() => {
    clearInterval(initUI);
    if (!video) {
      console.error('❌ Failed to initialize UI elements after 10 seconds');
      console.error('SignAuraUI:', window.SignAuraUI);
      if (statusEl) {
        statusEl.textContent = 'Initialization Error';
        statusEl.className = 'px-2 py-0.5 rounded bg-rose-100 text-rose-800 text-xs dark:bg-rose-900/30 dark:text-rose-300';
      }
    }
  }, 10000);
}, 500); // Give SignAuraUI 500ms to initialize

