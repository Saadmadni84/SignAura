// web/app.js

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const nmfText = document.getElementById('nmfText');
const featuresPre = document.getElementById('features');

// start webcam with error handling
async function startCamera() {
  try {
    console.log('Requesting camera access...');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    console.log('Camera started successfully:', overlay.width, 'x', overlay.height);
  } catch (err) {
    console.error('Camera error:', err);
    nmfText.innerText = 'Camera Error: ' + err.message + '. Please allow camera access and use HTTPS or localhost.';
  }
}
startCamera();

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
    transcript.push({ timestamp, text: result });
    lastDetection = result;
    lastDetectionTime = now;
    
    // Keep transcript to last 50 entries
    if (transcript.length > 50) transcript.shift();
  }
  
  return result;
}

// draw simple landmarks and show features
function drawResults(faceLandmarks, poseLandmarks, features) {
  ctx.clearRect(0,0,overlay.width,overlay.height);
  ctx.drawImage(video, 0, 0, overlay.width, overlay.height);
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
  featuresPre.textContent = JSON.stringify(features, null, 2);
  nmfText.innerText = mapToText(features);
}

// Setup MediaPipe face_mesh and pose
console.log('Initializing MediaPipe...');
console.log('FaceMesh available:', typeof FaceMesh !== 'undefined');
console.log('Pose available:', typeof Pose !== 'undefined');

const faceMesh = new FaceMesh({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`});
faceMesh.setOptions({maxNumFaces:1, refineLandmarks:true, minDetectionConfidence:0.5, minTrackingConfidence:0.5});
const pose = new Pose({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`});
pose.setOptions({modelComplexity:0, minDetectionConfidence:0.5});

console.log('MediaPipe models initialized');

// run both and combine results
let lastFace = null, lastPose = null;
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
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    await faceMesh.send({image: video});
    await pose.send({image: video});
  }
  requestAnimationFrame(loop);
}

video.addEventListener('playing', () => { 
  console.log('Video playing, starting detection loop...');
  loop().catch(err => {
    console.error('Loop error:', err);
    nmfText.innerText = 'Detection Error: ' + err.message;
  });
});

// startMediaPipe camera warmup after video plays
// Notes:
// The extractFeatures uses normalized ratios to be roughly scale-invariant.
// The rule thresholds (0.02, 0.08, 0.06, etc.) are starting values. You'll refine them quickly with short labelled samples.

// ========== PHASE C: UI FUNCTIONS ==========

// Update transcript display
function updateTranscriptDisplay() {
  const transcriptLog = document.getElementById('transcriptLog');
  if (!transcriptLog) return;
  
  // Create HTML for transcript entries
  let html = '<h3>Transcript History</h3>';
  if (transcript.length === 0) {
    html += '<p style="color: #888; font-style: italic;">No detections yet...</p>';
  } else {
    // Show most recent entries first
    const reversed = [...transcript].reverse();
    reversed.forEach(entry => {
      html += `<div class="transcript-entry">
        <span class="transcript-time">${entry.timestamp}</span>
        <span class="transcript-text">${entry.text}</span>
      </div>`;
    });
  }
  transcriptLog.innerHTML = html;
}

// Export transcript as text file
function exportTranscript() {
  if (transcript.length === 0) {
    alert('No transcript to export yet!');
    return;
  }
  
  // Format transcript as text
  let text = 'ISL Non-Manual Features Transcript\n';
  text += '==================================\n\n';
  transcript.forEach(entry => {
    text += `[${entry.timestamp}] ${entry.text}\n`;
  });
  
  // Create downloadable blob
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `isl-transcript-${new Date().toISOString().slice(0,10)}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  console.log('Transcript exported');
}

// Clear transcript history
function clearTranscript() {
  if (transcript.length === 0) {
    return;
  }
  
  if (confirm('Clear all transcript history?')) {
    transcript.length = 0; // Clear array
    lastDetection = '';
    updateTranscriptDisplay();
    console.log('Transcript cleared');
  }
}

// Toggle big font mode
function toggleBigFont() {
  document.body.classList.toggle('big-font');
  const btn = document.getElementById('bigFontBtn');
  if (document.body.classList.contains('big-font')) {
    btn.textContent = 'Normal Font';
  } else {
    btn.textContent = 'Big Font';
  }
}

// Save current features as labeled sample
function saveDataset() {
  const labelInput = document.getElementById('labelInput');
  const label = labelInput.value.trim();
  
  if (!label) {
    alert('Please enter a label first!');
    labelInput.focus();
    return;
  }
  
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
  
  // Provide feedback
  const sampleCount = dataset.split('\n').length - 2; // -2 for header and empty last line
  alert(`Sample saved! Total samples: ${sampleCount}`);
  
  // Clear label input for next sample
  labelInput.value = '';
  labelInput.focus();
  
  console.log('Saved sample:', label, currentFeatures);
}

// Download dataset from localStorage
function downloadDataset() {
  const dataset = localStorage.getItem('isl-nmf-dataset');
  
  if (!dataset) {
    alert('No dataset saved yet! Use "Save Sample" to collect data first.');
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
  
  console.log('Dataset downloaded');
}

// Update transcript display whenever transcript changes
const originalMapToText = mapToText;
function mapToTextWithUpdate(features) {
  const result = originalMapToText(features);
  updateTranscriptDisplay();
  return result;
}

// On-page overlay logger (helps when DevTools isn't available)
function startOverlayLogger() {
  if (window.__overlayLoggerId) return; // already running

  // create overlay container
  let overlay = document.getElementById('featureOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'featureOverlay';
    Object.assign(overlay.style, {
      position: 'fixed',
      right: '12px',
      top: '80px',
      zIndex: 99999,
      width: '320px',
      maxHeight: '60vh',
      overflow: 'auto',
      background: 'rgba(255,255,255,0.95)',
      color: '#111',
      border: '1px solid rgba(0,0,0,0.08)',
      boxShadow: '0 6px 18px rgba(0,0,0,0.12)',
      padding: '10px',
      fontFamily: 'monospace',
      fontSize: '12px',
      lineHeight: '1.2',
      borderRadius: '8px'
    });
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    Object.assign(closeBtn.style, { position: 'absolute', right: '6px', top: '6px', border: 'none', background: 'transparent', fontSize: '16px', cursor: 'pointer' });
    closeBtn.addEventListener('click', () => {
      stopOverlayLogger();
      overlay.remove();
    });
    overlay.appendChild(closeBtn);

    const title = document.createElement('div');
    title.textContent = 'Feature Logger';
    Object.assign(title.style, { fontWeight: '600', marginBottom: '6px' });
    overlay.appendChild(title);

    const content = document.createElement('pre');
    content.id = 'featureOverlayContent';
    Object.assign(content.style, { whiteSpace: 'pre-wrap', margin: '0' });
    overlay.appendChild(content);

    document.body.appendChild(overlay);
  }

  function computeRawBrowRatio() {
    try {
      if (!window.lastFace) return null;
      const lf = window.lastFace;
      const leftEyeTop = lf[105], leftEyeBottom = lf[159], rightEyeTop = lf[386], rightEyeBottom = lf[145];
      const leftBrow = lf[70], rightBrow = lf[300];
      const leftEyeCenterY = (leftEyeTop.y + leftEyeBottom.y)/2;
      const rightEyeCenterY = (rightEyeTop.y + rightEyeBottom.y)/2;
      const browEyeDist = ((leftEyeCenterY - leftBrow.y) + (rightEyeCenterY - rightBrow.y)) / 2;
      const interOcular = Math.hypot(lf[33].x - lf[263].x, lf[33].y - lf[263].y) + 1e-6;
      return browEyeDist / interOcular;
    } catch (e) {
      return null;
    }
  }

  const contentEl = document.getElementById('featureOverlayContent');
  window.__overlayLoggerId = setInterval(() => {
    const f = (window.featureHistory && featureHistory.length) ? featureHistory[featureHistory.length-1] : null;
    const rawBrow = computeRawBrowRatio();
    const txt = [];
    txt.push('time: ' + new Date().toLocaleTimeString());
    txt.push('lastDetection: ' + (window.lastDetection || '')); 
    txt.push('\nSmoothed features:');
    txt.push(f ? JSON.stringify(f, null, 2) : '  (no features yet)');
    txt.push('\nRaw browRatio: ' + (rawBrow !== null ? rawBrow.toFixed(4) : 'n/a'));
    txt.push('\nTranscript (last 6):');
    try { txt.push(JSON.stringify((window.transcript||[]).slice(-6), null, 2)); } catch(e){ txt.push('n/a'); }
    contentEl.textContent = txt.join('\n');
  }, 500);
}

function stopOverlayLogger(){
  if (window.__overlayLoggerId) { clearInterval(window.__overlayLoggerId); delete window.__overlayLoggerId; }
}

// Initialize on load
window.addEventListener('load', () => {
  console.log('Page loaded, UI functions initialized');
  updateTranscriptDisplay();
  // start overlay logger automatically so user doesn't need DevTools
  try { startOverlayLogger(); } catch (e) { console.warn('Overlay logger failed:', e); }
});

