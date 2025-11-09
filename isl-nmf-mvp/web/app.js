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
    torsoLean: 0
  };
  
  featureHistory.forEach(f => {
    avgFeatures.eyeRatio += f.eyeRatio;
    avgFeatures.browRatio += f.browRatio;
    avgFeatures.mouthOpen += f.mouthOpen;
    avgFeatures.roll += f.roll;
    avgFeatures.nod += f.nod;
    avgFeatures.torsoLean += f.torsoLean;
  });
  
  const count = featureHistory.length;
  avgFeatures.eyeRatio /= count;
  avgFeatures.browRatio /= count;
  avgFeatures.mouthOpen /= count;
  avgFeatures.roll /= count;
  avgFeatures.nod /= count;
  avgFeatures.torsoLean /= count;
  
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
  const browEyeDist = (dist(leftBrow,leftEyeTop) + dist(rightBrow,rightEyeTop))/2;
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
  const nod = (noseTip.y - chin.y); // negative when chin below nose

  // torso lean (if pose available)
  let torsoLean = 0;
  if (poseLandmarks && poseLandmarks.length>0) {
    const leftShoulder = poseLandmarks[11];
    const rightShoulder = poseLandmarks[12];
    const shoulderY = (leftShoulder.y + rightShoulder.y)/2;
    const midHip = poseLandmarks[23] && poseLandmarks[24] ? ((poseLandmarks[23].y+poseLandmarks[24].y)/2) : null;
    if (midHip) torsoLean = shoulderY - midHip; // positive means leaning forward/back depending on coordinate system
  }

  return {
    eyeRatio, browRatio, mouthOpen, roll, nod, torsoLean
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
  bodyLean: false
};

function mapToText(f) {
  if (!f) return "";
  const texts = [];

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

  // Eyebrows raised
  if (!detectionState.browsRaised && f.browRatio > 0.06) {
    detectionState.browsRaised = true;
  } else if (detectionState.browsRaised && f.browRatio < 0.05) {
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

  // Head nod
  if (!detectionState.headNod && f.nod < -0.02) {
    detectionState.headNod = true;
  } else if (detectionState.headNod && f.nod > -0.01) {
    detectionState.headNod = false;
  }
  if (detectionState.headNod) texts.push("head down / nod");

  // Body lean
  if (!detectionState.bodyLean && Math.abs(f.torsoLean) > 0.05) {
    detectionState.bodyLean = true;
  } else if (detectionState.bodyLean && Math.abs(f.torsoLean) < 0.03) {
    detectionState.bodyLean = false;
  }
  if (detectionState.bodyLean) texts.push("body leaning");

  return texts.length? texts.join(", ") : "neutral";
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
// The rule thresholds (0.02, 0.08, 0.06, etc.) are starting values. You’ll refine them quickly with short labelled samples.
