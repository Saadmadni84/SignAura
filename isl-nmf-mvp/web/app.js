// ============================================================================
// ISL NMF MVP - Real-time Non-Manual Feature Detection
// ============================================================================

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const nmfTextDiv = document.getElementById('nmfText');
const featuresDiv = document.getElementById('features');

// Set canvas size
canvas.width = 640;
canvas.height = 480;

// Feature state tracking
let currentFeatures = {
  eyebrowRaise: 0,
  mouthOpen: 0,
  headTilt: 0,
  headNod: 0,
  shoulderLean: 0,
  headShake: 0
};

let faceResults = null;
let poseResults = null;

// History for temporal features (head nod/shake)
const headHistory = [];
const MAX_HISTORY = 15; // ~0.5 seconds at 30fps

// ============================================================================
// MediaPipe Setup
// ============================================================================

const faceMesh = new FaceMesh({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
  }
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

faceMesh.onResults(onFaceResults);

const pose = new Pose({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
  }
});

pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

pose.onResults(onPoseResults);

// ============================================================================
// Camera Setup
// ============================================================================

const camera = new Camera(video, {
  onFrame: async () => {
    await faceMesh.send({ image: video });
    await pose.send({ image: video });
  },
  width: 640,
  height: 480
});

camera.start();

// ============================================================================
// Results Handlers
// ============================================================================

function onFaceResults(results) {
  faceResults = results;
  processFrame();
}

function onPoseResults(results) {
  poseResults = results;
}

// ============================================================================
// Feature Extraction
// ============================================================================

function processFrame() {
  if (!faceResults) return;

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw face mesh (optional visualization)
  if (faceResults.multiFaceLandmarks && faceResults.multiFaceLandmarks.length > 0) {
    const landmarks = faceResults.multiFaceLandmarks[0];
    
    // Draw landmarks
    ctx.fillStyle = '#00FF00';
    landmarks.forEach(landmark => {
      ctx.beginPath();
      ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 1, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Extract features
    extractFeatures(landmarks, poseResults);
    
    // Generate text from features
    const text = featuresToText(currentFeatures);
    nmfTextDiv.textContent = text || '—';
    
    // Display raw features
    displayFeatures();
  }
}

function extractFeatures(faceLandmarks, poseResults) {
  // 1. EYEBROW RAISE
  // Compare eyebrow landmarks (70, 63, 105, 66) with eye landmarks (159, 145, 386, 374)
  const leftEyebrow = faceLandmarks[70].y;
  const leftEye = faceLandmarks[159].y;
  const rightEyebrow = faceLandmarks[300].y;
  const rightEye = faceLandmarks[386].y;
  
  const leftBrowDist = leftEye - leftEyebrow;
  const rightBrowDist = rightEye - rightEyebrow;
  const avgBrowDist = (leftBrowDist + rightBrowDist) / 2;
  
  currentFeatures.eyebrowRaise = avgBrowDist > 0.04 ? 1 : 0;

  // 2. MOUTH OPEN
  // Upper lip (13) vs lower lip (14)
  const upperLip = faceLandmarks[13].y;
  const lowerLip = faceLandmarks[14].y;
  const mouthHeight = lowerLip - upperLip;
  
  currentFeatures.mouthOpen = mouthHeight > 0.02 ? 1 : 0;

  // 3. HEAD TILT (Roll - left/right tilt)
  const leftEar = faceLandmarks[234];
  const rightEar = faceLandmarks[454];
  const tiltAngle = Math.atan2(rightEar.y - leftEar.y, rightEar.x - leftEar.x);
  
  currentFeatures.headTilt = Math.abs(tiltAngle) > 0.15 ? (tiltAngle > 0 ? 1 : -1) : 0;

  // 4. HEAD NOD (Pitch - up/down movement)
  const noseTip = faceLandmarks[1];
  const chin = faceLandmarks[152];
  
  // Store current head position
  headHistory.push({
    noseY: noseTip.y,
    chinY: chin.y,
    timestamp: Date.now()
  });
  
  if (headHistory.length > MAX_HISTORY) {
    headHistory.shift();
  }

  // Detect vertical movement pattern
  if (headHistory.length >= 10) {
    const recentPositions = headHistory.slice(-10);
    const deltaY = recentPositions[recentPositions.length - 1].noseY - recentPositions[0].noseY;
    
    currentFeatures.headNod = Math.abs(deltaY) > 0.03 ? (deltaY > 0 ? 1 : -1) : 0;
  }

  // 5. HEAD SHAKE (Yaw - left/right rotation)
  if (headHistory.length >= 10) {
    const recentPositions = headHistory.slice(-10);
    const nose = faceLandmarks[1];
    const faceCenter = faceLandmarks[168];
    
    const horizontalOffset = nose.x - faceCenter.x;
    currentFeatures.headShake = Math.abs(horizontalOffset) > 0.05 ? (horizontalOffset > 0 ? 1 : -1) : 0;
  }

  // 6. SHOULDER LEAN (from pose landmarks)
  if (poseResults && poseResults.poseLandmarks) {
    const leftShoulder = poseResults.poseLandmarks[11];
    const rightShoulder = poseResults.poseLandmarks[12];
    
    if (leftShoulder && rightShoulder) {
      const shoulderTilt = Math.atan2(
        rightShoulder.y - leftShoulder.y,
        rightShoulder.x - leftShoulder.x
      );
      
      currentFeatures.shoulderLean = Math.abs(shoulderTilt) > 0.2 ? (shoulderTilt > 0 ? 1 : -1) : 0;
    }
  }
}

// ============================================================================
// Rule-based NMF to Text Mapping
// ============================================================================

function featuresToText(features) {
  const parts = [];

  // Eyebrow raise → Question, Surprise, Emphasis
  if (features.eyebrowRaise === 1) {
    parts.push('[QUESTION/EMPHASIS]');
  }

  // Mouth open → Exclamation, Surprise
  if (features.mouthOpen === 1) {
    if (features.eyebrowRaise === 1) {
      parts.push('[SURPRISE!]');
    } else {
      parts.push('[EXCLAMATION]');
    }
  }

  // Head tilt → Uncertainty, Curiosity
  if (features.headTilt !== 0) {
    parts.push(features.headTilt > 0 ? '[CURIOUS/UNSURE →]' : '[CURIOUS/UNSURE ←]');
  }

  // Head nod → Affirmation, Agreement
  if (features.headNod > 0) {
    parts.push('[YES/AGREE ↓]');
  } else if (features.headNod < 0) {
    parts.push('[THINKING ↑]');
  }

  // Head shake → Negation
  if (features.headShake !== 0) {
    parts.push('[NO/NEGATIVE ↔]');
  }

  // Shoulder lean → Emphasis, Direction
  if (features.shoulderLean !== 0) {
    parts.push(features.shoulderLean > 0 ? '[EMPHASIS →]' : '[EMPHASIS ←]');
  }

  // Combined patterns
  if (features.eyebrowRaise && features.headNod > 0) {
    parts.push('[RHETORICAL QUESTION?]');
  }

  if (features.headShake !== 0 && features.eyebrowRaise === 1) {
    parts.push('[CONFUSED/DISBELIEF]');
  }

  return parts.join(' ') || 'Neutral';
}

// ============================================================================
// Display Features
// ============================================================================

function displayFeatures() {
  const text = `
Eyebrow Raise:   ${currentFeatures.eyebrowRaise === 1 ? '✓ RAISED' : '  neutral'}
Mouth Open:      ${currentFeatures.mouthOpen === 1 ? '✓ OPEN' : '  closed'}
Head Tilt:       ${currentFeatures.headTilt > 0 ? '→ RIGHT' : currentFeatures.headTilt < 0 ? '← LEFT' : '  center'}
Head Nod:        ${currentFeatures.headNod > 0 ? '↓ DOWN' : currentFeatures.headNod < 0 ? '↑ UP' : '  neutral'}
Head Shake:      ${currentFeatures.headShake > 0 ? '→ RIGHT' : currentFeatures.headShake < 0 ? '← LEFT' : '  neutral'}
Shoulder Lean:   ${currentFeatures.shoulderLean > 0 ? '→ RIGHT' : currentFeatures.shoulderLean < 0 ? '← LEFT' : '  neutral'}
  `.trim();
  
  featuresDiv.textContent = text;
}

// ============================================================================
// Error Handling
// ============================================================================

window.addEventListener('error', (e) => {
  console.error('Error:', e.error);
  nmfTextDiv.textContent = 'Error: ' + e.error.message;
});

console.log('ISL NMF MVP initialized. Allow camera access to begin.');
