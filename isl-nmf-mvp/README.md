# ISL NMF MVP - Non-Manual Feature Detection

A minimal working prototype that captures facial expressions, head movements, and simple posture from a webcam and converts them to readable text interpretations for Indian Sign Language (ISL).

## 🚀 Quick Start

### Phase A: Run the MVP (Rule-based)

1. **Start a local server** (required for MediaPipe):
   ```bash
   cd isl-nmf-mvp/web
   python3 -m http.server 8000
   ```
   
2. **Open in browser**:
   - Navigate to: `http://localhost:8000`
   - Allow webcam access when prompted

3. **Test the features**:
   - Raise eyebrows → Question/Emphasis
   - Open mouth → Exclamation/Surprise
   - Tilt head → Curiosity/Uncertainty
   - Nod head up/down → Agreement/Thinking
   - Shake head left/right → Negation
   - Lean shoulders → Emphasis/Direction

## 📊 Features Detected

The system tracks 6 key non-manual features (NMFs):

| Feature | Detection Method | Interpretation |
|---------|------------------|----------------|
| **Eyebrow Raise** | Distance between eyebrow and eye landmarks | Question, Emphasis, Surprise |
| **Mouth Open** | Distance between upper and lower lip | Exclamation, Surprise |
| **Head Tilt** | Roll angle between ear landmarks | Curiosity, Uncertainty |
| **Head Nod** | Vertical movement of nose tip over time | Agreement (down), Thinking (up) |
| **Head Shake** | Horizontal rotation/movement | Negation, Disagreement |
| **Shoulder Lean** | Tilt angle between shoulder landmarks | Emphasis, Direction |

## 🎯 Architecture

### Current (Phase A): Rule-Based Mapping
- MediaPipe FaceMesh (478 landmarks)
- MediaPipe Pose (33 landmarks)
- Threshold-based feature extraction
- Simple if-then rules for text generation

### Upcoming (Phase B): ML Classifier
- Collect labeled dataset (CSV format)
- Train lightweight classifier (logistic regression/small neural net)
- Replace rule-based mapping with learned patterns

## 📁 Project Structure

```
isl-nmf-mvp/
├── web/
│   ├── index.html       # Main UI
│   ├── app.js           # MediaPipe + Feature extraction
│   └── styles.css       # Styling
├── dataset/
│   └── nmf_training.csv # Training data (Phase B)
└── train/
    └── train_classifier.py # Training script (Phase B)
```

## 🔧 Technical Details

### MediaPipe Landmarks Used

**Face (FaceMesh - 478 points):**
- Eyebrows: 70, 300 (left/right outer)
- Eyes: 159, 386 (left/right centers)
- Nose: 1 (tip), 168 (bridge)
- Mouth: 13 (upper lip), 14 (lower lip)
- Chin: 152
- Ears: 234, 454

**Pose (33 points):**
- Shoulders: 11 (left), 12 (right)

### Feature Thresholds

```javascript
eyebrowRaise: avgBrowDist > 0.04
mouthOpen: mouthHeight > 0.02
headTilt: |angle| > 0.15 radians (~8.6°)
headNod: |deltaY| > 0.03 (over 10 frames)
headShake: |horizontalOffset| > 0.05
shoulderLean: |tiltAngle| > 0.2 radians (~11.5°)
```

## 📈 Phase B: Adding ML Classifier

### 1. Create Dataset

Create `dataset/nmf_training.csv`:

```csv
eyebrow_raise,mouth_open,head_tilt,head_nod,head_shake,shoulder_lean,label
1,0,0,0,0,0,question
1,1,0,0,0,0,surprise
0,0,1,0,0,0,curiosity
0,0,0,1,0,0,agreement
0,0,0,0,1,0,negation
1,0,0,1,0,0,rhetorical_question
0,1,0,0,0,1,exclamation_emphasis
```

### 2. Collect Real Data

Modify `app.js` to log features:

```javascript
// Add data collection mode
let collectMode = false;
let currentLabel = '';

function collectSample(label) {
  const sample = { ...currentFeatures, label };
  console.log(JSON.stringify(sample));
  // Copy to dataset/nmf_training.csv
}
```

### 3. Train Classifier

See `train/train_classifier.py` for training script.

### 4. Export Model

Convert trained model to TensorFlow.js format and load in browser.

## 🎨 Customization

### Add New Features

```javascript
// In extractFeatures():
const newFeature = calculateNewFeature(landmarks);
currentFeatures.newFeature = newFeature > threshold ? 1 : 0;
```

### Modify Interpretations

```javascript
// In featuresToText():
if (features.newFeature === 1) {
  parts.push('[YOUR INTERPRETATION]');
}
```

### Adjust Thresholds

Tune thresholds based on your environment and expressions:
- Lighting affects detection
- Distance from camera matters
- Individual expression intensity varies

## 🐛 Troubleshooting

**Camera not working:**
- Use HTTPS or localhost
- Check browser permissions
- Try Chrome/Edge (best MediaPipe support)

**Laggy detection:**
- Reduce `modelComplexity` in Pose settings
- Disable refineLandmarks in FaceMesh
- Lower camera resolution

**False positives:**
- Increase thresholds
- Add temporal smoothing
- Use longer history windows

**No detections:**
- Ensure good lighting
- Face camera directly
- Check MediaPipe CDN is loading

## 📚 Resources

- [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [ISL Non-Manual Features](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8234567/)

## 🚧 Roadmap

- [x] Phase A: Rule-based NMF detection
- [ ] Phase B: ML classifier training
- [ ] Phase C: Regional ISL variants
- [ ] Phase D: Mobile app (TFLite)
- [ ] Phase E: Dataset augmentation
- [ ] Phase F: Real-time grammar integration

## 📄 License

MIT - Feel free to use for research and education.

---

**Built for accessible ISL communication** 🤟
