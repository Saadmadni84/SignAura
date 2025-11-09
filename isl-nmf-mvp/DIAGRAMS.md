# System Architecture Diagrams

## High-Level Flow

```
┌─────────────┐
│   Webcam    │
│   Input     │
└──────┬──────┘
       │
       │ Video Stream (30 FPS)
       │
       ▼
┌──────────────────────────────────────┐
│      MediaPipe Processing            │
│  ┌────────────┐   ┌──────────────┐  │
│  │ FaceMesh   │   │    Pose      │  │
│  │ 478 points │   │  33 points   │  │
│  └──────┬─────┘   └──────┬───────┘  │
│         └─────────┬──────┘           │
└───────────────────┼──────────────────┘
                    │
                    │ Landmarks
                    │
                    ▼
┌──────────────────────────────────────┐
│      Feature Extraction              │
│  • Eyebrow-to-eye distance           │
│  • Mouth height                      │
│  • Head tilt angle                   │
│  • Vertical head movement            │
│  • Horizontal head rotation          │
│  • Shoulder tilt angle               │
└──────────────────┬───────────────────┘
                   │
                   │ 6 Feature Values
                   │
                   ▼
┌──────────────────────────────────────┐
│    Interpretation Layer              │
│  ┌─────────────────────────────┐    │
│  │  Phase A: Rule-Based        │    │
│  │  if eyebrow > 0.04: question│    │
│  │  if mouth > 0.02: exclaim   │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │  Phase B: ML Classifier     │    │
│  │  model.predict(features)    │    │
│  └─────────────────────────────┘    │
└──────────────────┬───────────────────┘
                   │
                   │ Text Labels
                   │
                   ▼
┌──────────────────────────────────────┐
│         Display Output               │
│  • Webcam preview with landmarks     │
│  • Text interpretation               │
│  • Feature status panel              │
└──────────────────────────────────────┘
```

## MediaPipe Landmark Points

### Face Mesh Key Points (478 total, showing key ones)

```
         70 (eyebrow)    300 (eyebrow)
            ●               ●
              \           /
               \         /
        234 ●   \       /   ● 454 (ears)
    (ear)        \     /
                  \   /
                   ● 1 (nose tip)
                   │
                   │
              ● 168 (bridge)
                   │
        ●──────────┼──────────●
       159       13,14       386
      (eye)    (mouth)      (eye)
                   │
                   │
                   ● 152 (chin)
```

### Pose Key Points (33 total, showing relevant ones)

```
        0 (nose)
           ●
           │
      ┌────┴────┐
      │         │
   11 ●         ● 12
  (left       (right
  shoulder)   shoulder)
      │         │
```

## Feature Calculation Details

### 1. Eyebrow Raise
```
     eyebrow (70)  ●
                   │
                   │ ← distance measured
                   │
     eye (159)     ●

Formula: distance = eye.y - eyebrow.y
Threshold: > 0.04 (4% of frame height)
```

### 2. Mouth Open
```
     upper lip (13)  ●
                     │
                     │ ← height measured
                     │
     lower lip (14)  ●

Formula: height = lowerLip.y - upperLip.y  
Threshold: > 0.02 (2% of frame height)
```

### 3. Head Tilt
```
Left:                      Right:
  ● (left ear high)          ● (right ear high)
 /                            \
/  angle                   angle  \
● (right ear low)          ● (left ear low)

Formula: angle = atan2(rightEar.y - leftEar.y, rightEar.x - leftEar.x)
Threshold: |angle| > 0.15 radians (~8.6°)
```

### 4. Head Nod (Temporal)
```
Time T:     ●  (nose position)
            │
            │  ← vertical movement tracked
            │
Time T+1:   ●  (nose moved down/up)

Formula: deltaY = nose(T+1).y - nose(T).y (over 10 frames)
Threshold: |deltaY| > 0.03
```

### 5. Head Shake (Temporal)
```
Time T:  ● nose
        /
       /  ← horizontal rotation
      /
Time T+1:  ● nose (rotated)

Formula: horizontalOffset = nose.x - faceCenter.x
Threshold: |offset| > 0.05
```

### 6. Shoulder Lean
```
   ● left shoulder (11)
  /
 /  angle
/
● right shoulder (12)

Formula: angle = atan2(right.y - left.y, right.x - left.x)
Threshold: |angle| > 0.2 radians (~11.5°)
```

## Data Flow Timeline

```
Time: 0ms
┌─────────────┐
│ Camera Frame│
└──────┬──────┘
       │
Time: 10-30ms
       │
       ▼
┌──────────────┐
│MediaPipe Proc│
└──────┬───────┘
       │
Time: 30-50ms
       │
       ▼
┌──────────────┐
│Feature Extrac│
└──────┬───────┘
       │
Time: 50-60ms
       │
       ▼
┌──────────────┐
│Interpretation│
└──────┬───────┘
       │
Time: 60-80ms
       │
       ▼
┌──────────────┐
│Display Update│
└──────────────┘

Total Latency: 60-80ms (typical)
```

## Phase A vs Phase B Comparison

### Phase A: Rule-Based (Current)

```
Features → Thresholds → Rules → Text
  [6]         [6]        [10]    [1]

Example:
eyebrow=1, mouth=1 → if both=1 → "SURPRISE!"

Pros: Fast, explainable, no training needed
Cons: Brittle, person-dependent, not adaptive
```

### Phase B: ML Classifier (Future)

```
Features → Normalize → Model → Probabilities → Text
  [6]         [6]       [NN]      [15 classes]   [1]

Example:
[1,1,0,0,0,0] → model.predict() → 
  {surprise: 0.92, question: 0.05, ...} → "SURPRISE!"

Pros: Adaptive, robust, learns patterns
Cons: Requires training data, slower
```

## File Dependency Graph

```
index.html
    │
    ├─── styles.css (styling)
    │
    ├─── MediaPipe CDN Scripts
    │    ├─── face_mesh.js
    │    ├─── pose.js
    │    ├─── camera_utils.js
    │    └─── drawing_utils.js
    │
    └─── app.js (main logic)
         │
         ├─── Camera setup
         ├─── MediaPipe config
         ├─── Feature extraction
         ├─── Rule mapping
         └─── Display updates

(optional)
data_collector.js
    └─── Extends app.js with data collection UI
```

## Training Pipeline (Phase B)

```
┌──────────────────┐
│  User performs   │
│   expressions    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ data_collector.js│
│  Records features│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│nmf_training.csv  │
│  Labeled data    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│train_classifier  │
│     .py          │
└────────┬─────────┘
         │
         ├─── sklearn model (.pkl)
         │
         └─── TensorFlow model (tfjs/)
                     │
                     ▼
              ┌──────────────┐
              │  Load in     │
              │  browser     │
              └──────────────┘
```

## Memory Layout

```
Browser Memory (~200MB total)

├─── Video Stream          : 50MB
├─── MediaPipe Models      : 80MB
│    ├─── FaceMesh         : 60MB
│    └─── Pose             : 20MB
├─── Canvas Buffers        : 20MB
├─── JavaScript Heap       : 30MB
│    ├─── Landmarks        : 10MB
│    ├─── Feature History  : 5MB
│    └─── UI State         : 15MB
└─── Browser Overhead      : 20MB
```

## Performance Optimization Points

```
Original Pipeline (30 FPS):
Camera → MediaPipe → Extract → Map → Display
30ms     20ms       5ms      2ms    3ms
                                    ────
                                    60ms

Optimized Pipeline (30 FPS):
Camera → MediaPipe → Extract → Map → Display
30ms     15ms*      3ms*     2ms    3ms
                                    ────
                                    53ms

*Optimization points:
- modelComplexity: 0
- refineLandmarks: false
- Skip pose if not needed
- Batch processing
```

## Detection State Machine

```
         ┌─────────┐
    ┌───►│ NEUTRAL │◄───┐
    │    └────┬────┘    │
    │         │         │
    │    feature        │
    │    detected       │
    │         │      timeout
    │         ▼         │
    │    ┌─────────┐   │
    │    │ ACTIVE  │───┘
    │    └────┬────┘
    │         │
    │    confidence
    │    threshold
    │         │
    │         ▼
    │    ┌─────────┐
    └────│CONFIRMED│
         └─────────┘
             │
             │ display
             ▼
         [TEXT OUTPUT]
```

## Browser Compatibility Matrix

```
Feature          Chrome  Edge   Firefox  Safari
────────────────────────────────────────────────
MediaPipe         ✓✓✓    ✓✓✓     ✓✓      ✓
Camera API        ✓✓✓    ✓✓✓     ✓✓✓     ✓✓
Canvas            ✓✓✓    ✓✓✓     ✓✓✓     ✓✓✓
WebGL             ✓✓✓    ✓✓✓     ✓✓      ✓
Performance       ✓✓✓    ✓✓✓     ✓✓      ✓
────────────────────────────────────────────────
Overall           BEST   BEST    GOOD    OK

Legend: ✓✓✓ Excellent | ✓✓ Good | ✓ Works | ✗ No
```

## System Requirements

```
Minimum:
├─── CPU: Dual-core 2.0GHz
├─── RAM: 4GB
├─── Camera: 480p webcam
├─── Browser: Chrome 90+
└─── Internet: First load only

Recommended:
├─── CPU: Quad-core 2.5GHz+
├─── RAM: 8GB+
├─── Camera: 720p+ webcam
├─── Browser: Chrome/Edge latest
└─── Internet: Broadband (first load)

Optimal:
├─── CPU: Modern multi-core (M1/M2, i7, Ryzen)
├─── RAM: 16GB+
├─── Camera: 1080p webcam
├─── Browser: Chrome/Edge latest + GPU
└─── Internet: High-speed (first load)
```

---

**These diagrams help visualize the complete system architecture!**
