# 🎭 Testing & Demo Guide

This guide helps you test the ISL NMF MVP and prepare for demos.

## Quick Test Checklist

### ✅ Basic Functionality Test (5 minutes)

1. **Start the server**
   ```bash
   cd web
   python3 -m http.server 8000
   ```

2. **Open browser**: http://localhost:8000

3. **Test each feature individually:**

   - [ ] **Neutral face** → Should show "Neutral"
   - [ ] **Raise eyebrows** → `[QUESTION/EMPHASIS]`
   - [ ] **Open mouth** → `[EXCLAMATION]`
   - [ ] **Raise eyebrows + open mouth** → `[SURPRISE!]`
   - [ ] **Tilt head right** → `[CURIOUS/UNSURE →]`
   - [ ] **Tilt head left** → `[CURIOUS/UNSURE ←]`
   - [ ] **Nod head down** → `[YES/AGREE ↓]`
   - [ ] **Nod head up** → `[THINKING ↑]`
   - [ ] **Shake head left-right** → `[NO/NEGATIVE ↔]`
   - [ ] **Lean shoulders right** → `[EMPHASIS →]`
   - [ ] **Lean shoulders left** → `[EMPHASIS ←]`

4. **Test combinations:**

   - [ ] **Eyebrows + nod** → `[RHETORICAL QUESTION?]`
   - [ ] **Head shake + eyebrows** → `[CONFUSED/DISBELIEF]`

5. **Check visualizations:**

   - [ ] Green dots visible on face
   - [ ] Feature values update in real-time
   - [ ] Text changes match your expressions

### 🎯 Demo Scenarios

#### Scenario 1: Question (Yes/No)
```
Action: Raise eyebrows
Expected: [QUESTION/EMPHASIS]
ISL Context: "Are you coming?" (with manual sign)
```

#### Scenario 2: Strong Affirmation
```
Action: Nod head down while raising eyebrows
Expected: [YES/AGREE ↓] [QUESTION/EMPHASIS] → [RHETORICAL QUESTION?]
ISL Context: "Of course!" / "Definitely yes"
```

#### Scenario 3: Surprise
```
Action: Raise eyebrows + open mouth
Expected: [SURPRISE!]
ISL Context: "Really?!" / "I can't believe it!"
```

#### Scenario 4: Negation
```
Action: Shake head left-right
Expected: [NO/NEGATIVE ↔]
ISL Context: "No" / "I disagree" / "Not happening"
```

#### Scenario 5: Confusion
```
Action: Shake head + raise eyebrows
Expected: [NO/NEGATIVE ↔] [QUESTION/EMPHASIS] → [CONFUSED/DISBELIEF]
ISL Context: "What?!" / "I don't understand" / "Are you sure?"
```

#### Scenario 6: Emphasis
```
Action: Open mouth + lean shoulder right
Expected: [EXCLAMATION] [EMPHASIS →]
ISL Context: "Over there!" / "That direction!" (with pointing)
```

## 🔍 Calibration Guide

Different people express differently. Here's how to calibrate:

### If detection is TOO SENSITIVE:

**Edit app.js, increase thresholds:**

```javascript
// Eyebrow (line ~107)
currentFeatures.eyebrowRaise = avgBrowDist > 0.05 ? 1 : 0;  // was 0.04

// Mouth (line ~114)
currentFeatures.mouthOpen = mouthHeight > 0.03 ? 1 : 0;  // was 0.02

// Head tilt (line ~121)
currentFeatures.headTilt = Math.abs(tiltAngle) > 0.20 ? ... // was 0.15

// Head nod (line ~137)
currentFeatures.headNod = Math.abs(deltaY) > 0.04 ? ... // was 0.03

// Head shake (line ~148)
currentFeatures.headShake = Math.abs(horizontalOffset) > 0.07 ? ... // was 0.05

// Shoulder (line ~158)
currentFeatures.shoulderLean = Math.abs(shoulderTilt) > 0.25 ? ... // was 0.2
```

### If detection is NOT SENSITIVE ENOUGH:

**Decrease thresholds (opposite of above)**

### Recommended values by expression intensity:

| Expression Type | Subtle | Normal | Exaggerated |
|----------------|--------|--------|-------------|
| Eyebrow | 0.05 | 0.04 | 0.03 |
| Mouth | 0.03 | 0.02 | 0.015 |
| Head Tilt | 0.20 | 0.15 | 0.10 |
| Head Nod | 0.04 | 0.03 | 0.02 |
| Head Shake | 0.07 | 0.05 | 0.03 |
| Shoulder | 0.25 | 0.20 | 0.15 |

## 📊 Performance Testing

### Frame Rate Test

Open browser console (F12) and run:

```javascript
let frameCount = 0;
let lastTime = Date.now();

setInterval(() => {
  const now = Date.now();
  const fps = frameCount / ((now - lastTime) / 1000);
  console.log(`FPS: ${fps.toFixed(1)}`);
  frameCount = 0;
  lastTime = now;
}, 1000);

// Add to processFrame()
frameCount++;
```

**Target**: 20-30 FPS
**Minimum acceptable**: 15 FPS

### Latency Test

Measure time from expression to detection:

```javascript
// Add to app.js
let expressionStartTime = null;

// When you start an expression, run in console:
expressionStartTime = Date.now();

// In featuresToText(), add:
if (expressionStartTime && parts.length > 0) {
  console.log(`Latency: ${Date.now() - expressionStartTime}ms`);
  expressionStartTime = null;
}
```

**Target**: < 200ms
**Acceptable**: < 500ms

## 🎥 Recording a Demo

### Preparation

1. **Lighting**: Set up good front lighting
2. **Background**: Clean, uncluttered background
3. **Camera**: Position at eye level, 1-2 feet away
4. **Screen**: Share browser window in recordings

### Demo Script (2-3 minutes)

**Opening (15 seconds)**
```
"This is an ISL Non-Manual Feature detector that captures facial expressions 
and head movements in real-time and interprets their meaning in sign language context."
```

**Demonstration (90 seconds)**
1. Show neutral state
2. Demonstrate each feature:
   - "A raised eyebrow indicates a question or emphasis"
   - "An open mouth shows exclamation or strong emotion"
   - "A head nod means agreement or affirmation"
   - "A head shake means negation or disagreement"
3. Show combinations:
   - "Combining eyebrow raise with a nod creates a rhetorical question"
   - "Head shake with raised eyebrows shows confusion or disbelief"

**Technical Demo (30 seconds)**
4. Show the feature panel updating in real-time
5. Explain landmarks visualization

**Closing (15 seconds)**
```
"This MVP uses MediaPipe for real-time detection and can be extended with 
machine learning for more accurate interpretations. All processing happens 
in the browser - no cloud services needed."
```

## 🐛 Common Issues & Solutions

### Issue: "Jittery" detections

**Symptoms**: Features flicker on/off rapidly

**Solutions**:
1. Add temporal smoothing (average over last N frames)
2. Require feature to persist for minimum duration
3. Add hysteresis (different thresholds for on/off)

```javascript
// Example: Temporal smoothing
const featureHistory = [];
const SMOOTHING_WINDOW = 5;

function smoothFeature(value) {
  featureHistory.push(value);
  if (featureHistory.length > SMOOTHING_WINDOW) {
    featureHistory.shift();
  }
  const avg = featureHistory.reduce((a, b) => a + b) / featureHistory.length;
  return avg > 0.5 ? 1 : 0;
}
```

### Issue: Landmarks not visible

**Symptoms**: No green dots on face

**Check**:
1. Camera permissions granted?
2. MediaPipe models loaded? (Check network tab)
3. Face in frame and well-lit?
4. Console errors?

### Issue: Wrong interpretations

**Symptoms**: Detections don't match expressions

**Solutions**:
1. Recalibrate thresholds for your face
2. Check lighting (shadows affect landmarks)
3. Ensure face is perpendicular to camera
4. Collect data and train ML model (Phase B)

### Issue: High CPU usage / hot laptop

**Solutions**:
1. Reduce model complexity (see QUICKSTART.md)
2. Lower camera resolution
3. Reduce frame rate
4. Disable pose detection if not needed:

```javascript
// Comment out pose in app.js
// await pose.send({ image: video });
```

## 📈 Data Collection Best Practices

### For Training ML Model (Phase B)

**Sample Distribution**:
- 30% neutral/baseline
- 40% single features
- 30% combinations

**Per Expression**:
- Minimum: 20 samples
- Recommended: 50 samples
- Optimal: 100+ samples

**Variation**:
- Different intensities (subtle, normal, exaggerated)
- Different head positions (slight angles)
- Different lighting
- Different times of day
- Multiple people if possible

**Labeling**:
- Be consistent
- Use lowercase with underscores: `question`, `strong_negation`
- Avoid ambiguous labels
- Document label meanings

### Sample Collection Session (30 minutes)

1. **Neutral** (5 min): 50+ samples, relaxed face
2. **Eyebrow raise** (3 min): 30 samples, varying intensity
3. **Mouth open** (3 min): 30 samples
4. **Head tilt** (3 min): 15 left, 15 right
5. **Head nod** (3 min): 15 down, 15 up
6. **Head shake** (3 min): 30 samples
7. **Shoulder lean** (3 min): 15 left, 15 right
8. **Combinations** (7 min): 
   - Eyebrow + mouth: 20 samples
   - Eyebrow + nod: 20 samples
   - Shake + eyebrow: 20 samples

## 🎓 Understanding Accuracy

### What "Good" Looks Like

**Excellent** (>90% accuracy):
- Detects obvious expressions reliably
- Minimal false positives on neutral
- Combinations detected correctly

**Good** (75-90%):
- Most expressions detected
- Occasional false positives
- Combinations sometimes missed

**Needs Work** (<75%):
- Frequent misdetections
- Many false positives
- Unreliable overall

### Improving Accuracy

1. **Better data**: More samples, more variety
2. **Better features**: Add eye gaze, blink rate
3. **Better model**: Neural network vs rules
4. **Better preprocessing**: Normalize lighting, distance
5. **Better thresholds**: Calibrate per-user

## 🚀 Next Steps After Testing

1. **Collect 200+ labeled samples**
2. **Train ML classifier** (train_classifier.py)
3. **Integrate with manual signs**
4. **Add temporal grammar** (question at start vs. end)
5. **Build mobile app** (TFLite export)
6. **User testing** with ISL users
7. **Regional variants** (different ISL dialects)

---

**Good luck with your demo! 🤟**
