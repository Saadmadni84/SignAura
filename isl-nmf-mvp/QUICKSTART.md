# 🚀 Quick Start Guide - ISL NMF MVP

Get your webcam-based non-manual feature detector running in **5 minutes**!

## Prerequisites

- A laptop/computer with a webcam
- Modern web browser (Chrome, Edge, Firefox recommended)
- Python 3 installed (for local server)

## Steps to Run

### 1. Navigate to the project
```bash
cd "/Users/saadmadni/Downloads/3rd year/project/SignAura/isl-nmf-mvp/web"
```

### 2. Start a local web server

**Option A: Python (recommended)**
```bash
python3 -m http.server 8000
```

**Option B: Node.js (if you have it)**
```bash
npx http-server -p 8000
```

**Option C: PHP**
```bash
php -S localhost:8000
```

### 3. Open in browser

Navigate to: **http://localhost:8000**

### 4. Allow camera access

When prompted, click "Allow" to grant webcam permissions.

### 5. Test it!

Try these expressions and movements:

| Action | Expected Detection |
|--------|-------------------|
| Raise eyebrows | `[QUESTION/EMPHASIS]` |
| Open mouth wide | `[EXCLAMATION]` |
| Raise eyebrows + open mouth | `[SURPRISE!]` |
| Tilt head left/right | `[CURIOUS/UNSURE]` |
| Nod head down | `[YES/AGREE ↓]` |
| Shake head left-right | `[NO/NEGATIVE ↔]` |
| Lean shoulders | `[EMPHASIS]` |
| Eyebrows + nod | `[RHETORICAL QUESTION?]` |

## 🎥 What You Should See

1. **Left panel**: Live webcam feed with green dots (face landmarks)
2. **Right panel**: 
   - Top: Interpreted text (e.g., "[QUESTION/EMPHASIS]")
   - Bottom: Raw feature values (e.g., "Eyebrow Raise: ✓ RAISED")

## 🐛 Troubleshooting

### Camera not working
- ✅ Make sure you're using `localhost` or `127.0.0.1` (required for camera access)
- ✅ Check browser permissions (click lock icon in address bar)
- ✅ Try Chrome or Edge (best MediaPipe support)
- ✅ Close other apps using the webcam (Zoom, Teams, etc.)

### Page won't load
- ✅ Check server is running (terminal should show "Serving HTTP")
- ✅ Try a different port: `python3 -m http.server 8080`
- ✅ Check firewall isn't blocking

### Detection too sensitive/not sensitive enough
Edit `app.js` and adjust thresholds:

```javascript
// Line ~107 - Eyebrow raise threshold
currentFeatures.eyebrowRaise = avgBrowDist > 0.04 ? 1 : 0;
// Try 0.05 for less sensitive, 0.03 for more sensitive

// Line ~114 - Mouth open threshold  
currentFeatures.mouthOpen = mouthHeight > 0.02 ? 1 : 0;
// Try 0.03 for less sensitive, 0.015 for more sensitive
```

### Landmarks not showing
- ✅ Ensure good lighting (face the light source)
- ✅ Keep face 1-2 feet from camera
- ✅ Face camera directly
- ✅ Remove glasses if possible
- ✅ Check browser console (F12) for errors

### Slow/laggy
Edit `app.js` to reduce quality:

```javascript
// Line ~50 - Reduce face mesh quality
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: false,  // Change to false
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// Line ~62 - Reduce pose quality
pose.setOptions({
  modelComplexity: 0,  // Change from 1 to 0
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
```

## 📊 Phase B: Collect Your Own Data

Want to train your own classifier? Follow these steps:

### 1. Enable data collection

Add to `index.html` before closing `</body>`:
```html
<script src="data_collector.js"></script>
<script>
  window.addEventListener('load', addDataCollectionUI);
</script>
```

Then modify `app.js`, add this line in `processFrame()` function (around line 100):
```javascript
function processFrame() {
  if (!faceResults) return;
  // ... existing code ...
  collectSample(); // ADD THIS LINE
}
```

### 2. Collect samples

1. Reload the page
2. Enter a label (e.g., "question")
3. Click "Start Collecting"
4. Perform the expression for 3-5 seconds
5. Click "Stop Collecting"
6. Repeat for different expressions (aim for 20-50 samples each)
7. Click "Download CSV"

### 3. Train classifier

```bash
cd ../train
pip install pandas scikit-learn tensorflow numpy
python train_classifier.py
```

### 4. Integrate trained model

See `train/train_classifier.py` for instructions on exporting to TensorFlow.js.

## 🎯 Tips for Best Results

### Lighting
- Face a window or light source
- Avoid backlighting (don't sit with light behind you)
- Even, diffused lighting works best

### Distance
- 1-2 feet from camera optimal
- Too close: landmarks might go out of frame
- Too far: detection becomes less accurate

### Expressions
- Make expressions slightly exaggerated
- Hold expressions for 1-2 seconds
- Move smoothly (not jerky)

### Data Collection
- Collect in different conditions (lighting, position)
- Include "neutral" baseline samples
- Get multiple people to perform expressions
- Label consistently

## 📈 Next Steps

1. ✅ Get basic MVP running (you are here!)
2. ⬜ Collect labeled data for your specific use case
3. ⬜ Train ML classifier (Phase B)
4. ⬜ Integrate with sign language vocabulary
5. ⬜ Add temporal grammar rules
6. ⬜ Export to mobile (TFLite)
7. ⬜ Add ISL regional variants

## 📚 Understanding the Output

### Text Interpretations

| Output | Meaning | ISL Context |
|--------|---------|-------------|
| `[QUESTION/EMPHASIS]` | Eyebrow raise | Yes/no questions, emphasis |
| `[SURPRISE!]` | Eyebrows + mouth | Unexpected information |
| `[EXCLAMATION]` | Open mouth | Strong emotion, loudness |
| `[CURIOUS/UNSURE]` | Head tilt | Uncertainty, "what?" |
| `[YES/AGREE ↓]` | Head nod down | Affirmation, agreement |
| `[NO/NEGATIVE ↔]` | Head shake | Negation, disagreement |
| `[EMPHASIS →/←]` | Shoulder lean | Directional emphasis |
| `[RHETORICAL QUESTION?]` | Eyebrows + nod | Question expecting agreement |
| `[CONFUSED/DISBELIEF]` | Head shake + eyebrows | "Really?", doubt |

### Feature Values

```
Eyebrow Raise:   ✓ RAISED / neutral
Mouth Open:      ✓ OPEN / closed
Head Tilt:       → RIGHT / ← LEFT / center
Head Nod:        ↓ DOWN / ↑ UP / neutral
Head Shake:      → RIGHT / ← LEFT / neutral
Shoulder Lean:   → RIGHT / ← LEFT / neutral
```

## 🔬 Technical Deep Dive

For understanding the code:
- `index.html` - UI structure
- `app.js` - Main logic (MediaPipe, feature extraction, mapping)
- `styles.css` - Styling
- `data_collector.js` - Optional data collection helper

Key functions in `app.js`:
- `extractFeatures()` - Computes NMF values from landmarks
- `featuresToText()` - Maps features to text interpretations
- `processFrame()` - Main processing loop

## 💡 Ideas to Extend

1. **Add more features**: Eye gaze direction, tongue position
2. **Temporal patterns**: Detect sequences (e.g., repeated nods)
3. **Intensity levels**: Not just on/off, but slight/moderate/strong
4. **Context awareness**: Combine with manual signs
5. **Regional variants**: Different NMF meanings in regional ISL
6. **Audio feedback**: Speak detected interpretations
7. **Recording**: Save sessions for later analysis

## 📞 Need Help?

Common MediaPipe errors:
- "Could not load model" → Check internet connection (CDN loads)
- "Camera not found" → Check permissions, close other apps
- "Failed to fetch" → Use localhost, not file:// protocol

---

**Happy coding! 🤟 Built for accessible ISL communication.**
