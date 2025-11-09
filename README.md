# 🧠 SignAura: Capturing Non-Manual Features of Indian Sign Language (ISL) and Converting to Text

### 🌐 Empowering Inclusive Communication Through AI Vision

**SignAura** is an intelligent system that captures and interprets **Non-Manual Features (NMFs)** of *Indian Sign Language (ISL)* — such as **facial expressions, head movements, and body posture** — and converts them into **contextually meaningful text** in real time.  

Non-manual features are essential for understanding tone, emotion, and grammar in ISL. **SignAura** bridges this gap using **computer vision** and **machine learning**, enhancing the translation accuracy and inclusivity of sign language communication.

---

## 🚀 Key Features

- 🎥 **Real-Time Detection:** Captures facial and body landmarks using **MediaPipe** or **TensorFlow.js**.
- 🤖 **Feature Extraction Engine:** Calculates metrics like eyebrow raise, mouth openness, head tilt, nod, and torso lean.
- 💬 **Text Translation Logic:** Converts detected NMFs into contextual text (e.g., “question”, “affirmative”, “surprise”).
- 🧠 **Lightweight ML Model:** Optional classifier (Logistic Regression / TensorFlow.js) trained on labeled NMF datasets.
- 💻 **Modern UI:** Clean dashboard with live video feed, animated text output, and visual feature indicators.
- 🔒 **Privacy-Friendly:** Runs entirely in the browser — no video data leaves your device.
- 🌍 **ISL Region Aware:** Extendable to support regional ISL variations.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6), Tailwind CSS |
| **Computer Vision** | MediaPipe (FaceMesh + Pose) |
| **ML / AI** | scikit-learn, TensorFlow.js |
| **Visualization** | Canvas API, Tailwind UI Components |
| **Optional Tools** | Electron.js (offline desktop), Framer Motion (animations) |

---

## ⚙️ How It Works

1. **Webcam Capture**  
   → Streams live video input from the user.

2. **Feature Extraction**  
   → MediaPipe detects facial and body landmarks.  
   → The app computes normalized ratios (eye openness, eyebrow raise, mouth open, head roll, nod, torso lean).

3. **Rule-Based or ML Mapping**  
   → Extracted features are mapped to contextual meanings (e.g., “affirmative”, “question”, “neutral”).

4. **Text Translation Display**  
   → Interpreted meaning is displayed live and logged as a transcript.

5. **Dataset Collection (Optional)**  
   → Record samples and train your own model using scikit-learn or TensorFlow.js.

---

## 🧪 Quick Start

### 🖥️ Run Locally
```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/SignAura.git

# 2. Navigate to the web folder
cd SignAura/web

# 3. Start a local development server
npx http-server

# 4. Open your browser
http://localhost:8080

📊 Dataset & Model 
Record labeled NMF samples and train your own model:
cd train
python train_classifier.py

This script produces model.joblib.
You can convert it to JSON for use directly in the browser for real-time predictions.
🧠 Future Enhancements
🧩 Combine manual and non-manual ISL features for complete translation
🧠 Integrate temporal modeling (e.g., RNN or LSTM) for gesture sequences
💬 Real-time ISL → Text → Speech conversion
📱 Build as a PWA or Electron app for offline use
🌍 Regional ISL NMF variations support

🧩 Architecture Overview
Webcam 
   ↓
MediaPipe (FaceMesh + Pose)
   ↓
Feature Extractor
   ↓
Rule/Model Mapper
   ↓
Text Translator
   ↓
UI Display + Transcript Log

🛡️ Privacy & Ethics
All computations run locally on the user’s device.
No video or personal data is uploaded to any server.
Designed to assist communication, not replace professional ISL interpreters.
This project aims to improve accessibility, inclusion, and research in Indian Sign Language technology.
🏛️ Developed Under
Indian Sign Language Research and Training Centre (ISLRTC)
Department of Empowerment of Persons with Disabilities (DePWD)
Ministry of Social Justice and Empowerment, Government of India

🤝 Contributors
Saad Madni — Developer & Researcher

🌟 License
MIT License — Free for educational, academic, and research use.
