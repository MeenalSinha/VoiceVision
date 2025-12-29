# 👁️ VoiceVision – AI Accessibility Assistant 🇮🇳

<div align="center">

![VoiceVision Logo](https://img.shields.io/badge/VoiceVision-AI%20Accessibility-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Designed to empower visually impaired individuals in India**

[Features](#-key-features) • [Demo](#-demo-guide) • [Installation](#-installation) • [Tech Stack](#-tech-stack) • [Roadmap](#-future-scope)

</div>

---

## 🎯 Problem Statement

Visually impaired individuals in India face daily challenges that limit their independence:

- 💰 **Currency Identification** – Difficulty identifying ₹10 to ₹2000 notes independently
- 📝 **Reading Signage** – Hindi and regional language signs are inaccessible
- 👁️ **Spatial Awareness** – Difficulty detecting obstacles and objects nearby
- 🧭 **Indoor Navigation** – Challenges moving safely in unfamiliar environments

**The Gap:** Most existing solutions are either not optimized for Indian contexts (currency, languages, environments) or lack transparency about AI limitations, creating trust and safety concerns.

---

## 💡 Solution Overview

VoiceVision is an **AI-powered accessibility suite** that provides **four core assistance modes** through a single, easy-to-use interface:

> **Note:** Currency detection operates in demo mode using OCR pattern matching. See [Model Status](#-system-performance) for full transparency on implementation details.

<table>
<tr>
<td width="50%">

### 💰 Currency Reader (Demo)
- Detects ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000 notes
- OCR-based denomination recognition
- Audio announcement of detected amount
- Confidence scoring for reliability

</td>
<td width="50%">

### 📝 Multilingual Text Reader
- English + Hindi OCR support
- Reads signboards, documents, labels
- On-capture text extraction
- Spoken audio output

</td>
</tr>
<tr>
<td width="50%">

### 👁️ Object Awareness
- Detects obstacles and objects
- Spatial position classification (left/center/right)
- Visual feedback with bounding boxes
- Relative spatial positioning

</td>
<td width="50%">

### 🧭 Navigation Assistance
- Directional guidance (move left/right)
- Obstacle avoidance cues
- Path clearance detection
- Audio-based navigation

</td>
</tr>
</table>

---

## ✨ Key Features

### 🇮🇳 Indian Context Optimization
- **Currency Recognition** – All Indian denominations from ₹10 to ₹2000
- **Multilingual Support** – Hindi + English text recognition
- **Local Environments** – Designed for Indian indoor/outdoor conditions

### ♿ Accessibility First
- **🔊 Audio Feedback** – Spoken guidance for all detections
- **⌨️ Keyboard Navigation** – Fully keyboard-accessible interface
- **📱 Multiple Input Methods** – Camera capture or file upload
- **🔇 Silent Mode** – Toggle audio on/off as needed

### 🛡️ Responsible AI
- **⚠️ Confidence Thresholds** – Warns when AI is uncertain
- **📊 Transparent Metrics** – Shows processing time and confidence
- **🚨 Safety Disclaimers** – Clear prototype limitations
- **🔍 Model Status Indicators** – Honest about demo vs production features

### 🎨 Premium User Experience
- **Glassmorphism UI** – Modern, visually appealing design
- **Smooth Animations** – Engaging hover effects and transitions
- **Responsive Layout** – Works on various screen sizes
- **Intuitive Navigation** – Clean, organized interface

---

## 🛠️ Tech Stack

### Core Technologies
```
🐍 Python 3.8+          📦 Streamlit          🖼️ OpenCV
🤖 EasyOCR             🗣️ gTTS              📊 NumPy
🎨 Pillow              🧠 PyTorch           🔧 scipy
```

### AI/ML Components
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **OCR Engine** | EasyOCR | Multilingual text recognition |
| **Obstacle Detection** | OpenCV (Edge-based) | Spatial awareness |
| **Currency Detection** | OCR Pattern Matching | Denomination identification |
| **Text-to-Speech** | gTTS | Audio feedback generation |

### Architecture
```
┌─────────────────────────────────────────┐
│         Streamlit Web Interface         │
├─────────────────────────────────────────┤
│  Camera Input  │  Image Upload  │ Audio │
├─────────────────────────────────────────┤
│  EasyOCR  │  OpenCV  │  gTTS  │  NumPy │
├─────────────────────────────────────────┤
│      Processing Pipeline & Logic        │
└─────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for camera capture mode)
- Internet connection (for TTS and model downloads)

### Quick Start

1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/voicevision.git
cd voicevision
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Run the application**
```bash
streamlit run app.py
```

4️⃣ **Open in browser**
```
http://localhost:8501
```

---

## 🎬 Demo Guide (For Judges & Reviewers)

### 💰 Currency Reader Mode
1. Select **💰 Currency Reader** from sidebar
2. Choose **📷 Camera** or **📁 Upload**
3. Show any Indian currency note (₹10 to ₹2000)
4. Click **🔍 Detect Denomination**
5. Listen to audio announcement: *"Detected: 500 rupees note"*

**Expected Result:** Audio announcement + confidence score + processing time

### 📝 Text Reader Mode
1. Select **📝 Text Reader (OCR)** from sidebar
2. Show printed text (English or Hindi signboard)
3. Click **📖 Extract & Read Text**
4. System reads text aloud automatically

**Expected Result:** Extracted text displayed + audio playback

### 👁️ Object Awareness Mode
1. Select **👁️ Object Awareness** from sidebar
2. Point camera at surroundings (desk, room, etc.)
3. Click **🔍 Detect Objects**
4. See bounding boxes + position labels

**Expected Result:** Visual overlay + audio description of obstacles

### 🧭 Navigation Assist Mode
1. Select **🧭 Navigation Assist** from sidebar
2. Capture image of path ahead
3. Click **🧭 Get Navigation Guidance**
4. Receive directional cue: *"Obstacle on left. Move right."*

**Expected Result:** Directional guidance + audio instruction

---

## 📊 System Performance

> **Note:** The following metrics are indicative values observed during local testing under good lighting conditions and typical usage scenarios.

| Metric | Value | Notes |
|--------|-------|-------|
| **OCR Accuracy** | ~85-90% | Good lighting, clear text |
| **Processing Time** | 200-800ms | Varies by image complexity |
| **Supported Languages** | 2 | English + Hindi |
| **Currency Notes** | 7 | ₹10 to ₹2000 |
| **Audio Latency** | <1s | gTTS generation time |

### Model Status

| Feature | Status | Implementation |
|---------|--------|----------------|
| ✅ OCR Engine | **Active** | EasyOCR (Production) |
| ⚠️ Currency Detection | **Demo Mode** | OCR-based pattern matching |
| ⚠️ Object Detection | **Simplified** | Edge-based detection |
| ✅ Audio TTS | **Active** | gTTS (Production) |

---

## ⚠️ Important Disclosures

### Current Limitations
- **🏗️ Prototype Status** – This is a demonstration system, not production-ready
- **💰 Currency Detection** – Uses OCR pattern matching, not trained ML model
- **🌙 Lighting Dependency** – Requires good lighting for optimal performance
- **🌐 Internet Required** – TTS needs online connectivity
- **💻 Desktop Only** – No mobile app deployment yet
- **🔍 Object Detection** – Rule-based, not semantic deep learning

### Safety Notices
```
⚠️ NOT A MEDICAL DEVICE – This is an assistive prototype
⚠️ NOT SAFETY-CRITICAL – Always use human supervision
⚠️ ADVISORY ONLY – Navigation guidance is not autonomous
⚠️ DEMO FEATURES – Some features use simplified algorithms
```

### Ethical Considerations
- ✅ **Privacy-First** – No image storage or data collection
- ✅ **Transparent AI** – Clear confidence scores and limitations
- ✅ **Accessible Design** – Keyboard navigation and audio feedback
- ✅ **Honest Communication** – Demo mode clearly indicated

---

## 🏗️ Production Roadmap

To transition from prototype to production, we've identified key areas for enhancement:

1. **Model Quality** – Training YOLO models on Indian datasets
2. **Offline Execution** – Mobile deployment and on-device inference
3. **Performance** – Latency guarantees and optimization
4. **Error Handling** – Accessible error recovery
5. **User Preferences** – Persistent settings and calibration
6. **Security & Privacy** – Data protection and consent
7. **Testing** – Comprehensive test suite and validation

For detailed implementation plans, see our development notes.

---

## 👥 Target Users

### Primary Users
- **🦯 Visually Impaired Individuals** – Complete blindness or low vision
- **👴 Elderly Users** – Age-related vision loss
- **🎓 Students** – Learning and educational assistance

### Secondary Users
- **🏪 Shopkeepers** – Currency verification assistance
- **👨‍👩‍👧 Family Members** – Supporting visually impaired relatives
- **🏥 Care Providers** – Assistive technology in care facilities

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add test cases
- 🌐 Add language support
- 🎨 Enhance UI/UX

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

**Attribution Required** – Please credit VoiceVision in derivative works.

---

## 🙏 Acknowledgements

Built with the goal of making AI **inclusive, ethical, and accessible**, especially for communities that need it most.

---

## 📞 Contact & Support

- 📧 **Email:** meenal.sinha09@gmail.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/MeenalSinha/voicevision/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/MeenalSinha/voicevision/discussions)
- 📖 **Documentation:** [Wiki](https://github.com/MeenalSinha/voicevision/wiki)

---

## 🌟 Star History

If you find VoiceVision helpful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Made with ❤️ for accessibility in India**

![Accessibility](https://img.shields.io/badge/Accessibility-First-blue?style=flat-square)
![AI Ethics](https://img.shields.io/badge/AI-Ethical-green?style=flat-square)
![Open Source](https://img.shields.io/badge/Open-Source-orange?style=flat-square)

*Empowering independence through artificial intelligence*

</div>
