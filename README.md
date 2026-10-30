# Face, Age, Gender & Object Detection System

A real-time Computer Vision application combining **OpenCV YuNet & SFace** for facial recognition, **Caffe / ONNX models** for age, gender, and facial expression detection, and **YOLOv8** for multi-class object detection, served via a high-performance **FastAPI** backend with WebSockets and an interactive dashboard.

---

## 🚀 Features

- **Real-Time Facial Recognition**:
  - Face detection powered by **YuNet** (fast, accurate on edge).
  - Face identification & embedding matching with **SFace**.
  - Dynamic user profile management with stored face descriptors.
- **Age, Gender & Emotion Detection**:
  - Deep learning models for age classification, gender estimation, and facial emotion analysis.
  - Standalone script `age_gender_emotion.py` supporting webcam and image input.
- **Object Detection with YOLOv8**:
  - Live object classification and bounding box localization powered by Ultralytics YOLOv8.
- **FastAPI & WebSockets**:
  - Low-latency bidirectional WebSocket video streaming.
  - RESTful APIs for profile registration, verification, and live testing.
- **Interactive Web Interface**:
  - Live webcam streaming with real-time detection overlays.
  - Image upload & batch testing.
  - Profile viewer and detection analytics.
- **Optional Gemini AI Integration**:
  - Visual reasoning and scene insights using Google Gemini API.

---

## 📁 Project Structure

```
face/
├── models/
│   ├── yunet.onnx          # YuNet face detection model
│   └── sface.onnx          # SFace face recognition model
├── static/
│   ├── index.html          # Web application UI
│   ├── style.css           # Modern dashboard styling
│   ├── app.js              # Real-time WebSocket & camera controller
│   ├── profiles/           # Registered face profiles
│   └── samples/            # Test sample images
├── facerecognisat.py       # Standalone face recognition pipeline
├── server.py               # FastAPI backend with WebSockets
├── verify_all.py           # Verification and testing script
├── yolov8n.pt              # YOLOv8 nano model weights
├── profiles.json           # Face database & descriptor metadata
├── .env.example            # Environment variables template
└── requirements.txt        # Python dependencies
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Shiva272001/Face-and-age-detection.git
cd Face-and-age-detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)
Copy `.env.example` to `.env` and add your API keys if using Gemini AI:
```bash
cp .env.example .env
```

---

## ⚡ Running the Application

### Start the Server
```bash
python server.py
# or
uvicorn server.py:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
