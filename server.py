import os
import sys
import io
import time
import json
import uuid
import base64
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

from dotenv import load_dotenv

# Google GenAI import
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROFILES_DIR = os.path.join(STATIC_DIR, "profiles")
PROFILES_FILE = os.path.join(BASE_DIR, "profiles.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

app = FastAPI(title="OmniVision AI - Face Recognition & YOLO Object Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Computer Vision Models ───────────────────────────────────────────────
print("⏳ Initializing YOLOv8 Object Detection...")
yolo_model_name = os.environ.get("YOLO_MODEL", "yolov8n.pt")
yolo_model_path = os.path.join(BASE_DIR, yolo_model_name)
if not os.path.exists(yolo_model_path):
    yolo_model_path = os.path.join(BASE_DIR, "yolov8n.pt")
if not os.path.exists(yolo_model_path):
    yolo_model_path = os.path.join(BASE_DIR, "yolov8s.pt")
yolo_model = YOLO(yolo_model_path)
print(f"✅ YOLOv8 loaded successfully ({os.path.basename(yolo_model_path)}).")

# Model warmup
try:
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = yolo_model(dummy_frame, verbose=False)
except Exception:
    pass

print("⏳ Initializing OpenCV YuNet + SFace Face Engines...")
yunet_path = os.path.join(MODELS_DIR, "yunet.onnx")
sface_path = os.path.join(MODELS_DIR, "sface.onnx")

face_detector = None
face_recognizer = None

if os.path.exists(yunet_path) and os.path.exists(sface_path):
    face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
    face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    print("✅ YuNet & SFace models loaded successfully.")
else:
    print("⚠️ YuNet or SFace models missing in models/ directory.")

# ─── Age, Gender & Facial Expression Models ──────────────────────────────────
print("⏳ Initializing Age, Gender and Emotion Recognition Networks...")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LIST = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

age_net = None
gender_net = None
emotion_net = None

age_onnx = os.path.join(MODELS_DIR, "age_googlenet.onnx")
if os.path.exists(age_onnx):
    try:
        age_net = cv2.dnn.readNet(age_onnx)
        print("✅ Age ONNX model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Age model load failed: {e}")

gender_onnx = os.path.join(MODELS_DIR, "gender_googlenet.onnx")
if os.path.exists(gender_onnx):
    try:
        gender_net = cv2.dnn.readNet(gender_onnx)
        print("✅ Gender ONNX model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Gender model load failed: {e}")

emotion_onnx = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")
if os.path.exists(emotion_onnx):
    try:
        emotion_net = cv2.dnn.readNet(emotion_onnx)
        print("✅ Emotion ONNX model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Emotion model load failed: {e}")

def get_square_face(frame: np.ndarray, face_box: List[int], padding: int = 10) -> np.ndarray:
    x1, y1, x2, y2 = face_box
    width, height = x2 - x1, y2 - y1
    size = max(width, height) + (padding * 2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = max(0, cx - size // 2)
    ny1 = max(0, cy - size // 2)
    nx2 = min(frame.shape[1], nx1 + size)
    ny2 = min(frame.shape[0], ny1 + size)
    return frame[ny1:ny2, nx1:nx2]

def predict_face_attributes(face_crop: np.ndarray, full_frame: np.ndarray, face_box: List[int]) -> Dict[str, Any]:
    gender = "Unknown"
    gender_conf = 0.0
    age = "Unknown"
    age_conf = 0.0
    emotion = "Neutral"
    emotion_conf = 0.0

    if (gender_net is not None or age_net is not None) and face_crop.size > 0:
        try:
            blob = cv2.dnn.blobFromImage(
                face_crop,
                1.0,
                (224, 224),
                MODEL_MEAN_VALUES,
                swapRB=False
            )
            if gender_net is not None:
                gender_net.setInput(blob)
                g_preds = gender_net.forward()
                g_idx = int(g_preds[0].argmax())
                gender = GENDER_LIST[g_idx]
                gender_conf = round(float(g_preds[0][g_idx] * 100), 1)

            if age_net is not None:
                age_net.setInput(blob)
                a_preds = age_net.forward()
                a_idx = int(a_preds[0].argmax())
                age = AGE_LIST[a_idx]
                age_conf = round(float(a_preds[0][a_idx] * 100), 1)
        except Exception as e:
            print(f"Attribute inference error: {e}")

    if emotion_net is not None:
        try:
            em_face = get_square_face(full_frame, face_box, padding=10)
            if em_face.size > 0:
                gray = cv2.cvtColor(em_face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                gray = cv2.equalizeHist(gray).astype(np.float32)
                em_blob = cv2.dnn.blobFromImage(gray, 1.0, (64, 64), (0,), swapRB=False, crop=False)
                emotion_net.setInput(em_blob)
                em_preds = emotion_net.forward()
                em_scores = np.squeeze(em_preds)
                # Softmax
                em_scores = em_scores - np.max(em_scores)
                em_exp = np.exp(em_scores)
                em_probs = em_exp / np.sum(em_exp)
                em_idx = int(np.argmax(em_probs))
                emotion = EMOTION_LIST[em_idx]
                emotion_conf = round(float(em_probs[em_idx] * 100), 1)
        except Exception as e:
            print(f"Emotion inference error: {e}")

    return {
        "gender": gender,
        "gender_conf": gender_conf,
        "age": age,
        "age_conf": age_conf,
        "emotion": emotion,
        "emotion_conf": emotion_conf
    }

# ─── Profile Store ─────────────────────────────────────────────────────────────
def load_profiles() -> List[Dict[str, Any]]:
    if not os.path.exists(PROFILES_FILE):
        return []
    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading profiles: {e}")
        return []

def save_profiles(profiles: List[Dict[str, Any]]):
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

def extract_face_embedding(img_bgr: np.ndarray) -> Optional[tuple]:
    """Detects primary face, crops/aligns, returns (embedding_list, bbox_list)"""
    if face_detector is None or face_recognizer is None:
        return None
    h, w, _ = img_bgr.shape
    face_detector.setInputSize((w, h))
    faces = face_detector.detect(img_bgr)[1]
    if faces is None or len(faces) == 0:
        return None
    
    # Pick largest face
    best_face = max(faces, key=lambda f: f[2] * f[3])
    aligned = face_recognizer.alignCrop(img_bgr, best_face)
    feat = face_recognizer.feature(aligned)
    bbox = [int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])]
    return feat[0].tolist(), bbox

def match_face(feat_vec: List[float], profiles: List[Dict[str, Any]], threshold: float = 0.363):
    """Matches 128-d feature vector against known profiles via cosine similarity"""
    if face_recognizer is None or not feat_vec:
        return None, 0.0
    
    feat_np = np.array([feat_vec], dtype=np.float32)
    best_profile = None
    best_score = -1.0
    
    for prof in profiles:
        emb = prof.get("embedding")
        if not emb:
            continue
        emb_np = np.array([emb], dtype=np.float32)
        score = float(face_recognizer.match(feat_np, emb_np, cv2.FaceRecognizerSF_FR_COSINE))
        if score > best_score:
            best_score = score
            best_profile = prof
            
    if best_profile and best_score >= threshold:
        return best_profile, best_score
    return None, best_score

# Lock for OpenCV DNN / YOLO inference
cv_lock = threading.Lock()

# ─── Pydantic Request Models ──────────────────────────────────────────────────
class ImageAnalyzeRequest(BaseModel):
    image_base64: str
    conf_thresh: float = 0.35
    detect_objects: bool = True
    detect_faces: bool = True
    draw_annotated: bool = True
    target_classes: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    model_name: Optional[str] = "gemini-2.5-flash"
    profile: Optional[Dict[str, Any]] = None
    detected_objects: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None

class ValidateKeyRequest(BaseModel):
    api_key: str
    model_name: Optional[str] = "gemini-2.5-flash"

class AutoGreetRequest(BaseModel):
    api_key: Optional[str] = None
    model_name: Optional[str] = "gemini-2.5-flash"
    profile: Dict[str, Any]
    detected_objects: Optional[List[str]] = None

class ProfileCreateRequest(BaseModel):
    name: str
    age: Optional[str] = "Unknown"
    job: Optional[str] = "Team Member"
    department: Optional[str] = "Engineering"
    notes: Optional[str] = ""
    image_base64: str

# ─── Helper Functions ──────────────────────────────────────────────────────────
def decode_image_base64(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img

def encode_image_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

# ─── Core API Endpoints ────────────────────────────────────────────────────────

@app.get("/api/cameras")
def list_available_cameras():
    """Detects physically available hardware camera indices (built-in & external) via OpenCV"""
    available = []
    cameras_info = []
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    for idx in range(4):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(idx)
                h, w = frame.shape[:2]
                label = "Camera 0 (Built-in Webcam)" if idx == 0 else f"Camera {idx} (External Camera)"
                cameras_info.append({
                    "index": idx,
                    "label": label,
                    "type": "builtin" if idx == 0 else "external",
                    "width": int(w),
                    "height": int(h)
                })
            cap.release()
    return {"available_cameras": available, "cameras": cameras_info, "count": len(available)}

@app.get("/api/camera-snapshot/{cam_id}")
def get_hardware_camera_snapshot(cam_id: int):
    """Captures a snapshot directly from hardware Camera 0 or Camera 1"""
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot open camera index {cam_id}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to read frame from camera {cam_id}")
    
    b64 = encode_image_base64(frame)
    return {"success": True, "camera_id": cam_id, "image_base64": b64}

@app.get("/api/status")
def get_system_status():
    profiles = load_profiles()
    env_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "online",
        "yolo_ready": True,
        "yolo_model": "YOLOv8n (COCO 80 classes)",
        "face_engine_ready": face_detector is not None and face_recognizer is not None,
        "age_model_ready": age_net is not None,
        "gender_model_ready": gender_net is not None,
        "emotion_model_ready": emotion_net is not None,
        "profiles_count": len(profiles),
        "profiles": [{"id": p.get("id"), "name": p.get("name"), "job": p.get("job")} for p in profiles],
        "genai_sdk_available": GENAI_AVAILABLE,
        "server_api_key_configured": env_key,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/validate-key")
def validate_gemini_key(req: ValidateKeyRequest):
    key = req.api_key.strip() if req.api_key else ""
    if not key:
        return JSONResponse(status_code=400, content={"valid": False, "error": "API key cannot be empty."})
    
    if not GENAI_AVAILABLE:
        return JSONResponse(status_code=500, content={"valid": False, "error": "google-genai SDK not installed."})

    try:
        client = genai.Client(api_key=key)
        # Test prompt
        model = req.model_name or "gemini-2.5-flash"
        resp = client.models.generate_content(
            model=model,
            contents="Say 'OK' if you can read this."
        )
        return {"valid": True, "message": "Key is valid and active!", "model": model, "preview": resp.text[:50]}
    except Exception as e:
        err_msg = str(e)
        if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
            err_msg = "Invalid API Key. Please verify in Google AI Studio."
        return JSONResponse(status_code=400, content={"valid": False, "error": err_msg})

@app.post("/api/analyze")
def analyze_image(req: ImageAnalyzeRequest):
    t_start = time.time()
    try:
        img_bgr = decode_image_base64(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")

    h, w, _ = img_bgr.shape
    annotated = img_bgr.copy() if req.draw_annotated else None
    
    detected_objects = []
    detected_faces = []
    active_profile = None

    # Thread-safe model forward
    with cv_lock:
        # 1. YOLO Object Detection
        if req.detect_objects:
            yolo_results = yolo_model(img_bgr, conf=req.conf_thresh, verbose=False)
            for r in yolo_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = yolo_model.names[cls_id]
                    conf = float(box.conf[0])
                    
                    if req.target_classes and cls_name not in req.target_classes:
                        continue

                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    bx = xyxy[0]
                    by = xyxy[1]
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]

                    obj_info = {
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [bx, by, bw, bh],
                        "color": "#10b981"  # Emerald green for YOLO
                    }
                    detected_objects.append(obj_info)

                    if annotated is not None:
                        # Draw YOLO bounding box
                        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (16, 185, 129), 2)
                        label = f"{cls_name} {int(conf*100)}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated, (bx, by - 22), (bx + tw + 8, by), (16, 185, 129), -1)
                        cv2.putText(annotated, label, (bx + 4, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Face Recognition
        if req.detect_faces and face_detector is not None and face_recognizer is not None:
            profiles = load_profiles()
            face_detector.setInputSize((w, h))
            faces = face_detector.detect(img_bgr)[1]

            if faces is not None and len(faces) > 0:
                for face in faces:
                    fbx = int(face[0])
                    fby = int(face[1])
                    fbw = int(face[2])
                    fbh = int(face[3])

                    aligned = face_recognizer.alignCrop(img_bgr, face)
                    feat = face_recognizer.feature(aligned)
                    feat_vec = feat[0].tolist()

                    matched_prof, score = match_face(feat_vec, profiles, threshold=0.363)
                    
                    # Crop face for age, gender and emotion prediction
                    crop_pad = 20
                    cx1 = max(0, fbx - crop_pad)
                    cy1 = max(0, fby - crop_pad)
                    cx2 = min(w, fbx + fbw + crop_pad)
                    cy2 = min(h, fby + fbh + crop_pad)
                    face_crop = img_bgr[cy1:cy2, cx1:cx2]

                    # Predict Age, Gender, Emotion
                    attrs = predict_face_attributes(face_crop, img_bgr, [fbx, fby, fbx + fbw, fby + fbh])

                    if matched_prof:
                        face_info = {
                            "name": matched_prof["name"],
                            "matched": True,
                            "confidence": round(float(score), 3),
                            "age": attrs["age"] if attrs["age"] != "Unknown" else matched_prof.get("age", "N/A"),
                            "age_detected": attrs["age"],
                            "age_conf": attrs["age_conf"],
                            "gender": attrs["gender"],
                            "gender_conf": attrs["gender_conf"],
                            "emotion": attrs["emotion"],
                            "emotion_conf": attrs["emotion_conf"],
                            "job": matched_prof.get("job", "Team Member"),
                            "department": matched_prof.get("department", "General"),
                            "notes": matched_prof.get("notes", ""),
                            "bbox": [fbx, fby, fbw, fbh],
                            "id": matched_prof.get("id"),
                            "image": matched_prof.get("image", ""),
                            "color": "#00f2fe"  # Cyan for recognized face
                        }
                        if active_profile is None:
                            active_profile = face_info
                    else:
                        face_info = {
                            "name": "Guest Visitor",
                            "matched": False,
                            "confidence": round(float(score), 3) if score > 0 else 0.0,
                            "age": attrs["age"],
                            "age_detected": attrs["age"],
                            "age_conf": attrs["age_conf"],
                            "gender": attrs["gender"],
                            "gender_conf": attrs["gender_conf"],
                            "emotion": attrs["emotion"],
                            "emotion_conf": attrs["emotion_conf"],
                            "job": "Visitor",
                            "department": "Guest Area",
                            "notes": "Unrecognized face in database.",
                            "bbox": [fbx, fby, fbw, fbh],
                            "color": "#f59e0b"  # Amber for guest
                        }
                        if active_profile is None:
                            active_profile = face_info

                    detected_faces.append(face_info)

                    if annotated is not None:
                        # Draw Face bounding box
                        box_color = (254, 242, 0) if face_info["matched"] else (11, 158, 245)
                        cv2.rectangle(annotated, (fbx, fby), (fbx + fbw, fby + fbh), box_color, 2)

                        # Top badge: Name + match %
                        flabel = f"👤 {face_info['name']} ({int(face_info['confidence']*100)}%)" if face_info['matched'] else "👤 Guest"
                        (tw, th), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        cv2.rectangle(annotated, (fbx, fby - 24), (fbx + tw + 10, fby), box_color, -1)
                        cv2.putText(annotated, flabel, (fbx + 5, fby - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                        # Bottom badge: Gender + Age + Emotion
                        attr_label = f"{attrs['gender']} | Age: {attrs['age']} | {attrs['emotion']}"
                        (atw, ath), _ = cv2.getTextSize(attr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        badge_y1 = fby + fbh
                        badge_y2 = min(h, badge_y1 + ath + 12)
                        cv2.rectangle(annotated, (fbx, badge_y1), (fbx + atw + 10, badge_y2), (30, 41, 59), -1)
                        cv2.putText(annotated, attr_label, (fbx + 5, badge_y1 + ath + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    inference_ms = round((time.time() - t_start) * 1000, 1)

    annotated_b64 = encode_image_base64(annotated) if annotated is not None else None

    return {
        "success": True,
        "inference_ms": inference_ms,
        "image_size": {"width": w, "height": h},
        "annotated_image": annotated_b64,
        "objects": detected_objects,
        "faces": detected_faces,
        "active_profile": active_profile,
        "summary": {
            "total_objects": len(detected_objects),
            "total_faces": len(detected_faces),
            "object_classes": list(set(o["class"] for o in detected_objects))
        }
    }

# ─── Gemini Chat Endpoint ──────────────────────────────────────────────────────

@app.post("/api/chat")
def chat_with_gemini(req: ChatRequest):
    # Determine API key
    api_key = req.api_key.strip() if req.api_key else os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    profile = req.profile or {
        "name": "Guest",
        "job": "Visitor",
        "department": "Office",
        "age": "Unknown"
    }

    objects_context = ""
    detected_list = req.detected_objects or []
    if detected_list:
        counts = {}
        for obj in detected_list:
            counts[obj] = counts.get(obj, 0) + 1
        obj_summary = ", ".join([f"{v} {k}{'s' if v > 1 and not k.endswith('s') else ''}" for k, v in counts.items()])
        objects_context = f"\nCurrently detected objects in the user's camera environment: {obj_summary}."

    system_instruction = f"""You are OmniVision AI, an advanced, highly intelligent smart office and computer vision AI assistant.
You are currently interacting with a user whose identity has been detected by the vision system:
- Name: {profile.get('name', 'Guest')}
- Role/Job: {profile.get('job', 'Visitor')}
- Department: {profile.get('department', 'N/A')}
- Age: {profile.get('age', 'N/A')}
- Profile Notes: {profile.get('notes', 'None')}
{objects_context}

Communication Guidelines:
1. Address the person naturally by their first name when appropriate.
2. Tailor technical depth and tone to their profession (e.g. data science, engineering, or general visitor).
3. If asked about their surroundings, refer accurately to the detected objects in their workspace.
4. Keep answers concise, insightful, helpful, and professional with clean markdown formatting.
5. If they are a Guest, be welcoming, courteous, and offer visitor assistance.
"""

    model_name = req.model_name or "gemini-2.5-flash"

    # Try calling Google GenAI
    if api_key and GENAI_AVAILABLE and not api_key.startswith("invalid"):
        try:
            client = genai.Client(api_key=api_key)
            
            contents = []
            if req.history:
                for item in req.history[-8:]:
                    role = "user" if item.get("role") in ("user", "human") else "model"
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=item.get("content", ""))]
                    ))
            
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=req.message)]
            ))

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=800
            )

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )

            if response.text:
                return {
                    "reply": response.text,
                    "profile_addressed": profile.get("name"),
                    "model_used": model_name
                }
        except Exception as e:
            print(f"Gemini API execution error: {e}")
            # Fall through to contextual fallback with helpful diagnostic

    # Contextual Natural Conversational Engine
    name = profile.get("name", "there").title()
    job = profile.get("job", "Team Member")
    dept = profile.get("department", "Engineering")
    msg_lower = req.message.lower().strip()
    
    # Formulate object summary
    if detected_list:
        counts = {}
        for obj in detected_list:
            counts[obj] = counts.get(obj, 0) + 1
        obj_str = ", ".join([f"{v} {k}{'s' if v > 1 and not k.endswith('s') else ''}" for k, v in counts.items()])
    else:
        obj_str = "no extra items"

    hour = datetime.now().hour
    time_greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    if any(g in msg_lower for g in ["hi", "hello", "hey", "greet", "greetings"]):
        reply = (
            f"Hello **{name}**! {time_greeting}.\n\n"
            f"It's great to see you in the workspace. As an **{job}** in **{dept}**, how is your session going today?\n\n"
            f"I have our YOLOv8 and Face Recognition models running live — currently seeing **{obj_str}** in your environment. "
            f"How can I assist you right now?"
        )
    elif any(q in msg_lower for q in ["what do you see", "what objects", "what's around", "surroundings", "desk", "room", "environment"]):
        reply = (
            f"**Workspace Environment Scan for {name}:**\n\n"
            f"- **Recognized Identity:** {name} ({job})\n"
            f"- **Active Objects Detected:** {obj_str}\n"
            f"- **Vision Engine:** YOLOv8n Object Detector + OpenCV SFace\n\n"
            f"All detections are running smoothly in real-time."
        )
    elif any(g in msg_lower for g in ["goal", "today's goal", "productivity", "task"]):
        reply = (
            f"Here is a recommended focus goal for today, **{name}**:\n\n"
            f"1. **Core Development**: Optimize your AI model inference pipelines and test performance.\n"
            f"2. **Vision Integration**: Validate real-time object detection and face recognition workflows.\n"
            f"3. **Milestone Review**: Document today's key achievements before wrapping up.\n\n"
            f"Let me know if you'd like tips on any specific machine learning or vision techniques!"
        )
    elif any(e in msg_lower for e in ["ergonomic", "posture", "chair"]):
        reply = (
            f"**Ergonomic Workspace Assessment for {name}:**\n\n"
            f"I detect a **{obj_str}** in your station. Make sure your monitor is at eye level, "
            f"your shoulders are relaxed, and take a 2-minute break every hour to keep focus high!"
        )
    else:
        reply = (
            f"**{name}**, regarding your query: *\"{req.message}\"*\n\n"
            f"As an **{job}**, that's a great topic. I am actively analyzing your workspace ({obj_str}) "
            f"and ready to assist with model training, computer vision pipelines, or system architecture. "
            f"What specific area would you like to explore next?"
        )

    return {
        "reply": reply,
        "profile_addressed": name,
        "model_used": "OmniVision Conversational Assistant"
    }

@app.post("/api/greet")
def auto_greet_person(req: AutoGreetRequest):
    api_key = req.api_key.strip() if req.api_key else os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Fallback greeting without API call
        name = req.profile.get("name", "there")
        job = req.profile.get("job", "")
        return {
            "greeting": f"Hello {name}! Welcome back. How can I assist you with your {job} tasks today?",
            "is_generated": False
        }

    hour = datetime.now().hour
    time_greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    profile = req.profile
    name = profile.get("name", "Guest")
    job = profile.get("job", "Visitor")

    detected_str = f"Objects in view: {', '.join(req.detected_objects)}." if req.detected_objects else ""

    prompt = f"""Generate a short, warm, 1-to-2 sentence proactive voice greeting for {name}, who is a {job} in {profile.get('department', 'our team')}.
Current time: {time_greeting}. {detected_str}
Greet them personally, mention their role briefly or wish them a productive session. Do not use quotes around the greeting."""

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=req.model_name or "gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.8, max_output_tokens=150)
        )
        return {"greeting": resp.text.strip(), "is_generated": True}
    except Exception as e:
        return {
            "greeting": f"{time_greeting}, {name}! Welcome to the workspace. How may I assist your {job} work today?",
            "is_generated": False
        }

# ─── Profile Management Endpoints ──────────────────────────────────────────────

@app.get("/api/profiles")
def list_profiles():
    profiles = load_profiles()
    sanitized = []
    for p in profiles:
        item = {k: v for k, v in p.items() if k != "embedding"}
        item["has_embedding"] = bool(p.get("embedding"))
        sanitized.append(item)
    return {"profiles": sanitized}

@app.post("/api/profiles")
def create_profile(req: ProfileCreateRequest):
    try:
        img_bgr = decode_image_base64(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")

    extracted = extract_face_embedding(img_bgr)
    if extracted is None:
        raise HTTPException(
            status_code=400,
            detail="No clear face detected in the provided image. Please ensure your face is well-lit and facing the camera directly."
        )

    feat_list, bbox = extracted
    profile_id = f"prof_{uuid.uuid4().hex[:8]}"
    
    # Save image file
    img_filename = f"{profile_id}.jpg"
    img_rel_path = f"static/profiles/{img_filename}"
    img_abs_path = os.path.join(STATIC_DIR, "profiles", img_filename)
    cv2.imwrite(img_abs_path, img_bgr)

    new_profile = {
        "id": profile_id,
        "name": req.name.strip(),
        "age": req.age.strip() if req.age else "Unknown",
        "job": req.job.strip() if req.job else "Team Member",
        "department": req.department.strip() if req.department else "General",
        "notes": req.notes.strip() if req.notes else "",
        "image": img_rel_path,
        "embedding": feat_list,
        "created_at": datetime.now().isoformat()
    }

    profiles = load_profiles()
    profiles.append(new_profile)
    save_profiles(profiles)

    return {
        "success": True,
        "message": f"Profile '{new_profile['name']}' successfully registered!",
        "profile": {k: v for k, v in new_profile.items() if k != "embedding"}
    }

@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str):
    profiles = load_profiles()
    initial_len = len(profiles)
    matched = [p for p in profiles if p.get("id") == profile_id]
    
    if not matched:
        raise HTTPException(status_code=404, detail="Profile not found.")

    profiles = [p for p in profiles if p.get("id") != profile_id]
    save_profiles(profiles)

    # Clean up image file if exists
    img_path = matched[0].get("image")
    if img_path and os.path.exists(os.path.join(BASE_DIR, img_path)):
        try:
            os.remove(os.path.join(BASE_DIR, img_path))
        except Exception:
            pass

    return {"success": True, "message": "Profile removed successfully."}

# ─── Static Files & Frontend ───────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>OmniVision AI</h1><p>Frontend static/index.html is building...</p>"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
