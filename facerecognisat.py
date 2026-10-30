# -*- coding: utf-8 -*-
"""
Standalone Face Recognition & Conversational AI System
Adapted for local execution (Windows/macOS/Linux) using OpenCV YuNet + SFace and Gemini AI.
"""

import os
import sys
import time
import cv2
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Set UTF-8 encoding for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# STEP 1 & 2 — Environment & Model Setup
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "sface.onnx")

face_detector = None
face_recognizer = None

if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
    face_detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320))
    face_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    print("✅ YuNet & SFace models loaded successfully.")
else:
    print("⚠️ YuNet or SFace models missing in models/ directory.")

# ─── Age, Gender & Emotion Analysis Models ────────────────────────────────────
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
        print("✅ Age ONNX model loaded.")
    except Exception as e:
        print(f"⚠️ Age model load failed: {e}")

gender_onnx = os.path.join(MODELS_DIR, "gender_googlenet.onnx")
if os.path.exists(gender_onnx):
    try:
        gender_net = cv2.dnn.readNet(gender_onnx)
        print("✅ Gender ONNX model loaded.")
    except Exception as e:
        print(f"⚠️ Gender model load failed: {e}")

emotion_onnx = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")
if os.path.exists(emotion_onnx):
    try:
        emotion_net = cv2.dnn.readNet(emotion_onnx)
        print("✅ Emotion ONNX model loaded.")
    except Exception as e:
        print(f"⚠️ Emotion model load failed: {e}")

def get_square_face(frame: np.ndarray, face_box, padding: int = 10) -> np.ndarray:
    x1, y1, x2, y2 = face_box
    width, height = x2 - x1, y2 - y1
    size = max(width, height) + (padding * 2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = max(0, cx - size // 2)
    ny1 = max(0, cy - size // 2)
    nx2 = min(frame.shape[1], nx1 + size)
    ny2 = min(frame.shape[0], ny1 + size)
    return frame[ny1:ny2, nx1:nx2]

def predict_face_attributes(face_crop: np.ndarray, full_frame: np.ndarray, face_box) -> dict:
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
        except Exception:
            pass

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
                em_scores = em_scores - np.max(em_scores)
                em_exp = np.exp(em_scores)
                em_probs = em_exp / np.sum(em_exp)
                em_idx = int(np.argmax(em_probs))
                emotion = EMOTION_LIST[em_idx]
                emotion_conf = round(float(em_probs[em_idx] * 100), 1)
        except Exception:
            pass

    return {
        "gender": gender,
        "gender_conf": gender_conf,
        "age": age,
        "age_conf": age_conf,
        "emotion": emotion,
        "emotion_conf": emotion_conf
    }

# Setup Gemini LLM via LangChain (with graceful offline fallback)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
llm = None
HumanMessage = None
AIMessage = None
ChatPromptTemplate = None
MessagesPlaceholder = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    if api_key and not api_key.startswith("invalid") and not api_key.startswith("AIzaSyCYKuDj8"):
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
        print("✅ Gemini AI language model configured.")
    else:
        # We will attempt Gemini first, and fall back to local conversational engine if key is invalid
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key or "demo")
except Exception as e:
    pass

# ─────────────────────────────────────────────
# STEP 3 — Face History Database
# ─────────────────────────────────────────────
FACE_DB = [
    {
        "name": "Alice",
        "age": 28,
        "job": "Data Scientist",
        "department": "AI Research",
        "image": "static/profiles/alice.jpg"
    },
    {
        "name": "Bob",
        "age": 35,
        "job": "Software Engineer",
        "department": "Backend",
        "image": "static/profiles/bob.jpg"
    },
]

# If profiles.json exists, also merge any registered profiles
profiles_file = os.path.join(BASE_DIR, "profiles.json")
if os.path.exists(profiles_file):
    try:
        with open(profiles_file, "r", encoding="utf-8") as f:
            saved_profiles = json.load(f)
            names_in_db = {p["name"].lower() for p in FACE_DB}
            for sp in saved_profiles:
                if sp.get("name") and sp.get("name").lower() not in names_in_db:
                    FACE_DB.append({
                        "name": sp.get("name"),
                        "age": sp.get("age", 30),
                        "job": sp.get("job", "Team Member"),
                        "department": sp.get("department", "Engineering"),
                        "image": sp.get("image", ""),
                        "embedding": sp.get("embedding")
                    })
    except Exception as e:
        print(f"⚠️ Note reading profiles.json: {e}")

# ─────────────────────────────────────────────
# STEP 4 — Load & Encode Faces from DB
# ─────────────────────────────────────────────
def extract_embedding(img_bgr: np.ndarray):
    """Detects primary face using YuNet and extracts 128-d embedding using SFace."""
    if face_detector is None or face_recognizer is None or img_bgr is None:
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
    return feat[0]

known_encodings = []
known_profiles = []

for person in FACE_DB:
    # Check if precomputed embedding is present
    if person.get("embedding"):
        known_encodings.append(np.array(person["embedding"], dtype=np.float32))
        known_profiles.append(person)
        print(f"✅ Loaded: {person['name']} ({person['job']}) [Cached Embedding]")
        continue

    img_path = person["image"]
    full_path = os.path.join(BASE_DIR, img_path) if not os.path.isabs(img_path) else img_path

    if os.path.exists(full_path):
        img = cv2.imread(full_path)
        feat = extract_embedding(img)
        if feat is not None:
            known_encodings.append(feat)
            known_profiles.append(person)
            print(f"✅ Loaded: {person['name']} ({person['job']})")
        else:
            print(f"⚠️  No face found in {img_path}")
    else:
        print(f"⚠️  File not found: {img_path}")

print(f"\n📦 {len(known_profiles)} people loaded into face history.\n")

# ─────────────────────────────────────────────
# STEP 5 — Chat History Store
# ─────────────────────────────────────────────
chat_histories: dict[str, list] = {}

def get_history(name: str) -> list:
    if name not in chat_histories:
        chat_histories[name] = []
    return chat_histories[name]

def show_history(name: str):
    history = chat_histories.get(name, [])
    if not history:
        print(f"📭 No chat history for '{name}' yet.")
        return
    print(f"\n{'─'*50}")
    print(f"📜  Chat history — {name}")
    print(f"{'─'*50}")
    for item in history:
        if isinstance(item, tuple):
            role, content = item
            prefix = "🧑 You " if role == "user" else "🤖 AI  "
        elif HumanMessage and isinstance(item, HumanMessage):
            prefix, content = "🧑 You ", item.content
        else:
            prefix, content = "🤖 AI  ", getattr(item, 'content', str(item))
        print(f"{prefix}: {content}")
    print(f"{'─'*50}\n")

def clear_history(name: str = None):
    if name:
        chat_histories.pop(name, None)
        print(f"🗑️  Cleared history for {name}")
    else:
        chat_histories.clear()
        print("🗑️  All chat histories cleared")

def list_active_sessions():
    if not chat_histories:
        print("No active sessions.")
        return
    print("\n🗂️  Active chat sessions:")
    for name, msgs in chat_histories.items():
        print(f"  • {name}: {len(msgs)//2} exchanges")

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# STEP 6 — Live Camera & External Camera Capture
# ─────────────────────────────────────────────
def detect_available_cameras(max_indices: int = 4) -> list:
    """
    Scans hardware camera indices to discover both Built-in and External cameras.
    Returns list of dicts: [{'index': int, 'name': str, 'width': int, 'height': int}]
    """
    found = []
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                label = "Built-in Webcam" if idx == 0 else f"External Camera (Cam {idx})"
                found.append({
                    "index": idx,
                    "name": label,
                    "width": w,
                    "height": h
                })
            cap.release()
    return found

def capture_live_photo(filename='captured.jpg'):
    """
    Captures photo from live built-in or external webcam, with live camera switching via TAB.
    Supports CLI arguments:
      --image <path>        Use an existing image
      --camera <0|1|...>    Select specific camera index (e.g. --camera 1 for external camera)
    """
    # 1. Check CLI arguments for image file
    for i, arg in enumerate(sys.argv):
        if arg in ("--image", "-i") and i + 1 < len(sys.argv):
            arg_path = sys.argv[i + 1]
            if os.path.exists(arg_path):
                frame = cv2.imread(arg_path)
                cv2.imwrite(filename, frame)
                print(f"✅ Loaded image from argument: {arg_path}")
                return filename

    # 2. Check CLI argument for specific camera index
    cli_camera_idx = None
    for i, arg in enumerate(sys.argv):
        if arg in ("--camera", "-c") and i + 1 < len(sys.argv):
            val = sys.argv[i + 1].strip().lower()
            if val in ("external", "ext", "usb"):
                cli_camera_idx = 1
            else:
                try:
                    cli_camera_idx = int(val)
                except ValueError:
                    cli_camera_idx = 0
            print(f"📷 Command line camera selection: Index {cli_camera_idx}")

    sample_alice = os.path.join(BASE_DIR, "static", "samples", "alice_test.jpg")
    sample_bob = os.path.join(BASE_DIR, "static", "samples", "bob_test.jpg")

    # Detect available cameras
    cameras = detect_available_cameras(max_indices=4)

    selected_camera_idx = cli_camera_idx

    if selected_camera_idx is None:
        print("\n📸 Select Input Source:")
        menu_items = []
        if cameras:
            for cam in cameras:
                menu_idx = len(menu_items) + 1
                menu_items.append(("cam", cam["index"]))
                print(f"  [{menu_idx}] Live Camera: {cam['name']} ({cam['width']}x{cam['height']})")
        else:
            menu_idx = len(menu_items) + 1
            menu_items.append(("cam", 0))
            print("  [1] Live Camera (Default)")

        menu_items.append(("sample", sample_alice))
        print(f"  [{len(menu_items)}] Test Sample: Alice ({os.path.basename(sample_alice)})")

        menu_items.append(("sample", sample_bob))
        print(f"  [{len(menu_items)}] Test Sample: Bob ({os.path.basename(sample_bob)})")

        menu_items.append(("custom", None))
        print(f"  [{len(menu_items)}] Custom Image File Path")

        try:
            choice_str = input(f"👉 Enter choice [1-{len(menu_items)}] (default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice_str = "1"

        try:
            choice_num = int(choice_str) if choice_str else 1
            if 1 <= choice_num <= len(menu_items):
                item_type, item_val = menu_items[choice_num - 1]
            else:
                item_type, item_val = menu_items[0]
        except ValueError:
            # Check if user typed camera number e.g. "0" or "1"
            if choice_str in ("0", "1", "2"):
                item_type, item_val = ("cam", int(choice_str))
            else:
                item_type, item_val = menu_items[0]

        if item_type == "sample":
            if os.path.exists(item_val):
                frame = cv2.imread(item_val)
                cv2.imwrite(filename, frame)
                print(f"✅ Using sample: {item_val}")
                return filename
        elif item_type == "custom":
            try:
                custom_path = input("Enter image path: ").strip()
            except (EOFError, KeyboardInterrupt):
                custom_path = ""
            if custom_path and os.path.exists(custom_path):
                frame = cv2.imread(custom_path)
                cv2.imwrite(filename, frame)
                print(f"✅ Loaded custom image: {custom_path}")
                return filename
            else:
                print("⚠️ File not found. Falling back to primary camera.")
                selected_camera_idx = 0
        else:
            selected_camera_idx = item_val

    if selected_camera_idx is None:
        selected_camera_idx = 0

    available_indices = [c["index"] for c in cameras] if cameras else [selected_camera_idx]
    if selected_camera_idx not in available_indices:
        available_indices.append(selected_camera_idx)

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

    def open_camera(cam_idx):
        cap = cv2.VideoCapture(cam_idx, backend)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    current_idx_pos = available_indices.index(selected_camera_idx) if selected_camera_idx in available_indices else 0
    active_idx = available_indices[current_idx_pos]

    print(f"\n📷 Opening Camera Index {active_idx}...")
    cap = open_camera(active_idx)

    # Fallback to other available cameras if failed
    if cap is None:
        for idx in available_indices:
            if idx != active_idx:
                print(f"⚠️ Failed to open Cam {active_idx}, attempting Cam {idx}...")
                cap = open_camera(idx)
                if cap is not None:
                    active_idx = idx
                    current_idx_pos = available_indices.index(idx)
                    break

    if cap is None or not cap.isOpened():
        print("⚠️ No physical webcam could be opened.")
        default_sample = sample_alice if os.path.exists(sample_alice) else sample_bob
        print(f"💡 Falling back to sample image: {default_sample}")
        if os.path.exists(default_sample):
            frame = cv2.imread(default_sample)
            cv2.imwrite(filename, frame)
            return filename
        return None

    captured = False
    window_name = "OmniVision Camera - [SPACE] Capture | [TAB] Switch Camera | [Q] Cancel"

    print("👉 Preview Controls:")
    print("   • [SPACE] or [C]: Capture Photo")
    print("   • [TAB] or [T]:   Switch Camera (Built-in <-> External)")
    print("   • [0-9]:          Switch directly to Camera Index")
    print("   • [ESC] or [Q]:   Cancel")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Failed to grab frame from Camera {active_idx}. Re-opening...")
            time.sleep(0.1)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        h, w = frame.shape[:2]
        cam_type = "Built-in" if active_idx == 0 else f"External Cam {active_idx}"

        # Draw modern bottom banner
        cv2.rectangle(frame, (0, h - 45), (w, h), (18, 18, 24), -1)
        cv2.putText(
            frame,
            f"Active: Cam {active_idx} [{cam_type}] | [SPACE] Capture | [TAB] Switch Cam | [Q] Cancel",
            (14, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 242, 254),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # Capture
        if key in (32, ord('c'), ord('C')):
            cv2.imwrite(filename, frame)
            captured = True
            print(f"✅ Photo captured from Camera {active_idx} and saved: {filename}")
            break

        # Switch camera with TAB (ASCII 9), T, or N
        elif key in (9, ord('t'), ord('T'), ord('n'), ord('N')) and len(available_indices) > 1:
            current_idx_pos = (current_idx_pos + 1) % len(available_indices)
            new_idx = available_indices[current_idx_pos]
            print(f"🔄 Switching from Camera {active_idx} to Camera {new_idx}...")
            cap.release()
            new_cap = open_camera(new_idx)
            if new_cap is not None:
                cap = new_cap
                active_idx = new_idx
                print(f"✅ Now streaming on Camera {active_idx}!")
            else:
                print(f"❌ Failed to open Camera {new_idx}. Reverting to Camera {active_idx}.")
                cap = open_camera(active_idx)

        # Direct number key switch (0, 1, 2, ...)
        elif ord('0') <= key <= ord('9'):
            target_idx = key - ord('0')
            if target_idx in available_indices and target_idx != active_idx:
                print(f"🔄 Switching directly to Camera {target_idx}...")
                cap.release()
                new_cap = open_camera(target_idx)
                if new_cap is not None:
                    cap = new_cap
                    active_idx = target_idx
                    current_idx_pos = available_indices.index(active_idx)
                    print(f"✅ Now streaming on Camera {active_idx}!")
                else:
                    print(f"❌ Failed to open Camera {target_idx}. Reverting to Camera {active_idx}.")
                    cap = open_camera(active_idx)

        # Cancel
        elif key in (27, ord('q'), ord('Q')):
            print("❌ Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return filename if captured else None

# ─────────────────────────────────────────────
# STEP 7 — Face Recognition from Captured Photo
# ─────────────────────────────────────────────
def recognize_face(img_path: str) -> dict | None:
    """
    Returns profile dict if face matched in history,
    or a 'Guest' profile if not matched,
    or None if no face detected at all.
    """
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to decode image: {img_path}")
        return None

    query_feat = extract_embedding(img)
    if query_feat is None:
        print("😶  No face detected in the photo. Try again.")
        return None

    print(f"👤  Face detected and embedding extracted.")

    best_match = None
    best_score = -1.0

    # Compare with known profiles using SFace cosine similarity
    for idx, known_feat in enumerate(known_encodings):
        score = float(face_recognizer.match(
            np.array([query_feat], dtype=np.float32),
            np.array([known_feat], dtype=np.float32),
            cv2.FaceRecognizerSF_FR_COSINE
        ))
        if score > best_score:
            best_score = score
            best_match = known_profiles[idx]

    # Threshold for OpenCV SFace: cosine similarity >= 0.363 is considered a match
    match_threshold = 0.363
    if best_match and best_score >= match_threshold:
        confidence_pct = min(100, int((best_score / 0.8) * 100))
        print(f"✅  Matched: {best_match['name']} ({best_match['job']}) [Similarity: {best_score:.3f}, ~{confidence_pct}% match]")
        return best_match
    else:
        closest_name = best_match['name'] if best_match else "None"
        print(f"🔍  No confident match found (closest: {closest_name} at {best_score:.3f}). Proceeding as Guest.")
        return {
            "name": "Guest",
            "age": "Unknown",
            "job": "Visitor",
            "department": "Office"
        }

# ─────────────────────────────────────────────
# STEP 8 — Conversational Engine (Gemini / Fallback)
# ─────────────────────────────────────────────
def build_system_prompt(profile: dict) -> str:
    return f"""You are a warm, professional smart office AI assistant.

The person you are speaking with has been identified:
- Name       : {profile.get('name', 'Guest')}
- Age        : {profile.get('age', 'Unknown')}
- Job        : {profile.get('job', 'Visitor')}
- Department : {profile.get('department', 'N/A')}

Guidelines:
1. Address them by their first name naturally.
2. Tailor your responses to their job role when relevant.
3. Remember everything they tell you in this conversation.
4. Be concise but warm.
5. If they are a Guest (unknown person), still be helpful and welcoming.
"""

def chat_with_profile(profile: dict, user_input: str) -> str:
    """
    Send a message using profile context + chat history.
    Tries Gemini via LangChain first, and falls back to a smart contextual response if offline.
    """
    name = profile.get("name", "Guest")
    history = get_history(name)

    # 1. Try Gemini via LangChain if available
    if llm and ChatPromptTemplate and MessagesPlaceholder:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", build_system_prompt(profile)),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            chain = prompt | llm
            
            # Format history for LangChain
            lc_history = []
            for item in history:
                if isinstance(item, tuple):
                    role, text = item
                    lc_history.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
                else:
                    lc_history.append(item)

            response = chain.invoke({"history": lc_history, "input": user_input})
            reply_text = getattr(response, 'text', None) or (response.content if isinstance(response.content, str) else str(response.content))

            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=reply_text))
            return reply_text
        except Exception:
            # Fall through to offline conversational engine
            pass

    # 2. Contextual Intelligent Fallback Engine
    job = profile.get("job", "Team Member")
    dept = profile.get("department", "Engineering")
    hour = datetime.now().hour
    time_greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    msg_lower = user_input.lower().strip()

    if any(w in msg_lower for w in ["hi", "hello", "hey", "greet", "greetings"]):
        reply_text = (
            f"Hello {name}! {time_greeting}.\n"
            f"Great to see you today. As an esteemed {job} in {dept}, how is your work progressing?"
        )
    elif "status" in msg_lower or "system" in msg_lower:
        reply_text = (
            f"All systems are operating at peak performance! Face recognition models (YuNet + SFace) "
            f"are active and your profile is loaded."
        )
    elif "job" in msg_lower or "role" in msg_lower or "who am i" in msg_lower:
        reply_text = (
            f"You are recognized as {name}, working as a {job} in {dept}."
        )
    elif "bye" in msg_lower or "see you" in msg_lower:
        reply_text = f"Goodbye {name}! Have a fantastic rest of your day."
    else:
        reply_text = (
            f"Thanks for sharing that, {name}. As a {job}, that's an interesting perspective. "
            f"Is there anything specific in {dept} you'd like assistance with?"
        )

    history.append(("user", user_input))
    history.append(("assistant", reply_text))
    return reply_text

def auto_greet(profile: dict) -> str:
    """Generate a first greeting based on profile data."""
    time_of_day = datetime.now().strftime("%H")
    hour = int(time_of_day)
    greeting_time = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    first_msg = (
        f"{greeting_time}! I've just recognized you. "
        f"Please greet me warmly and say something helpful or interesting "
        f"related to my work as a {profile.get('job', 'team member')}."
    )
    return chat_with_profile(profile, first_msg)

# ─────────────────────────────────────────────
# STEP 9 — Interactive Chat Loop
# ─────────────────────────────────────────────
def start_chat_session(profile: dict):
    """
    Full interactive chat session in the terminal.
    Type 'quit' to exit, 'history' to see chat log,
    'clear' to reset, 'profile' to see user details.
    """
    name = profile.get("name", "Guest")

    print(f"\n{'═'*55}")
    print(f"  💬  Chat session started — {name}")
    print(f"  🔑  Commands: 'quit' | 'history' | 'clear' | 'profile'")
    print(f"{'═'*55}\n")

    # Auto-greet on first recognition
    if not get_history(name):
        print("🤖 AI: ", end="", flush=True)
        greeting = auto_greet(profile)
        print(greeting)
        print()

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Session ended.")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() == 'quit':
            print(f"👋  Goodbye, {name}!")
            break
        elif user_input.lower() == 'history':
            show_history(name)
            continue
        elif user_input.lower() == 'clear':
            clear_history(name)
            print("🗑️  Your chat history has been cleared.")
            continue
        elif user_input.lower() == 'profile':
            print(f"\n👤  Profile: {json.dumps(profile, indent=2)}\n")
            continue

        # Normal chat
        print("🤖 AI: ", end="", flush=True)
        reply = chat_with_profile(profile, user_input)
        print(reply)
        print()

# ─────────────────────────────────────────────
# STEP 10 — MAIN: Run the full pipeline
# ─────────────────────────────────────────────
def run_face_chat():
    print("🚀  Starting Face Recognition Chat System...")
    print("─" * 45)

    # 1. Capture live photo
    img_path = capture_live_photo()
    if img_path is None:
        return

    # 2. Recognize face against history
    profile = recognize_face(img_path)
    if profile is None:
        return

    # 3. Start dynamic AI chat
    start_chat_session(profile)

if __name__ == "__main__":
    run_face_chat()