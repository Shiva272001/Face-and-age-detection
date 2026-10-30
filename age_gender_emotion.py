# -*- coding: utf-8 -*-
"""
A Gender, Age and Facial Expression Detection Program
Integrated with OpenCV DNN (supports both ONNX and Caffe/TensorFlow models).
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Set UTF-8 encoding for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ==========================================================
# ARGUMENT PARSER
# ==========================================================
parser = argparse.ArgumentParser(description="Gender, Age and Facial Expression Detection")
parser.add_argument('--image', '-i', type=str, help='Path to input image file')
parser.add_argument('--device', '-d', type=int, default=0, help='Webcam index (default 0)')
args, _ = parser.parse_known_args()

# ==========================================================
# LABELS & CONSTANTS
# ==========================================================
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = [
    '(0-2)',
    '(4-6)',
    '(8-12)',
    '(15-20)',
    '(25-32)',
    '(38-43)',
    '(48-53)',
    '(60-100)'
]

genderList = [
    'Male',
    'Female'
]

emotionList = [
    'Neutral',
    'Happiness',
    'Surprise',
    'Sadness',
    'Anger',
    'Disgust',
    'Fear',
    'Contempt'
]

# ==========================================================
# SOFTMAX & UTILITIES
# ==========================================================
def softmax(scores):
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

def getSquareFace(frame, faceBox, padding=10):
    x1, y1, x2, y2 = faceBox
    width = x2 - x1
    height = y2 - y1

    size = max(width, height) + (padding * 2)
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2

    newX1 = max(0, centerX - size // 2)
    newY1 = max(0, centerY - size // 2)
    newX2 = min(frame.shape[1], newX1 + size)
    newY2 = min(frame.shape[0], newY1 + size)

    return frame[newY1:newY2, newX1:newX2]

def highlightFace(net, frame, conf_threshold=0.6):
    frameHeight, frameWidth = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * frameWidth))
            y1 = max(0, int(detections[0, 0, i, 4] * frameHeight))
            x2 = min(frameWidth - 1, int(detections[0, 0, i, 5] * frameWidth))
            y2 = min(frameHeight - 1, int(detections[0, 0, i, 6] * frameHeight))

            if x2 > x1 and y2 > y1:
                faceBoxes.append([x1, y1, x2, y2])
    return faceBoxes

# ==========================================================
# LOAD MODELS (WITH AUTO-DISCOVERY & ONNX SUPPORT)
# ==========================================================
print("⏳ Loading models...")

# 1. Face Detector
faceNet = None
pb_path = os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb")
pbtxt_path = os.path.join(MODELS_DIR, "opencv_face_detector.pbtxt")
if os.path.exists(pb_path) and os.path.exists(pbtxt_path):
    try:
        faceNet = cv2.dnn.readNet(pb_path, pbtxt_path)
    except Exception as e:
        print(f"⚠️ Face detector PB failed: {e}")

# Fallback to YuNet if PB not available
yunet_detector = None
yunet_path = os.path.join(MODELS_DIR, "yunet.onnx")
if faceNet is None and os.path.exists(yunet_path):
    try:
        yunet_detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
        print("✅ Using YuNet ONNX for face detection.")
    except Exception as e:
        print(f"⚠️ YuNet failed: {e}")

# 2. Age Model
ageNet = None
age_onnx = os.path.join(MODELS_DIR, "age_googlenet.onnx")
age_caffe = os.path.join(MODELS_DIR, "age_net.caffemodel")
age_proto = os.path.join(MODELS_DIR, "age_deploy.prototxt")

if os.path.exists(age_onnx):
    try:
        ageNet = cv2.dnn.readNet(age_onnx)
        print("✅ Age model loaded (ONNX).")
    except Exception as e:
        print(f"⚠️ Age ONNX failed: {e}")

if ageNet is None and os.path.exists(age_caffe) and os.path.exists(age_proto):
    try:
        ageNet = cv2.dnn.readNet(age_caffe, age_proto)
        print("✅ Age model loaded (Caffe).")
    except Exception as e:
        print(f"⚠️ Age Caffe failed: {e}")

# 3. Gender Model
genderNet = None
gender_onnx = os.path.join(MODELS_DIR, "gender_googlenet.onnx")
gender_caffe = os.path.join(MODELS_DIR, "gender_net.caffemodel")
gender_proto = os.path.join(MODELS_DIR, "gender_deploy.prototxt")

if os.path.exists(gender_onnx):
    try:
        genderNet = cv2.dnn.readNet(gender_onnx)
        print("✅ Gender model loaded (ONNX).")
    except Exception as e:
        print(f"⚠️ Gender ONNX failed: {e}")

if genderNet is None and os.path.exists(gender_caffe) and os.path.exists(gender_proto):
    try:
        genderNet = cv2.dnn.readNet(gender_caffe, gender_proto)
        print("✅ Gender model loaded (Caffe).")
    except Exception as e:
        print(f"⚠️ Gender Caffe failed: {e}")

# 4. Emotion Model
emotionNet = None
emotion_onnx = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")
if os.path.exists(emotion_onnx):
    try:
        emotionNet = cv2.dnn.readNet(emotion_onnx)
        print("✅ Emotion model loaded (ONNX).")
    except Exception as e:
        print(f"⚠️ Emotion ONNX failed: {e}")

print("✨ All models ready!\n")

def detect_faces(frame):
    """Detects face boxes using available face network."""
    if faceNet is not None:
        return highlightFace(faceNet, frame, conf_threshold=0.6)
    elif yunet_detector is not None:
        h, w = frame.shape[:2]
        yunet_detector.setInputSize((w, h))
        faces = yunet_detector.detect(frame)[1]
        if faces is not None and len(faces) > 0:
            boxes = []
            for f in faces:
                x1, y1, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                x2, y2 = min(w - 1, x1 + bw), min(h - 1, y1 + bh)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
            return boxes
    return []

def predict_attributes(face_crop, frame, face_box):
    """Predicts Gender, Age, and Emotion for a detected face."""
    gender = "Unknown"
    gender_conf = 0.0
    age = "Unknown"
    age_conf = 0.0
    emotion = "Unknown"
    emotion_conf = 0.0

    # Age & Gender Blob
    if (genderNet is not None or ageNet is not None) and face_crop.size > 0:
        blob = cv2.dnn.blobFromImage(
            face_crop,
            1.0,
            (224, 224),
            MODEL_MEAN_VALUES,
            swapRB=False
        )

        if genderNet is not None:
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            genderIndex = int(genderPreds[0].argmax())
            gender = genderList[genderIndex]
            gender_conf = float(genderPreds[0][genderIndex] * 100)

        if ageNet is not None:
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            ageIndex = int(agePreds[0].argmax())
            age = ageList[ageIndex]
            age_conf = float(agePreds[0][ageIndex] * 100)

    # Emotion Prediction
    if emotionNet is not None:
        emotionFace = getSquareFace(frame, face_box, padding=10)
        if emotionFace.size > 0:
            grayFace = cv2.cvtColor(emotionFace, cv2.COLOR_BGR2GRAY)
            grayFace = cv2.resize(grayFace, (64, 64), interpolation=cv2.INTER_AREA)
            grayFace = cv2.equalizeHist(grayFace).astype(np.float32)

            emotionBlob = cv2.dnn.blobFromImage(
                grayFace,
                scalefactor=1.0,
                size=(64, 64),
                mean=(0,),
                swapRB=False,
                crop=False
            )
            emotionNet.setInput(emotionBlob)
            emotionPreds = emotionNet.forward()
            emotionScores = np.squeeze(emotionPreds)
            emotionProbabilities = softmax(emotionScores)
            emotionIndex = int(np.argmax(emotionProbabilities))
            emotion = emotionList[emotionIndex]
            emotion_conf = float(emotionProbabilities[emotionIndex] * 100)

    return {
        "gender": gender,
        "gender_conf": gender_conf,
        "age": age,
        "age_conf": age_conf,
        "emotion": emotion,
        "emotion_conf": emotion_conf
    }

# ==========================================================
# MAIN EXECUTION LOOP
# ==========================================================
def main():
    source = args.image if args.image else args.device
    video = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not video.isOpened():
        video = cv2.VideoCapture(source)

    if not video.isOpened():
        print(f"❌ Unable to open video source: {source}")
        return

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    detection_interval = 3
    last_results = []
    prev_time = time.time()
    fps = 0

    print("🚀 Video loop started. Press 'q' to exit.")

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            if args.image:
                # Wait until key press on still image
                cv2.waitKey(0)
            break

        frame_count += 1

        if frame_count % detection_interval == 0:
            faceBoxes = detect_faces(frame)
            last_results = []

            for faceBox in faceBoxes:
                x1 = max(0, faceBox[0] - 20)
                y1 = max(0, faceBox[1] - 20)
                x2 = min(frame.shape[1], faceBox[2] + 20)
                y2 = min(frame.shape[0], faceBox[3] + 20)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                attrs = predict_attributes(face, frame, faceBox)
                last_results.append((faceBox, attrs))

                print("\n-----------------------------")
                print(f"Gender     : {attrs['gender']} ({attrs['gender_conf']:.1f}%)")
                print(f"Age        : {attrs['age']} ({attrs['age_conf']:.1f}%)")
                print(f"Expression : {attrs['emotion']} ({attrs['emotion_conf']:.1f}%)")
                print("-----------------------------")

        # Draw results
        for (faceBox, attrs) in last_results:
            x1, y1, x2, y2 = faceBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_top = f"{attrs['gender']} | Age: {attrs['age']}"
            cv2.putText(frame, label_top, (x1, max(25, y1 - 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            label_bot = f"{attrs['emotion']} ({attrs['emotion_conf']:.0f}%)"
            cv2.putText(frame, label_bot, (x1, max(50, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

        # FPS calculation
        curr_time = time.time()
        diff = curr_time - prev_time
        if diff > 0:
            curr_fps = 1 / diff
            fps = curr_fps if fps == 0 else 0.9 * fps + 0.1 * curr_fps
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Age Gender and Expression Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
