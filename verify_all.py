import requests
import base64
import sys

sys.stdout.reconfigure(encoding='utf-8')
base_url = 'http://127.0.0.1:8000'

print('=== 1. Checking System Status ===')
r = requests.get(f'{base_url}/api/status')
assert r.status_code == 200, f"Status check failed: {r.status_code}"
status = r.json()
print(f"Status: {status['status']}, YOLO: {status['yolo_ready']}, Face Engine: {status['face_engine_ready']}, Profiles: {status['profiles_count']}")

print('\n=== 2. Checking Registered Profiles ===')
r = requests.get(f'{base_url}/api/profiles')
assert r.status_code == 200, f"Profiles fetch failed: {r.status_code}"
profiles = r.json()['profiles']
for p in profiles:
    print(f"- {p['name']} ({p['job']} in {p['department']})")

print('\n=== 3. Testing YOLOv8 Object Detection on Office Desk ===')
with open('static/samples/office_desk.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

r = requests.post(f'{base_url}/api/analyze', json={
    'image_base64': b64,
    'conf_thresh': 0.35,
    'detect_objects': True,
    'detect_faces': True
})
assert r.status_code == 200, f"Analyze failed: {r.status_code}"
data = r.json()
print(f"Inference latency: {data['inference_ms']} ms")
print(f"Total detected objects: {len(data['objects'])}")
print(f"Object classes detected: {data['summary']['object_classes']}")

print('\n=== 4. Testing Face Recognition on Alice ===')
with open('static/samples/alice_test.jpg', 'rb') as f:
    b64_alice = base64.b64encode(f.read()).decode('utf-8')

r = requests.post(f'{base_url}/api/analyze', json={
    'image_base64': b64_alice,
    'conf_thresh': 0.35,
    'detect_objects': False,
    'detect_faces': True
})
assert r.status_code == 200, f"Alice face check failed: {r.status_code}"
alice_data = r.json()
face = alice_data['faces'][0]
print(f"Recognized Face: {face['name']} (Match: {int(face['confidence']*100)}%, Role: {face['job']})")

print('\n=== 5. Testing Face Recognition on Bob ===')
with open('static/samples/bob_test.jpg', 'rb') as f:
    b64_bob = base64.b64encode(f.read()).decode('utf-8')

r = requests.post(f'{base_url}/api/analyze', json={
    'image_base64': b64_bob,
    'conf_thresh': 0.35,
    'detect_objects': False,
    'detect_faces': True
})
assert r.status_code == 200, f"Bob face check failed: {r.status_code}"
bob_data = r.json()
face_bob = bob_data['faces'][0]
print(f"Recognized Face: {face_bob['name']} (Match: {int(face_bob['confidence']*100)}%, Role: {face_bob['job']})")

print('\n=== 6. Testing Conversational Chat Endpoint ===')
chat_payload = {
    'message': 'What objects are around me?',
    'profile': {'name': 'Alice Vance', 'job': 'Data Scientist', 'department': 'AI Research'},
    'detected_objects': ['laptop', 'cell phone', 'cup']
}
r = requests.post(f'{base_url}/api/chat', json=chat_payload)
assert r.status_code == 200, f"Chat failed: {r.status_code}"
print("Chat endpoint status: 200 OK")
print("Reply snippet:", r.json()['reply'][:150].replace('\n', ' '), "...")

print('\n=== 7. Testing Frontend HTML & Assets ===')
r_html = requests.get(base_url)
assert r_html.status_code == 200 and 'OmniVision' in r_html.text, "Index HTML check failed"
r_css = requests.get(f'{base_url}/static/style.css')
assert r_css.status_code == 200 and len(r_css.text) > 5000, "CSS check failed"
r_js = requests.get(f'{base_url}/static/app.js')
assert r_js.status_code == 200 and len(r_js.text) > 5000, "JS check failed"
print("HTML, CSS, and JS all serving properly!")

print('\n=========================================')
print('>>> ALL FILES & SERVICES RUNNING 100% <<<')
print('=========================================')
