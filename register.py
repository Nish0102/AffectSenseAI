import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedder = models.resnet18(pretrained=True)
embedder.fc = torch.nn.Identity()
embedder = embedder.to(DEVICE)
embedder.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(face_img):
    pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = embedder(tensor)
    return emb.squeeze().cpu().numpy()

# ── ask who is registering ────────────────────────────────
name = input("Enter your name: ").strip().lower()
print(f"Registering: {name}")
print("Move your head slightly — left, right, up, down, tilt")

mp_face  = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.7)
cap      = cv2.VideoCapture(0)

embeddings = []
count = 0
TARGET = 60  # collect 60 embeddings

while count < TARGET:
    ret, frame = cap.read()
    if not ret:
        break

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x  = max(0, int(bbox.xmin * w))
            y  = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face_crop = frame[y:y+bh, x:x+bw]
            if face_crop.size == 0:
                continue

            emb = get_embedding(face_crop)
            embeddings.append(emb)
            count += 1
            time.sleep(0.05)

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {name}... {count}/{TARGET}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Register", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ── save as a STACK not just average ─────────────────────
# keeping all 60 embeddings (not just mean)
# gives much better recognition across angles
os.makedirs("faces", exist_ok=True)
np.save(f"faces/{name}.npy", np.array(embeddings))
print(f"Saved → faces/{name}.npy ({len(embeddings)} embeddings)")
