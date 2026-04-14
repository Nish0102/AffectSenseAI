import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import time

# we use ResNet18 as a feature extractor
# remove its final classification layer
# what's left outputs a 512-dim face embedding
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedder = models.resnet18(pretrained=True)
embedder.fc = torch.nn.Identity()
# Identity() means "don't do anything"
# just pass the 512-dim vector straight through
# this turns ResNet into a face feature extractor
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

mp_face  = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.7)
cap      = cv2.VideoCapture(0)

embeddings = []
print("Look at the camera from different angles...")
print("Collecting 60 frames — move your head slightly left, right, up, down")

count = 0
while count < 60:
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
            time.sleep(0.1)

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing... {count}/60",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Register Face — Dark", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# average all 60 embeddings into one reference vector
# averaging makes it robust to lighting and angle variation
mean_embedding = np.mean(embeddings, axis=0)
np.save("dark_embedding.npy", mean_embedding)
print(f"Done — dark_embedding.npy saved ({len(embeddings)} frames captured)")
