# ─────────────────────────────────────────────────────────
# LIBRARIES
# ─────────────────────────────────────────────────────────

import cv2
# OpenCV — Open Source Computer Vision Library
# handles everything camera related:
# reading webcam frames, drawing boxes/text on screen,
# converting between color formats (BGR, RGB, grayscale)
# BGR is OpenCV's default color order (not RGB)

import mediapipe as mp
# MediaPipe — Google's framework for real-time perception
# we use it specifically for face detection here
# it gives us bounding boxes around faces in each frame
# you already know this one

import torch
import torch.nn as nn
# PyTorch — the core deep learning framework
# torch: tensor operations, GPU/CPU management
# torch.nn: building blocks for neural networks
# (Conv2d, Linear, ReLU, BatchNorm etc.)

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# torchvision — PyTorch's computer vision companion library
# transforms: image preprocessing pipeline
#   (resize, normalize, convert to tensor, augmentation)
# datasets: easy loading of standard datasets like FER2013
# DataLoader: batches your dataset, shuffles it,
#   feeds it to the model during training efficiently

from torch import optim
# optim — optimization algorithms
# Adam, SGD etc. — these are what actually update
# the filter weights during backpropagation

from PIL import Image
# Pillow — Python Imaging Library
# used to open, convert, and manipulate images
# OpenCV reads images as numpy arrays
# PyTorch needs PIL images before applying transforms
# so PIL acts as the bridge between the two

import numpy as np
# NumPy — numerical computing library
# arrays, matrix operations
# OpenCV frames are NumPy arrays under the hood

import matplotlib.pyplot as plt
# Matplotlib — plotting library
# we use it to graph loss and accuracy curves
# after training so you can see if the model learned

import os
# os — Python's built-in file system module
# used to navigate folders, list files,
# build file paths for the dataset
from torchvision import models

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

DATA_DIR   = "./fer2013"
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.is_available() checks if you have a GPU
# if yes → trains on GPU (fast)
# if no  → falls back to CPU (slower but works fine)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print(f"Training on: {DEVICE}")


# ─────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING + TRANSFORMS
# ─────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    # Compose chains multiple transforms into one pipeline
    # applied in order top to bottom

    transforms.Resize((64, 64)),
    # resize every image to 64x64 pixels
    # FER2013 images are 48x48, we upscale slightly

    transforms.Grayscale(num_output_channels=3),
    # FER2013 is grayscale (1 channel)
    # our CNN expects 3 channels (RGB)
    # this duplicates the grayscale channel 3 times

    transforms.RandomHorizontalFlip(),
    # data augmentation — randomly mirror the image
    # a happy face flipped horizontally is still happy
    # doubles effective dataset size

    transforms.RandomRotation(10),
    # data augmentation — rotate up to 10 degrees
    # makes model robust to slightly tilted faces

    transforms.ToTensor(),
    # converts PIL image to PyTorch tensor
    # also scales pixel values from 0-255 → 0.0-1.0

    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # normalize pixel values to range -1.0 to 1.0
    # formula: (pixel - mean) / std
    # centered data trains faster and more stably
])

test_transforms = transforms.Compose([
    # no augmentation for test data
    # we want consistent evaluation, not random flips
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(DATA_DIR + "/train", transform=train_transforms)
test_data  = datasets.ImageFolder(DATA_DIR + "/test",  transform=test_transforms)
# ImageFolder automatically reads your folder structure
# fer2013/train/happy/ → label: "happy"
# fer2013/train/sad/   → label: "sad"
# no manual labeling needed

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)
# DataLoader groups images into batches of 32
# shuffle=True for training → randomizes order each epoch
# shuffle=False for testing → consistent evaluation

print(f"Classes: {train_data.classes}")
print(f"Train: {len(train_data)} | Test: {len(test_data)}")


# ─────────────────────────────────────────────────────────
# STEP 2 — MODEL DEFINITION
# ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    # nn.Module is the base class for all PyTorch models
    # every custom model or layer inherits from it
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # Sequential runs layers in order, output of one
            # feeds directly into the next

            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            # the actual convolution
            # in_ch: how many channels coming in
            # out_ch: how many filters (feature maps going out)
            # kernel_size=3: each filter is 3x3
            # padding=1: adds 1 pixel border so output
            #   stays same spatial size as input

            nn.BatchNorm2d(out_ch),
            # normalizes the output of conv layer
            # keeps values in a stable range during training
            # prevents exploding/vanishing gradients
            # makes training significantly faster

            nn.ReLU(),
            # activation function
            # negative values → 0, positive → unchanged
            # introduces non-linearity so network can
            # learn complex patterns, not just linear ones

            nn.MaxPool2d(2, 2)
            # takes max value in each 2x2 region
            # halves spatial dimensions (112→56, 56→28 etc.)
            # makes features position-invariant
            # reduces computation for next layer
        )

    def forward(self, x):
        return self.block(x)
        # forward() defines what happens when data
        # passes through this block


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Load pretrained ResNet18
        self.base = models.resnet18(pretrained=True)

        # Freeze early layers (they already learned useful features)
        for param in self.base.parameters():
            param.requires_grad = False

        # Replace final layer
        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Custom heads (same as your design)
        self.head_emotion = nn.Linear(512, num_classes)
        self.head_intensity = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.head_valence = nn.Linear(512, 2)

    def forward(self, x):
        x = self.base(x)
        return self.head_emotion(x), \
               self.head_intensity(x), \
               self.head_valence(x)


model = EmotionCNN(num_classes=len(train_data.classes)).to(DEVICE)
# .to(DEVICE) moves the model to GPU if available
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────────────────────
# STEP 3 — TRAINING
# ─────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
# loss function for classification
# compares predicted class scores vs true label
# returns a single number — how wrong the model is

optimizer = optim.Adam(model.parameters(), lr=LR)
# Adam — adaptive learning rate optimizer
# adjusts how much each weight updates automatically
# generally the best default choice for CNNs

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# reduces learning rate by half every 5 epochs
# early epochs: large steps to learn fast
# later epochs: small steps to fine-tune precisely

train_losses, train_accs, test_accs = [], [], []

for epoch in range(EPOCHS):
    model.train()
    # puts model in training mode
    # enables Dropout and BatchNorm training behavior

    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        # clear gradients from previous batch
        # PyTorch accumulates gradients by default
        # must reset each iteration

        emotion_out, intensity_out, valence_out = model(images)
        # forward pass — data flows through all layers

        # for FER2013 we only have emotion labels
        # intensity and valence heads will train properly
        # once we have richer labeled data
        loss = criterion(emotion_out, labels)

        loss.backward()
        # backpropagation — calculates gradient of loss
        # with respect to every weight in the network
        # this is where filters actually learn

        optimizer.step()
        # updates all weights using the gradients
        # one step of gradient descent

        total_loss += loss.item()
        preds = emotion_out.argmax(dim=1)
        # argmax picks the highest scoring class
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total * 100
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # evaluation
    model.eval()
    # switches off Dropout and BatchNorm training mode
    # for consistent inference

    correct, total = 0, 0
    with torch.no_grad():
        # no_grad — don't track gradients during evaluation
        # saves memory and speeds up inference
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            emotion_out, _, _ = model(images)
            preds = emotion_out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total * 100
    test_accs.append(test_acc)
    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Train: {train_acc:.1f}% | "
          f"Test: {test_acc:.1f}%")

torch.save(model.state_dict(), "emotion_cnn.pth")
# state_dict = all the learned weights
# save after training so you don't retrain every time
print("Model saved → emotion_cnn.pth")


# ─────────────────────────────────────────────────────────
# STEP 4 — PLOT TRAINING RESULTS
# ─────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses)
ax1.set_title("Loss over epochs")
ax1.set_xlabel("Epoch")
ax2.plot(train_accs, label="Train")
ax2.plot(test_accs, label="Test")
ax2.set_title("Accuracy over epochs")
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.savefig("training_results.png")
plt.show()


# ─────────────────────────────────────────────────────────
# STEP 5 — REAL TIME: FACE ID + EMOTION
# ─────────────────────────────────────────────────────────

import torch.nn.functional as F

# load emotion model
model.load_state_dict(torch.load("emotion_cnn.pth"))
model.eval()

# load face embedder (same ResNet18 extractor)
embedder = models.resnet18(pretrained=True)
embedder.fc = torch.nn.Identity()
embedder = embedder.to(DEVICE)
embedder.eval()

# load your saved face embedding
dark_embedding = np.load("dark_embedding.npy")
dark_tensor    = torch.tensor(dark_embedding).to(DEVICE)

THRESHOLD = 0.75
# cosine similarity above this = it's you
# below this = unknown face
# 0.75 is a good starting point, adjust if needed

def get_embedding(face_img):
    pil    = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = embedder(tensor)
    return emb.squeeze()

inference_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

mp_face  = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.7)
cap      = cv2.VideoCapture(0)

print("Webcam running — press Q to quit")

while True:
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

            # ── who is this? ──────────────────────────────
            live_emb  = get_embedding(face_crop)
            similarity = F.cosine_similarity(
                live_emb.unsqueeze(0),
                dark_tensor.unsqueeze(0)
            ).item()
            # cosine similarity measures angle between
            # two vectors — 1.0 = identical, 0.0 = unrelated
            # if your embedding and live face point in the
            # same direction in 512-dim space = same person

            name  = "Dark" if similarity > THRESHOLD else "Unknown"
            color = (0, 255, 0) if name == "Dark" else (0, 0, 255)
            # green for you, red for unknown

            # ── what emotion? ─────────────────────────────
            pil_img = Image.fromarray(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            )
            tensor  = inference_transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emotion_out, intensity_out, _ = model(tensor)
                class_idx  = emotion_out.argmax(dim=1).item()
                confidence = torch.softmax(emotion_out, dim=1).max().item()
                intensity  = intensity_out.item()

            emotion = EMOTIONS[class_idx]

            # ── draw on screen ────────────────────────────
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)

            # name above the box
            cv2.putText(frame, name,
                       (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # emotion below the name
            cv2.putText(frame, f"{emotion} {confidence*100:.0f}%",
                       (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            print({
                "name":      name,
                "similarity": round(similarity, 2),
                "emotion":   emotion,
                "confidence": round(confidence*100, 1),
                "intensity": round(intensity, 2)
            })

    cv2.imshow("Home AI — Identity + Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
