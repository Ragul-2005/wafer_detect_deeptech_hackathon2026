import os
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, accuracy_score

# ---------------- CONFIG ----------------
ONNX_MODEL = "mobilenet_v2_wafer.onnx"
UNSEEN_DIR = "unseen"
IMG_SIZE = 224

class_names = ['CMP', 'LER', 'VIA', 'bridge', 'clean', 'crack', 'open']
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

# ---------------- LOAD ONNX MODEL ----------------
session = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ---------------- HELPERS ----------------
def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img[np.newaxis, np.newaxis, :, :]
    return img

# ---------------- INFERENCE LOOP ----------------
y_true, y_pred = [], []

for cls in class_names:
    cls_path = os.path.join(UNSEEN_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    for fname in os.listdir(cls_path):
        img_path = os.path.join(cls_path, fname)
        img = preprocess(img_path)
        if img is None:
            continue

        output = session.run(None, {input_name: img})[0]
        pred = np.argmax(output)

        y_true.append(class_to_idx[cls])
        y_pred.append(pred)

# ---------------- METRICS ----------------
acc = accuracy_score(y_true, y_pred)

print("\n🧪 ONNX MODEL — UNSEEN DATA EVALUATION\n")
print(f"✅ Overall Accuracy: {acc * 100:.2f}%\n")

print("Per-class performance:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))
