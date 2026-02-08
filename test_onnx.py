import onnxruntime as ort
import numpy as np
import cv2

# ---------------- CONFIG ----------------
ONNX_PATH = "mobilenet_v2_wafer.onnx"
IMG_PATH = r"C:\Users\DELL\Downloads\Wafer_Defect_ML-20260208T143031Z-1-001\Wafer_Defect_ML\test.png"
THRESHOLD = 0.6

class_names = ['CMP', 'LER', 'VIA', 'bridge', 'clean', 'crack', 'open']

# ---------------- LOAD IMAGE ----------------
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = (img - 0.5) / 0.5
img = img[np.newaxis, np.newaxis, :, :]

# ---------------- LOAD ONNX MODEL ----------------
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# ---------------- INFERENCE ----------------
logits = sess.run(None, {input_name: img})[0]

# softmax
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

max_prob = float(np.max(probs))
pred_idx = int(np.argmax(probs))

# ---------------- DECISION ----------------
if max_prob < THRESHOLD:
    final_class = "other"
    final_conf = 1.0
else:
    final_class = class_names[pred_idx]
    final_conf = max_prob

# ---------------- OUTPUT ----------------
print("\n🔍 ONNX Prediction Result")
print("--------------------------")
print(f"Final class      : {final_class}")
print(f"Final confidence : {final_conf * 100:.2f}%\n")

print("Class-wise probabilities (known classes only):")
for cls, p in zip(class_names, probs[0]):
    print(f"{cls:>8s} : {p * 100:.2f}%")

print(f"{'OTHER':>8s} : {100.00 if final_class == 'other' else 0.00:.2f}%")
