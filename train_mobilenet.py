import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- CONFIG ----------------
DATA_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ---------------- TRANSFORMS ----------------
train_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_tfms = train_tfms

# ---------------- DATASETS ----------------
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tfms)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tfms)
test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test", transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.classes
print("Classes:", class_names)

# ---------------- MODEL ----------------
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Modify first conv layer for 1-channel input
model.features[0][0] = nn.Conv2d(
    1, 32, kernel_size=3, stride=2, padding=1, bias=False
)

# Modify classifier
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

model = model.to(DEVICE)

# ---------------- TRAINING SETUP ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN LOOP ----------------
train_acc, val_acc = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_accuracy = correct / total
    train_acc.append(train_accuracy)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_accuracy = correct / total
    val_acc.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Acc: {train_accuracy:.3f} | Val Acc: {val_accuracy:.3f}")

# ---------------- TEST + CONFUSION MATRIX ----------------
y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        outputs = model(x)
        preds = outputs.argmax(1).cpu()
        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "mobilenet_v2_wafer.pth")
print("✅ Model saved as mobilenet_v2_wafer.pth")
