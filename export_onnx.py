import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "mobilenet_v2_wafer.pth"
ONNX_PATH = "mobilenet_v2_wafer.onnx"
NUM_CLASSES = 7

# Load model
model = models.mobilenet_v2(weights=None)

# First conv for grayscale
model.features[0][0] = nn.Conv2d(
    1, 32, kernel_size=3, stride=2, padding=1, bias=False
)

# Classifier
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 1, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    do_constant_folding=True
)

print("✅ ONNX model exported:", ONNX_PATH)
