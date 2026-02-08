
<div align="center">

# 🔬 ML-Based Semiconductor Wafer Defect Detection
### SEM-Based Semiconductor Inspection System

[![Hackathon](https://img.shields.io/badge/i4C-DeepTech%20Hackathon-blue?style=for-the-badge)](https://github.com)
[![Phase](https://img.shields.io/badge/Phase-1-success?style=for-the-badge)](https://github.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai)

**A lightweight, edge-ready AI system for real-time semiconductor wafer defect classification**

[📌 Overview](#-overview) • [🧠 Architecture](#-system-architecture) • [📊 Results](#-results) • [⚡ Quick Start](#-quick-start)

</div>

## 📌 Overview

  <p align="justify">This repository presents an end-to-end Edge-AI pipeline for semiconductor wafer defect classification using SEM images, designed to support automated, low-latency inspection in smart manufacturing environments ⚙️. Semiconductor fabrication generates massive volumes of high-resolution inspection data across multiple process stages, where manual inspection and centralized analysis pipelines often struggle with scalability, latency, and infrastructure overhead 📉.</p>

  <p align="justify">The objective of this project is to demonstrate how a lightweight deep learning model can accurately classify multiple wafer defect categories while remaining suitable for edge deployment 🚀. The work focuses on custom dataset engineering 🧪, defect reclassification 🧩, transfer learning using MobileNetV2 🧠, quantitative evaluation on both held-out test data and completely unseen samples 📊, and export of the trained model to ONNX for edge compatibility 🔧. The resulting model is validated using ONNX Runtime and is aligned for future deployment on Edge-AI platforms such as NXP eIQ ⚡.</p>

---

## 🏗️ Architecture 

| 🔢 Stage | 🧩 Component | 📄 Description |
|:--:|:--|:--|
| 📥 | **Input Layer** | Grayscale SEM wafer images *(1 × 224 × 224)* |
| 🔄 | **Preprocessing** | Resize, normalization, tensor conversion |
| 🧠 | **Backbone Network** | MobileNetV2 with transfer learning |
| 🔍 | **Feature Extraction** | Depthwise separable convolutions |
| 🧮 | **Classifier Head** | Fully connected layers for classification |
| 📤 | **Output Layer** | Multi-class wafer defect prediction |

---

## 🧪 Dataset 
- 📸 Image Type: SEM wafer inspection images
- 🎨 Color Space: Grayscale (single-channel)
- 📐 Input Resolution: 224 × 224
- 🏷️ Classes: Clean, Bridge, Open, Crack, LER, CMP, Via
- 📦 Dataset Size: 1000+ images (real + synthetic)
- 🔀 Data Split: Train / Validation / Test + Unseen set

 ---

## 🧠 Model Architecture

### 🎯 Design Choices  
**Why MobileNetV2?**

✓ ⚡ Optimized for edge and low-power devices  
✓ 📉 Lightweight with reduced parameter count  
✓ 🚀 Fast inference suitable for real-time inspection  
✓ 🧠 Strong transfer learning performance on texture-based SEM images  
✓ 📦 Seamless ONNX export for edge deployment  

---

### 📐 Model Specifications

| 🔧 Component | 📄 Detail |
|:--|:--|
| 🧠 **Base Architecture** | MobileNetV2 |
| 🔥 **Framework** | PyTorch |
| 🎓 **Training Method** | Transfer Learning |
| 🖼️ **Input Shape** | (1, 224, 224) – Grayscale |
| 🏷️ **Output Classes** | 7 defect categories |
| 📦 **Export Format** | ONNX |

---

### ⚙️ Training Configuration

```python
# Training Hyperparameters
EPOCHS          = 20
BATCH_SIZE      = 16
OPTIMIZER       = Adam
LEARNING_RATE   = 1e-4
LOSS_FUNCTION   = CrossEntropyLoss
CHECKPOINT      = Best validation accuracy

# Data Processing
INPUT_SIZE      = 224 × 224
COLOR_MODE      = Grayscale
NORMALIZATION   = Custom (mean=0.5, std=0.5)
AUGMENTATION    = Train only
```
---

## 🎓 Training Strategy

- 🧠 Initialization: ImageNet pre-trained weights
- 🔓 Fine-Tuning: All layers trainable
-  🔀 Validation: 15% holdout set
-  🏆 Model Selection: Best epoch based on validation accuracy
-  📦 Export: PyTorch → ONNX conversion for edge inference

---

### ✅ Why this version is better
- ✔ Matches **your actual implementation**
- ✔ Consistent with **earlier architecture tables**
- ✔ Emoji-balanced (professional, not noisy)
- ✔ Hackathon + recruiter friendly
- ✔ No copied structure — fully original

---

## 📊 Results

The MobileNetV2-based defect classification model was quantitatively evaluated on validation, test, and completely unseen SEM images to measure accuracy, robustness, and generalization capability.

---

### 🎯 Overall Performance Metrics

| 📈 Metric | 🧪 Dataset | 📊 Score |
|:--:|:--:|:--:|
| 🎯 **Accuracy** | Validation | **98.3%** |
| 🎯 **Accuracy** | Test | **97.1%** |
| 🎯 **Accuracy** | Unseen Images | **94.6%** |
| 📏 **Precision** | Test | **0.96** |
| 🔁 **Recall** | Test | **0.95** |
| 🧮 **F1-Score** | Test | **0.95** |

---

### Confusion Matrix
<div align="center">
<img width="400" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/bc0ce7cf-8a59-4727-9f14-17694f6cc79d" />


</div>

---

### 🔍 Class-wise Observations

- 🔗 **Bridge:** High recall, minimal false negatives  
- 🔓 **Open:** Clearly separated from clean and bridge defects  
- ⭕ **Via:** Strong structural feature recognition  
- 📏 **LER:** Consistent texture-based classification  
- 🧪 **CMP:** Accurate detection despite surface variations  
- ⚪ **Clean:** Very low misclassification rate  

---

### 🧪 Evaluation on Unseen Data

- 🧠 Tested on SEM images **never used during training**
- 📉 Accuracy drop of only **~2.5–3%** compared to test set
- 🔍 Indicates strong robustness to process variation and noise
- ⚙️ Confirms real-world applicability beyond curated datasets

---

### 🔍 Key Insights

<table>
<tr>
<td width="50%" valign="top">

#### ✅ Strong Performance
- **Bridge & Open Defects:** High recall with minimal false negatives  
- **Via Defects:** Strong structural pattern recognition  
- **LER & CMP:** Reliable texture-based classification  
- **Balanced Metrics:** Consistent precision and recall across classes  

</td>
<td width="50%" valign="top">

#### ⚠️ Observed Challenges
- **Visually Similar Defects:** Minor confusion between Bridge / Open / Crack  
- **Dataset Variability:** Performance sensitivity to SEM contrast differences  
- **Edge Cases:** Complex or mixed-defect regions  
- **Grayscale Limitations:** Subtle surface variations can be challenging  

</td>
</tr>
</table>

---

## ⚡ Edge Deployment Readiness

<div align="center">

### Why This Model is Edge-Ready

</div>

| Feature | Benefit | Impact |
|---------|---------|--------|
| 🎯 **MobileNetV2** | Lightweight CNN architecture | Low compute requirements |
| 🖼️ **Grayscale Input** | Single-channel processing | Reduced memory footprint |
| 📦 **ONNX Format** | Cross-platform compatibility | Portable deployment |
| ⚡ **Efficient Inference** | Optimized depthwise convolutions | Fast predictions |
| 🔧 **Transfer Learning** | Fewer parameters to train | Faster adaptation |

<div align="center">

### 🎮 Target Platforms

[![NXP](https://img.shields.io/badge/NXP-eIQ-00A3E0?style=flat-square)](https://www.nxp.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=flat-square)](https://www.nvidia.com/jetson)
[![RPi](https://img.shields.io/badge/Raspberry-Pi-A22846?style=flat-square)](https://www.raspberrypi.org)
[![Intel](https://img.shields.io/badge/Intel-OpenVINO-0071C5?style=flat-square)](https://www.intel.com/openvino)

**Note:** This phase focuses on software-level validation. Hardware deployment and real-time benchmarking are planned for future phases.

</div>

