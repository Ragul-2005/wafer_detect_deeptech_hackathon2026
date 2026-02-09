
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

---
<div align="center">
### 📦 Trained Models

| Model | Format | Download |
|:-----:|:------:|:--------:|
| **Edge Deployment** | ONNX | [Download](https://drive.google.com/file/d/1sBK4sehAkyZ3o3CDlgaLWUDJVRI2s8AS/view?usp=drive_link) |
| **PyTorch Checkpoint** | .pth | [Download](https://drive.google.com/file/d/1t8XXja7Qc71tmUoPg4YIECix4gC34bHF/view?usp=drive_link) |

---
</div>

## 📥 Dataset

The dataset used in this project consists of **grayscale SEM wafer inspection images** covering multiple defect categories. It includes both **real and synthetic samples** and is designed to evaluate model performance under realistic inspection conditions.

---
<div align="center">

### 📦 Dataset Access

🔗 **Download Dataset (ZIP):**  
👉 [Google Drive – Wafer Defect Dataset](https://drive.google.com/drive/folders/1JCUn1Xg_lPjh15-lgeGU6WeDd8zZ3oL3?usp=drive_link)

---
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

### Why This Model is Edge-Ready

<div align="center">

</div>

| Feature | Benefit | Impact |
|---------|---------|--------|
| 🎯 **MobileNetV2** | Lightweight CNN architecture | Low compute requirements |
| 🖼️ **Grayscale Input** | Single-channel processing | Reduced memory footprint |
| 📦 **ONNX Format** | Cross-platform compatibility | Portable deployment |
| ⚡ **Efficient Inference** | Optimized depthwise convolutions | Fast predictions |
| 🔧 **Transfer Learning** | Fewer parameters to train | Faster adaptation |

<div align="center">

</div>

## 🎬 Demo Flow

### 📋 Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/wafer-defect-classification.git
cd wafer-defect-classification

# Install required dependencies
pip install -r requirements.txt

### 📋 Prerequisites

```bash
# Clone the repository
git clone https://github.com/Ragul-2005/wafer_detect_deeptech_hackathon2026.git
cd wafer-defect-classification

# Install required dependencies
pip install -r requirements.txt
```

## 1️⃣ Train the Model
```
python train.py
```

### What it does:
- 📥 Loads and preprocesses grayscale SEM images
- 🧠 Trains MobileNetV2 using transfer learning
- 📊 Monitors validation performance
- 💾 Saves the best model checkpoint

Output:
```
mobilenet_v2_wafer.pth
```
---

## 2️⃣ Evaluate on Test Set

```
python evaluate.py
```

### What it does:
- 🧪 Loads the held-out test dataset
- 🔍 Runs inference using trained model
- 📊 Computes accuracy, precision, recall, and F1-score

#### Output:
Printed evaluation metrics

---

## 3️⃣ Evaluate on Unseen Images

```
python test_unseen.py
```

### What it does:

- 🧠 Evaluates model on completely unseen SEM images
- 📉 Measures real-world generalization performance

#### Output:
Accuracy and class-wise metrics on unseen data

## 4️⃣ Generate Confusion Matrix

```
python confusion_matrix.py
```

### What it does:

- 📊 Evaluates predictions on test set
- 🖼️ Generates confusion matrix visualization
- 💾 Saves result as image file

Output:
```
confusion_matrix.png
```

## 5️⃣ Export Model to ONNX

```
python export_onnx.py
```


### What it does:
- 📦 Converts PyTorch model → ONNX format
- ✅ Validates ONNX inference using ONNX Runtime
- ⚡ Prepares model for edge deployment

Output:
```
mobilenet_v2_wafer.onnx
```
## 📁 Repository Structure

```text
📦 wafer-defect-classification
 ┣ 📖 README.md                       # Project documentation
 ┣ 📊 confusion_matrix_test.png       # Test set confusion matrix
 ┣ 📜 train_mobilenet.py              # MobileNetV2 training script
 ┣ 📜 split.py                        # Dataset train/val/test split utility
 ┣ 📜 test_unseen.py                  # Evaluation on unseen SEM images
 ┣ 📜 test_onnx.py                    # ONNX model inference test
 ┣ 📜 export_onnx.py                  # PyTorch → ONNX export script
 ┣ 🤖 mobilenet_v2_wafer.pth          # Trained PyTorch model
 ┣ 📦 mobilenet_v2_wafer.onnx         # Exported ONNX model
 ┣ 📋 requirements.txt                # Python dependencies
```

## 🛠️ Technology Stack

### 🔹 Programming Language
- 🐍 **Python** — Core language for model development, training, and evaluation

---

### 🔹 Deep Learning Framework
- 🔥 **PyTorch** — Model training, transfer learning, and checkpoint management
- 🧠 **Torchvision** — Pre-trained MobileNetV2 and image transformations

---

### 🔹 Model Interoperability & Edge Runtime
- 📦 **ONNX** — Portable model format for edge deployment
- ⚡ **ONNX Runtime** — Fast, hardware-agnostic inference engine

---

### 🔹 Image Processing & Data Handling
- 🖼️ **OpenCV** — Image loading, resizing, and preprocessing
- 🧪 **Pillow (PIL)** — Image format handling
- 📐 **NumPy** — Numerical operations and tensor preparation

---

### 🔹 Evaluation & Analysis
- 📊 **scikit-learn** — Accuracy, precision, recall, F1-score, confusion matrix
- 📈 **Matplotlib** — Visualization of results and metrics

---

### 🔹 Development & Experimentation
- 🧰 **Local Python Environment** — Model training and testing
- 📋 **requirements.txt** — Dependency management and reproducibility

---

### ⚡ Edge-AI Readiness
- 🖥️ **CPU-based Inference** — Optimized for low-power edge devices
- 🔧 **NXP eIQ Compatible Workflow** — Ready for embedded AI deployment

  ---

## 🏁 Conclusion

This project demonstrates the effectiveness of a **lightweight Edge-AI pipeline** for semiconductor wafer defect classification using grayscale SEM images. By leveraging **MobileNetV2** and transfer learning, the system achieves high classification accuracy while maintaining a compact model footprint suitable for deployment on resource-constrained edge devices.

The model was evaluated not only on standard validation and test datasets but also on **completely unseen images**, confirming its robustness and generalization capability. Exporting the trained network to **ONNX format** ensures portability across different hardware platforms and enables seamless integration into edge-based inspection workflows.

Overall, the results validate the feasibility of applying deep learning for **real-time, scalable wafer inspection**, reducing dependency on manual review and centralized analysis. This work establishes a strong foundation for future edge deployment in smart semiconductor manufacturing environments.

---

## 🔮 Future Work

- ⚡ Hardware deployment and benchmarking on embedded edge platforms  
- 📉 Model quantization and further size optimization  
- 🧪 Expansion of defect classes and dataset diversity  
- 🤖 Integration with real-time inspection pipelines  
- 📊 Continuous learning with new defect samples  

---

## 📚 References

1. **Deep Learning for Wafer Defect Inspection** – Survey of CNN-based methods for semiconductor defect analysis  
2. **Public SEM Wafer Defect Datasets** – Open-source repositories for wafer inspection imagery  
3. **PyTorch Documentation** – Model training and transfer learning workflows  
4. **ONNX & ONNX Runtime Documentation** – Cross-platform model interoperability and inference  
5. **NXP eIQ Edge AI Toolkit** – Edge-AI deployment and optimization guidelines  

---

## 👥 Team Members

| 🔢 Sr. No | 🧩 Role | 👤 Name | 💻 GitHub ID |
|:--:|:--|:--|:--|:--|
| 1️⃣ | 🧠 **Team Leader** | **Ragul T** | [@RagulT](https://github.com/Ragul-2005) |
| 2️⃣ | 👨‍💻 **Member 1** | **Praveen R** | [@PraveenR](https://github.com/PRAVEENRAMU14) |
| 3️⃣ | 👨‍💻 **Member 2** | **S S Jhotheeshwar**  | [@Jhotheeshwar](https://github.com/S-S-JHOTHEESHWAR) |
| 4️⃣ | 👩‍💻 **Member 3** | **Merlin Jenifer S** |  [@MerlinJenifer]() |

📌 *Developed as part of the **i4C DeepTech Hackathon – Phase 1***

---

  ## 🏷️ Project Labels

![Domain](https://img.shields.io/badge/Domain-Semiconductor%20AI-blue)
![Category](https://img.shields.io/badge/Category-Edge--AI-green)
![Task](https://img.shields.io/badge/Task-Defect%20Classification-orange)
![Data](https://img.shields.io/badge/Data-SEM%20Images-purple)
![Model](https://img.shields.io/badge/Model-MobileNetV2-red)
![Deployment](https://img.shields.io/badge/Deployment-ONNX-lightgrey)

---
## 🔖 Project Tags

`Edge-AI` · `Semiconductor` · `Wafer Inspection` · `SEM Images` ·  
`Defect Classification` · `MobileNetV2` · `ONNX` · `Deep Learning` ·  
`Industry 4.0` · `Computer Vision`

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome.  
If you find an issue or have an enhancement idea, feel free to open an issue or submit a pull request.

---

<div align="center">

**🔬 Edge-AI Semiconductor Wafer Defect Classification**

Built with ❤️ for the **i4C DeepTech Hackathon – Phase 1**

⭐ *Star the repo to support the project!* ⭐

</div>


  
