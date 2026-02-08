<div align="center">

# 🔬 Edge-AI Defect Classification
### Semiconductor Wafer / Die Inspection System

[![Hackathon](https://img.shields.io/badge/i4C-DeepTech%20Hackathon-blue?style=for-the-badge)](https://github.com)
[![Phase](https://img.shields.io/badge/Phase-1-success?style=for-the-badge)](https://github.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai)

**A lightweight, edge-ready AI system for real-time semiconductor wafer defect classification**

[📌 Overview](#-overview) • [🧠 Architecture](#-system-architecture) • [📊 Results](#-results) • [⚡ Quick Start](#-quick-start)

</div>

## 📌 Overview

This repository presents an end-to-end Edge-AI pipeline for semiconductor wafer defect classification using SEM images, designed to support automated, low-latency inspection in smart manufacturing environments ⚙️. Semiconductor fabrication generates massive volumes of high-resolution inspection data across multiple process stages, where manual inspection and centralized analysis pipelines often struggle with scalability, latency, and infrastructure overhead 📉.

The objective of this project is to demonstrate how a lightweight deep learning model can accurately classify multiple wafer defect categories while remaining suitable for edge deployment 🚀. The work focuses on custom dataset engineering 🧪, defect reclassification 🧩, transfer learning using MobileNetV2 🧠, quantitative evaluation on both held-out test data and completely unseen samples 📊, and export of the trained model to ONNX for edge compatibility 🔧. The resulting model is validated using ONNX Runtime and is aligned for future deployment on Edge-AI platforms such as NXP eIQ ⚡.

---
