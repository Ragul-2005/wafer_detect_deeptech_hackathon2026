<h1>ML-Based Semiconductor Wafer Defect Detection</h1>

## 📌 Project Overview

This repository presents an end-to-end machine learning pipeline for semiconductor wafer defect classification using SEM images, designed to enable automated and low-latency inspection in modern manufacturing environments ⚙️. Semiconductor fabrication processes generate massive volumes of high-resolution inspection data at multiple stages, where manual inspection and centralized analysis pipelines often struggle to scale due to latency, bandwidth, and infrastructure constraints 📉.

The objective of this project is to demonstrate how a lightweight deep learning model can accurately classify multiple wafer defect types while remaining suitable for edge deployment 🚀. The work focuses on custom dataset construction and preprocessing 🧪, defect reclassification 🧩, model training using transfer learning with MobileNetV2 🧠, quantitative evaluation on both test and completely unseen datasets 📊, and export of the trained model to ONNX format for edge compatibility 🔧. The trained model is validated using ONNX Runtime and is aligned for future deployment on edge-AI platforms such as NXP eIQ ⚡.
