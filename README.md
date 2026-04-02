# 🎨 CycleGAN: Sketch to Photo Translation

[![Streamlit App](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/mustehsannisarrao/cyclegan-sketch-to-photo-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **⚠️ Note:** Results are not production-ready. This project focuses on **understanding CycleGAN implementation**, not achieving state-of-the-art results.

Convert hand-drawn sketches into photos using **CycleGAN** - trained on Sketchy dataset for 50 epochs.

## ✨ Features

- ✏️ **Sketch → Photo Translation**
- 🎨 **Draw directly in browser** (canvas support!)
- 🔄 **Photo → Sketch Translation** (reverse mapping)
- 📱 Mobile responsive
- 🚀 Real-time inference

## 🎯 **Live Demo**

**Try it yourself:** [Hugging Face Space](https://huggingface.co/spaces/mustehsannisarrao/cyclegan-sketch-to-photo-demo)

> 💡 **Pro tip:** You can either upload a sketch OR draw directly using the canvas!

## ⚠️ **Results Disclaimer**

The generated photos are **not realistic**. This is expected because:

| Limitation | Reason |
|------------|--------|
| 50 epochs only | Limited compute (T4 GPU free tier) |
| 6 ResNet blocks | Standard CycleGAN uses 9 blocks |
| 2,000 samples | Full dataset has 75,000+ sketches |
| No hyperparameter tuning | Focus on implementation |

**Objective:** Understand CycleGAN architecture, loss functions, and training dynamics — NOT achieve SOTA results.

## 🏗️ Architecture

### Generator (Simplified - 6 ResNet blocks)
```python
- 2 downsampling conv layers
- 6 residual blocks  
- 2 upsampling conv layers
- Output: 128×128 RGB
