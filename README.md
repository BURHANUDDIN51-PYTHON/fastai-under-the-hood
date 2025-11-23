# Fast.ai Deep Learning Implementation

This repository documents my journey through Part 1 of the Fast.ai course, focusing on implementing core algorithms from scratch to understand the math behind the "Black Box."

## 📂 Project 1: MNIST Digit Classifier (Computer Vision)
**File:** `01_mnist_tensors_from_scratch.ipynb`

A binary classifier (3s vs 7s) built using **raw PyTorch tensors**.
* **Goal:** Understand how neural networks learn without using `nn.Linear` or `optim`.
* **Tech:** PyTorch (Broadcasting, Matrix Multiplication), SGD implementation.

---

## 📂 Project 2: Titanic Survival Predictor (Tabular Data)
**File:** `02_titanic_tabular_from_scratch.ipynb`

A predictive model built on the famous Titanic dataset.
* **Goal:** Master data preprocessing and tabular model architecture.
* **Tech:** Pandas for data cleaning, categorical embeddings, and custom model logic.

---

## 📂 Project 3: ResNet Architecture from Scratch (Computer Vision)
**File:** `03_resnet_from_scratch.ipynb` (or similar file name)

An end-to-end reconstruction of the **ResNet-50** architecture using PyTorch and fastai primitives. This project focused on shifting from a model 'user' to a model 'builder' by implementing key architectural decisions:
* **Residual Blocks:** Implementing **identity skip connections** (the **x + F(x)** mechanism) to ensure uninterrupted gradient flow through deep networks.
* **Bottleneck Layers:** Developing the efficient **1×1 → 3×3 → 1×1** convolution pipeline to optimize computation and parameter count.
* **The Stem:** Implementing the initial downsampling layers to save memory and compute early in the forward pass.
* **Tech:** PyTorch `nn.Module` and `nn.Conv2d` primitives, custom weight initialization, and tensor shape tracking.

---
*Built as part of the Fast.ai Practical Deep Learning for Coders (Part 1).*
