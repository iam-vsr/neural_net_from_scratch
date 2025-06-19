# 🧠 Neural Network From Scratch (NumPy Only)

> Build, train, and evaluate a neural network from the ground up — no frameworks, just pure NumPy and math.

---

## 📌 Project Overview

This project walks through the step-by-step process of building a **fully connected feedforward neural network from scratch**, using **only NumPy**.

We train it on the classic **MNIST handwritten digit** dataset to classify digits (0–9), and break down:

- Matrix-based forward propagation
- Activation functions (ReLU, Softmax)
- Manual backpropagation using chain rule
- Gradient descent-based weight updates
- One-hot encoding and prediction logic

---

## 🎯 Goals

- Understand how neural networks learn under the hood  
- Derive gradients and update rules from scratch  
- Train a model without relying on libraries like PyTorch or TensorFlow

---

## 🧱 Architecture

| Layer         | Size              | Activation |
|---------------|-------------------|------------|
| Input Layer   | 784 (28x28 pixels) | -          |
| Hidden Layer  | 128 neurons        | ReLU       |
| Output Layer  | 10 classes         | Softmax    |

---

📘 Read the full breakdown (math and intuition):  
[**neural_net_explained.md**](./neural_net_explained.md)

---

## 📦 Dataset Used

**MNIST** — 28x28 grayscale images of digits  
- 60,000 training images  
- 10,000 testing images  
- One-hot encoded labels

---

## 📈 Results

- 📊 **Final Test Accuracy**: ~`84.68%`  
- 🏃‍♂️ Training runs for `500 epochs`  
- 🧪 Optimized using plain gradient descent

---

## 🚀 Getting Started

No setup needed!

Click → [**Open in Colab**](https://colab.research.google.com/github/yourusername/NN_from_scratch/blob/main/NN_from_scratch.ipynb)  
Then click: `Runtime > Run all`

---








