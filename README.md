# 🧠 Build From Scratch - Part 1: Neural Network Using NumPy

This project implements a simple feedforward neural network **from scratch** using only `NumPy`, trained on the **MNIST handwritten digit dataset** — no deep learning frameworks used.

---

## 📌 Overview

| Feature         | Details                         |
|----------------|----------------------------------|
| Input Size      | 784 (28×28 grayscale image)      |
| Hidden Layer    | 128 neurons + ReLU               |
| Output Layer    | 10 neurons + Softmax             |
| Dataset         | MNIST (via Keras)                |
| Optimizer       | Gradient Descent                 |
| Final Accuracy  | ~% on test set                 |

---

## 🧠 Architecture & Math

### 🔹 Forward Propagation

```math
Z_1 = X \cdot W_1 + b_1 \\
A_1 = \text{ReLU}(Z_1) \\
Z_2 = A_1 \cdot W_2 + b_2 \\
A_2 = \text{Softmax}(Z_2)

