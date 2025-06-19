# 🧠 Neural Networks From Scratch - Full Math & Intuition Explained

## 📑 Table of Contents
- [Why Neural Networks](#-why-neural-networks)
- [Network Architecture](#-network-architecture)
- [Forward Propagation](#-forward-propagation)
- [Backpropagation](#-backpropagation)
- [Weight Updates](#-weight-updates)
- [Conclusion & Next Steps](#-conclusion--next-steps)


## 🤔 Why Neural Networks?

In the world of machine learning, traditional algorithms like logistic regression or decision trees are often limited by their **inability to capture complex patterns** in data.

But what if:
- The decision boundary isn’t linear?
- You’re working with raw pixels from images?
- You want a model that can **learn features automatically**?

This is where **Neural Networks (NNs)** shine.

### 🔍 Core Idea

A neural network is a **function approximator**.  
It tries to learn a mapping:
f(X) -> Y
...where \( X \) could be image pixels and \( Y \) could be class labels (like digits 0–9).

Rather than manually crafting features (as in classical ML), NNs **learn the representations directly from data** using a process called **backpropagation**.

---

## 🧱 Why Build One *From Scratch*?

Most people use frameworks like PyTorch or TensorFlow. But when you build one from the ground up:
- You gain a **clear understanding of how backpropagation works**
- You learn what **each layer actually does under the hood**
- You build **strong debugging and intuition skills**

This project uses only **NumPy** to build and train a neural network on the **MNIST digit dataset** — a classic problem in image classification.

> ✅ Goal: Learn how to implement and train a neural network by hand — one matrix at a time.

---

## 📦 Dataset Used: MNIST

- **28 × 28 grayscale images** of digits 0 through 9
- **60,000 training samples**, **10,000 test samples**
- Each image is flattened into a **784-dimensional vector**

---

## 🧱 Network Architecture

### 📐 Network Layout

We use a **simple 3-layer architecture** for classifying handwritten digits:

- **Input Layer**: 784 neurons (28x28 pixels, flattened grayscale image)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (for 10 digit classes)

---

### 🔄 Forward Propagation: Step by Step

Forward propagation is the process of passing input data through the layers to get the prediction.

Let:
- \( X \): Input matrix of shape (batch_size × 784)
- \( W1, b1 \): Weights and biases of layer 1
- \( A1 \): Activation output of hidden layer
- \( W2, b2 \): Weights and biases of layer 2
- \( A2 \): Final output probabilities

---

### 🔹 Step 1: Linear Transformation (Input → Hidden)

`Z₁ = X · W₁ + b₁`

- X:     Matrix of shape (batch_size x 784)  
- W₁:    Matrix of shape (784 x 128) 
- b₁:    Matrix of shape (1 x 128)  
- Z₁:    Matrix of shape (batch_size x 128)


---

### 🔹 Step 2: Apply Activation (ReLU)

`A₁ = ReLU(Z₁) = max(0, Z₁)`

- This introduces **non-linearity**, allowing the network to learn complex functions.

---

### 🔹 Step 3: Linear Transformation (Hidden → Output)

`Z₂ = A₁ · W₂ + b₂`

- A₁: Matrix of shape (batch_size × 128)  
- W₂: Matrix of shape (128 × 10)  
- b₂: Matrix of shape (1 × 10)  
- Z₂: Matrix of shape (batch_size × 10)

---

### 🔹 Step 4: Apply Activation (Softmax)

`A₂ = Softmax(Z₂)`

- Converts raw scores into a **probability distribution** over 10 classes  
- A₂: Matrix of shape (batch_size × 10), where each row sums to 1

**Softmax formula**:  
`softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)`

---

### 💡 Why ReLU and Softmax?

- **ReLU** (Rectified Linear Unit):
  - Simple and fast.
  - Prevents vanishing gradients (compared to sigmoid/tanh).
- **Softmax**:
  - Normalizes outputs to probabilities.
  - Used for multi-class classification.

---

### 🧪 Summary of Forward Flow

```text
Input X (784 features)
     ↓
Linear: Z₁ = XW₁ + b₁
     ↓
Activation: A₁ = ReLU(Z₁)
     ↓
Linear: Z₂ = A₁W₂ + b₂
     ↓
Activation: A₂ = Softmax(Z₂)
     ↓
Output: Probability for each digit class (0–9)
```

---

## 🔁 Backpropagation

### ❓ What is Backpropagation?

Backpropagation is the **learning algorithm** used to train neural networks.

It's how the network **calculates the gradient** (i.e., the direction and strength of change needed) of its loss function with respect to its parameters (weights and biases), so it can update them and **minimize the error**.

---

### 🤔 Why Do We Need It?

In forward propagation, we compute outputs.

But during training, we also have the **true labels** — and we want our network's prediction to match them.

To do that, we:
1. Compute the error (loss)
2. **Propagate that error backward** to figure out:
   - Which weights contributed most to the error
   - How much to adjust each weight to reduce that error

This is done using **the chain rule of calculus** — systematically computing gradients layer-by-layer, starting from the output and moving back to the input.

---

## 🧮 How Does It Work? (Step-by-Step)

We’ll assume we already did a **forward pass**, so we know:
- `A₂`: Output from Softmax
- `Y`: True labels (one-hot encoded)
- Intermediate values: `Z₁`, `A₁`, `Z₂`

Let’s derive the gradients for each parameter.

---

### 🔹 Step 1: Gradient of Loss w.r.t. Output (Softmax + Cross-Entropy)

**Loss Function (Categorical Cross-Entropy):**

L = - Σᵢ ( yᵢ * log(a₂ᵢ) )

Where:
- yᵢ is the true label (1 for correct class, else 0)
- a₂ᵢ is the predicted probability for class i (from softmax)


**Gradient of loss w.r.t. Z₂**:

`dZ₂ = A₂ - Y`


> ✅ **Why?** For Softmax + Cross-Entropy combined, this is the elegant result of their derivative — simplifies backprop significantly.

---

### 🔹 Step 2: Gradients for Output Layer Parameters (W₂, b₂)

Using `dZ₂`:

- `dW₂ = (A₁ᵀ · dZ₂) / m`  
- `db₂ = sum(dZ₂) / m`

Where:
- `m` = batch size
- `dW₂` is the gradient of the loss w.r.t. the weights connecting hidden → output
- `db₂` is the gradient for the output bias

---

### 🔹 Step 3: Backprop to Hidden Layer (dZ₁)

We now propagate error from output back to hidden:

- `dZ₁ = (dZ₂ · W₂ᵀ) * ReLU'(Z₁)`

> ✅ **Why the derivative of ReLU?**  
Because ReLU "kills" negative values — its gradient is 0 for `z ≤ 0`, and 1 for `z > 0`.

---

### 🔹 Step 4: Gradients for Hidden Layer Parameters (W₁, b₁)

Now that we have `dZ₁`, we can calculate:

- `dW₁ = (Xᵀ · dZ₁) / m`  
- `db₁ = sum(dZ₁) / m`

These gradients tell us how to update the input → hidden layer weights and biases.

---

### 📦 Summary of Backpropagation Flow

```text
Output Layer:
  dZ₂ = A₂ - Y
  dW₂ = A₁ᵀ · dZ₂ / m
  db₂ = sum(dZ₂) / m

Hidden Layer:
  dZ₁ = (dZ₂ · W₂ᵀ) * ReLU'(Z₁)
  dW₁ = Xᵀ · dZ₁ / m
  db₁ = sum(dZ₁) / m
```

---

## 🔧 Part 4: Updating Weights (Gradient Descent)

### ❓ What is Weight Update?

This is the **final step** of a training iteration — where the neural network actually learns.

After computing the gradients using backpropagation, we **update the weights and biases** so that the **loss decreases** in the next round.

This process is done using an optimization algorithm — in our case: **Gradient Descent**.

---

### 🤔 Why Gradient Descent?

Gradient Descent is the most common and intuitive optimizer:

- It updates parameters by taking a **small step in the direction of negative gradient**
- Think of it as **"descending down a hill"** (loss landscape) to reach a minimum

---

### 🧠 How Does Gradient Descent Work?

We update weights and biases using this rule:

```text
parameter = parameter - learning_rate * gradient

W₁ = W₁ - lr · dW₁  
b₁ = b₁ - lr · db₁  
W₂ = W₂ - lr · dW₂  
b₂ = b₂ - lr · db₂
```

---

### Forward → Compute Loss → Backpropagation → Update Weights → Repeat



