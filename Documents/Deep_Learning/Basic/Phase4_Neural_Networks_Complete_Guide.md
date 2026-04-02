# Neural Networks: Complete Guide with Example Data at Every Step

## The Problem

You have 8 pizza stores. You want to predict: **Will a store be successful (1) or not (0)?**
Same data as before — but now we'll use a neural network instead of logistic regression.

### Our Data

| Store | Rating (x₁) | Delivery_Min (x₂) | Successful (y) |
|-------|-------------|-------------------|----------------|
| S1    | 4.5         | 20                | 1 (Yes)        |
| S2    | 3.2         | 45                | 0 (No)         |
| S3    | 4.8         | 18                | 1 (Yes)        |
| S4    | 2.9         | 50                | 0 (No)         |
| S5    | 4.1         | 25                | 1 (Yes)        |
| S6    | 3.5         | 35                | 0 (No)         |
| S7    | 4.3         | 22                | 1 (Yes)        |
| S8    | 3.0         | 40                | 0 (No)         |

---

## STEP 1: From Logistic Regression to Neural Networks

### What You Already Know

Logistic regression is:
```
z = β₀ + β₁x₁ + β₂x₂     (linear combination)
p = σ(z) = 1/(1+e⁻ᶻ)       (sigmoid activation)
```

A neural network is just **many logistic regressions stacked together**.

### The Key Insight

```
Logistic Regression:  Input → [one linear + sigmoid] → Output
                      (can only learn LINEAR boundaries)

Neural Network:       Input → [linear + activation] → [linear + activation] → Output
                      (can learn ANY boundary — curves, circles, anything)
```

Why? Each layer transforms the data into a new representation where the problem becomes easier. The first layer might learn "is this a high-rating store?" and "is this a fast-delivery store?" The second layer combines these into "is this a successful store?"

---

## STEP 2: The Perceptron — A Single Neuron

### What Is a Neuron?

A neuron does exactly what logistic regression does:

```
Inputs: x₁ = Rating, x₂ = Delivery
Weights: w₁, w₂
Bias: b

Step 1: z = w₁x₁ + w₂x₂ + b          (weighted sum)
Step 2: a = activation(z)               (apply activation function)
Step 3: output = a                       (this neuron's output)
```

### Why "Neuron"?

```
Biological neuron:
  Dendrites (inputs) → Cell body (processing) → Axon (output)
  Multiple signals come in → neuron "fires" if total signal exceeds threshold

Artificial neuron:
  Features (inputs) → Weighted sum + activation (processing) → Output
  Multiple numbers come in → neuron outputs a number between 0 and 1
```

### Computing a Single Neuron for S1

```
Inputs: x₁ = 4.5 (Rating), x₂ = 20 (Delivery)
Weights: w₁ = 1.0, w₂ = -0.05 (random initialization)
Bias: b = -2.0

z = 1.0(4.5) + (-0.05)(20) + (-2.0)
  = 4.5 - 1.0 - 2.0
  = 1.5

a = σ(1.5) = 1/(1 + e⁻¹·⁵) = 1/(1 + 0.223) = 1/1.223 = 0.818

Output: 0.818 (81.8% — this neuron thinks S1 is likely successful)
```

---

## STEP 3: Building a Network — Layers of Neurons

### Architecture: 2-2-1 Network

```
Input Layer (2 neurons):    x₁ = Rating, x₂ = Delivery
Hidden Layer (2 neurons):   h₁, h₂ (learn intermediate features)
Output Layer (1 neuron):    ŷ (final prediction)

    x₁ ──→ h₁ ──→
         ╲╱        ŷ
         ╱╲
    x₂ ──→ h₂ ──→

Every input connects to every hidden neuron (fully connected).
Every hidden neuron connects to the output.
```

### Why "Hidden" Layer?

```
Input layer:  We choose these (Rating, Delivery)
Output layer: We define this (Success probability)
Hidden layer: The network LEARNS what these represent

h₁ might learn: "is this a premium store?" (high rating + fast delivery)
h₂ might learn: "is this a budget store?" (low rating + slow delivery)

We don't tell it this — it discovers useful features on its own.
This is REPRESENTATION LEARNING — the network's superpower.
```

### Counting Parameters

```
Input → Hidden:
  w₁₁ (x₁→h₁), w₁₂ (x₁→h₂) = 2 weights
  w₂₁ (x₂→h₁), w₂₂ (x₂→h₂) = 2 weights
  b₁ (h₁ bias), b₂ (h₂ bias) = 2 biases
  Subtotal: 6 parameters

Hidden → Output:
  v₁ (h₁→ŷ), v₂ (h₂→ŷ) = 2 weights
  c (output bias)          = 1 bias
  Subtotal: 3 parameters

Total: 9 parameters (for just 8 data points!)
This is why neural networks need lots of data — they have many parameters.
```

---

## STEP 4: Forward Pass — Computing Predictions

### Initialize Weights (Random)

```
Input → Hidden weights:
  w₁₁ = 0.5,  w₁₂ = -0.3   (x₁ → h₁, h₂)
  w₂₁ = -0.02, w₂₂ = 0.04  (x₂ → h₁, h₂)
  b₁ = -1.0,  b₂ = 0.5     (biases)

Hidden → Output weights:
  v₁ = 0.8,  v₂ = -0.6     (h₁, h₂ → ŷ)
  c = -0.1                   (bias)
```

### Forward Pass for S1 (Rating=4.5, Delivery=20)

**Hidden Layer:**
```
h₁: z₁ = w₁₁(4.5) + w₂₁(20) + b₁
       = 0.5(4.5) + (-0.02)(20) + (-1.0)
       = 2.25 - 0.4 - 1.0
       = 0.85
    a₁ = σ(0.85) = 1/(1 + e⁻⁰·⁸⁵) = 1/1.427 = 0.700

h₂: z₂ = w₁₂(4.5) + w₂₂(20) + b₂
       = (-0.3)(4.5) + 0.04(20) + 0.5
       = -1.35 + 0.8 + 0.5
       = -0.05
    a₂ = σ(-0.05) = 1/(1 + e⁰·⁰⁵) = 1/1.051 = 0.951... wait
    
    Actually: σ(-0.05) = 1/(1 + e⁰·⁰⁵) = 1/(1 + 1.051) = 1/2.051 = 0.488
```

**Output Layer:**
```
ŷ: z_out = v₁(a₁) + v₂(a₂) + c
         = 0.8(0.700) + (-0.6)(0.488) + (-0.1)
         = 0.560 - 0.293 - 0.1
         = 0.167
   ŷ = σ(0.167) = 1/(1 + e⁻⁰·¹⁶⁷) = 1/1.846... 

   Actually: σ(0.167) = 1/(1 + e⁻⁰·¹⁶⁷) = 1/(1 + 0.846) = 1/1.846 = 0.542
```

**Result: ŷ = 0.542 (54.2%) — barely positive. Actual = 1. Close but not confident.**

With random weights, the network makes poor predictions. Training will fix this.

---

## STEP 5: Loss Function — Same as Logistic Regression

```
Binary Cross-Entropy (Log Loss):
  Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

For S1: y=1, ŷ=0.542
  Loss = -[1 × log(0.542) + 0 × log(0.458)]
       = -[log(0.542)]
       = -(-0.613)
       = 0.613

For comparison, a perfect prediction (ŷ=0.99):
  Loss = -[log(0.99)] = 0.01  (much lower!)
```

---

## STEP 6: Backpropagation — How the Network Learns

### The Core Idea

```
Forward pass:  Input → Hidden → Output → Loss  (compute prediction)
Backward pass: Loss → Output → Hidden → Input  (compute gradients)

"How much did each weight contribute to the error?"
Then adjust each weight to reduce the error.
```

### Why "Back" Propagation?

```
The loss depends on the output weights (directly).
The output depends on the hidden activations.
The hidden activations depend on the input weights.

To find ∂Loss/∂w₁₁ (how input weight affects loss),
we chain through all the layers BACKWARDS:

∂Loss/∂w₁₁ = (∂Loss/∂ŷ) × (∂ŷ/∂a₁) × (∂a₁/∂z₁) × (∂z₁/∂w₁₁)

This is the CHAIN RULE applied through multiple layers.
Same concept as logistic regression (Step 4.1 in that guide),
but now with more links in the chain.
```

### Computing Gradients for Output Layer

```
∂Loss/∂v₁ = (ŷ - y) × a₁ = (0.542 - 1) × 0.700 = -0.458 × 0.700 = -0.321

Interpretation: v₁ should INCREASE (negative gradient → move opposite direction)
This makes sense — h₁ had a positive activation (0.700) and the prediction
was too low (0.542 vs 1.0), so we should increase v₁ to push the output up.
```

### Computing Gradients for Hidden Layer (the "back" part)

```
∂Loss/∂w₁₁ = (ŷ - y) × v₁ × a₁(1-a₁) × x₁
            = (-0.458) × 0.8 × 0.700 × 0.300 × 4.5
            = (-0.458) × 0.8 × 0.210 × 4.5
            = -0.346

The error signal flows BACKWARD through the network:
  Output error → scaled by output weight v₁ → scaled by sigmoid derivative → scaled by input x₁
```

### Weight Update

```
Learning rate η = 0.1

v₁_new = v₁ - η × ∂Loss/∂v₁ = 0.8 - 0.1 × (-0.321) = 0.8 + 0.032 = 0.832
w₁₁_new = w₁₁ - η × ∂Loss/∂w₁₁ = 0.5 - 0.1 × (-0.346) = 0.5 + 0.035 = 0.535

Both weights increased — pushing the prediction higher (toward the correct answer of 1).
```

---

## STEP 7: Activation Functions — Beyond Sigmoid

### Why Sigmoid Isn't Always Best

```
Sigmoid: σ(z) = 1/(1+e⁻ᶻ)
  Output range: (0, 1)
  Problem: Vanishing gradients — for large |z|, gradient ≈ 0
           The network stops learning in deep layers!
```

### ReLU — The Modern Default

```
ReLU(z) = max(0, z)

  z = -3 → ReLU = 0
  z = 0  → ReLU = 0
  z = 3  → ReLU = 3

  Gradient: 0 if z < 0, 1 if z > 0
  No vanishing gradient for positive values!
  Much faster to compute than sigmoid.
```

### When to Use Which

```
| Activation | Formula        | Range    | Use For                    |
|------------|----------------|----------|----------------------------|
| Sigmoid    | 1/(1+e⁻ᶻ)     | (0, 1)   | Output layer (binary)      |
| Tanh       | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (-1, 1) | Hidden layers (older)  |
| ReLU       | max(0, z)      | [0, ∞)   | Hidden layers (default)    |
| Softmax    | eᶻⁱ/Σeᶻʲ      | (0, 1)   | Output layer (multi-class) |
```

---

## STEP 8: Optimizers — Smarter Than Basic Gradient Descent

### The Problem with Basic Gradient Descent

```
Basic GD: w = w - η × gradient

Problems:
1. Same learning rate for all parameters
2. Can get stuck in saddle points
3. Oscillates in ravines (narrow valleys)
```

### SGD (Stochastic Gradient Descent)

```
Instead of computing gradient on ALL data, use one random sample (or mini-batch).

Batch GD:     gradient from all 8 stores → 1 update
Mini-batch GD: gradient from 2-4 stores → 1 update (faster, noisier)
SGD:          gradient from 1 store → 1 update (fastest, noisiest)

The noise actually HELPS — it can escape local minima!
```

### Adam — The Go-To Optimizer

```
Adam = Adaptive Moment Estimation

It maintains:
  m = running average of gradients (momentum — which direction to go)
  v = running average of squared gradients (how much to scale each parameter)

Update: w = w - η × m / (√v + ε)

Why it works:
  - Parameters with large gradients get smaller updates (prevents overshooting)
  - Parameters with small gradients get larger updates (prevents stalling)
  - Momentum helps push through flat regions and saddle points

In practice: just use Adam with η=0.001. It works for almost everything.
```

---

## STEP 9: Training Loop — The Complete Picture

```
for epoch in range(1000):           # repeat many times
    for batch in data:               # process data in batches
        
        # 1. Forward pass
        predictions = network(batch.inputs)
        
        # 2. Compute loss
        loss = cross_entropy(predictions, batch.labels)
        
        # 3. Backward pass (backpropagation)
        gradients = compute_gradients(loss)
        
        # 4. Update weights
        optimizer.step(gradients)
    
    # Track progress
    print(f"Epoch {epoch}: Loss = {loss}")
```

### Key Concepts

```
Epoch:      One complete pass through ALL training data
Batch:      A subset of data processed together (e.g., 32 samples)
Batch size: How many samples per batch (32, 64, 128 are common)
Iteration:  One weight update (one batch processed)

Example: 1000 samples, batch size 100
  → 10 iterations per epoch
  → 100 epochs = 1000 iterations total
```

---

## STEP 10: Overfitting in Neural Networks

### The Problem

```
Neural networks have MANY parameters (millions in modern networks).
With enough parameters, they can memorize the training data perfectly.

Training loss: 0.001 (perfect!)
Test loss:     2.500 (terrible!)
```

### Solutions

```
1. DROPOUT: Randomly "turn off" neurons during training
   Each neuron has a probability p (e.g., 0.5) of being disabled.
   Forces the network to not rely on any single neuron.
   Like training an ensemble of smaller networks.

2. EARLY STOPPING: Stop training when validation loss starts increasing
   Monitor validation loss each epoch.
   When it stops improving for N epochs → stop.
   The simplest and most effective regularization.

3. WEIGHT DECAY (L2 regularization): Same as Ridge regression
   Add λ × Σw² to the loss function.
   Penalizes large weights → simpler model.

4. BATCH NORMALIZATION: Normalize activations between layers
   Keeps activations in a reasonable range.
   Speeds up training and acts as mild regularization.

5. DATA AUGMENTATION: Create more training data artificially
   For images: rotate, flip, crop, change brightness.
   For text: synonym replacement, back-translation.
```

---

## COMPLETE FORMULA SUMMARY

```
1. Neuron:          z = Σ wᵢxᵢ + b,  a = activation(z)
2. Sigmoid:         σ(z) = 1/(1+e⁻ᶻ)
3. ReLU:            ReLU(z) = max(0, z)
4. Loss (binary):   L = -[y log(ŷ) + (1-y) log(1-ŷ)]
5. Backprop:        ∂L/∂w = chain rule through all layers
6. Update:          w = w - η × ∂L/∂w
7. Adam:            w = w - η × m/(√v + ε)
8. Dropout:         Randomly zero out neurons with probability p
```

---

## INTERVIEW CHEAT SHEET

**Q: "What is a neural network?"**
> "A series of layers, each containing neurons that compute a weighted sum of inputs, add a bias, and apply an activation function. The network learns by adjusting weights through backpropagation to minimize a loss function. Each layer transforms the data into a more useful representation."

**Q: "Explain backpropagation."**
> "It's the chain rule applied through the network layers in reverse. The loss gradient flows backward from the output to the input, computing how much each weight contributed to the error. Each weight is then adjusted proportionally to reduce the error. It's the same math as logistic regression's gradient, just applied through more layers."

**Q: "Why ReLU over sigmoid?"**
> "Sigmoid suffers from vanishing gradients — for large inputs, the gradient approaches zero, so deep layers stop learning. ReLU has a constant gradient of 1 for positive inputs, so gradients flow freely through deep networks. It's also computationally cheaper (just a max operation vs exponential)."

**Q: "What is dropout and why does it work?"**
> "Dropout randomly disables neurons during training with probability p. This prevents co-adaptation — neurons can't rely on specific other neurons being present. It's like training an ensemble of different sub-networks. At test time, all neurons are active but scaled by (1-p)."

**Q: "SGD vs Adam?"**
> "SGD uses the same learning rate for all parameters. Adam adapts the learning rate per parameter using running averages of gradients (momentum) and squared gradients (scaling). Adam converges faster and requires less tuning. Use Adam as default; SGD with momentum can sometimes generalize better with careful tuning."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
