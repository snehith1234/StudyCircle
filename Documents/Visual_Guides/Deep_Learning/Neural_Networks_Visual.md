# Neural Networks: Visual Guide with Mermaid Diagrams

> Visual companion to `Documents/Deep_Learning/Basic/Phase4_Neural_Networks_Complete_Guide.md`.
> Every diagram has explanatory text — what it shows, why it matters, and how to read it.

---

## 1. From Logistic Regression to Neural Networks

A neural network is just many logistic regressions stacked together. Logistic regression can only learn straight-line boundaries. By stacking layers, a neural network can learn any shape of boundary — curves, circles, anything. The diagram shows this progression.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    LR["Logistic Regression<br/>Input → linear, sigmoid → Output<br/><i>Only linear boundaries</i>"]
    LR -->|"Stack multiple layers"| NN["Neural Network<br/>Input → layer → layer → Output<br/><i>Any boundary shape</i>"]

    style LR fill:#252840,stroke:#f5b731,color:#c8cfe0
    style NN fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = what you already know. Green = the upgrade. Each layer transforms the data into a new representation where the problem becomes easier to solve.

---

## 2. A Single Neuron

A neuron does three things: multiply inputs by weights, add a bias, and apply an activation function. It's identical to logistic regression — the building block of all neural networks.

Before training, the network starts with **random weights**. These are just initial guesses — training will adjust them. For this example:

- **w₁ = 1.0** (weight for Rating — random starting value)
- **w₂ = −0.05** (weight for Delivery — random, slightly negative because longer delivery might hurt)
- **b = −2.0** (bias — random starting offset)

The neuron computes: z = w₁ × Rating + w₂ × Delivery + b, then applies the sigmoid function to squash the result into a probability between 0 and 1.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    X1["x₁ = Rating<br/>4.5"] --> SUM["Σ weighted sum<br/>z = w₁×x₁ plus w₂×x₂ plus b<br/>= 1.0×4.5 plus (−0.05)×20 plus (−2.0)<br/>= 4.5 − 1.0 − 2.0 = <b>1.5</b>"]
    X2["x₂ = Delivery<br/>20"] --> SUM
    W["Random weights:<br/>w₁=1.0, w₂=−0.05<br/>bias b=−2.0"] -.-> SUM
    SUM --> ACT["Activation<br/>a = σ(1.5)<br/>= <b>0.818</b>"]
    ACT --> OUT["Output<br/>81.8% likely<br/>successful"]

    style X1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style X2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style W fill:#1a1d2e,stroke:#f45d6d,color:#d8a8b8
    style SUM fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style ACT fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style OUT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = inputs (our pizza store data). Red (dashed) = random initial weights (these get adjusted during training). Blue = weighted sum computation. Purple = sigmoid activation. Green = final output. The weights are random guesses — with training, w₁ would increase (higher rating = more successful) and w₂ would become more negative (longer delivery = less successful).

---

## 3. Network Architecture — Layers of Neurons

The 2-2-1 network: 2 inputs, 2 hidden neurons, 1 output. Every input connects to every hidden neuron (fully connected). The hidden layer learns intermediate features that the input layer can't represent directly.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    X1["x₁ Rating"] --> H1["h₁<br/><i>learns: premium?</i>"]
    X1 --> H2["h₂<br/><i>learns: budget?</i>"]
    X2["x₂ Delivery"] --> H1
    X2 --> H2
    H1 --> Y["ŷ<br/>Success<br/>probability"]
    H2 --> Y

    style X1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style X2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style H1 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style H2 fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style Y fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = input layer (we choose these). Blue/Purple = hidden layer (network learns what these represent). Green = output. The hidden neurons might learn "is this a premium store?" and "is this a budget store?" — we don't tell it this, it discovers useful features on its own. This is representation learning.

---

## 4. Forward Pass vs Backward Pass

Training has two phases that repeat thousands of times. Forward pass computes the prediction. Backward pass computes how to fix the weights. The diagram shows both directions.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    FWD["⬇️ FORWARD PASS<br/>Input → Hidden → Output → Loss<br/><i>Compute prediction and error</i>"]
    FWD --> LOSS["Loss = 0.613<br/><i>How wrong is the prediction?</i>"]
    LOSS --> BWD["⬆️ BACKWARD PASS (Backprop)<br/>Loss → Output → Hidden → Input<br/><i>Compute gradients for each weight</i>"]
    BWD --> UPD["Update all weights<br/>w = w minus η × gradient<br/><i>Nudge weights to reduce error</i>"]
    UPD --> REPEAT["Repeat 1000s of times<br/>until loss is small"]

    style FWD fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style LOSS fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style BWD fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style UPD fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style REPEAT fill:#252840,stroke:#f5b731,color:#c8cfe0
```

Green = forward (compute prediction). Red = loss (measure error). Blue = backward (compute gradients via chain rule). Purple = update (fix weights). Yellow = repeat. Backpropagation is just the chain rule applied through multiple layers — same math as logistic regression, more links in the chain.

---

## 5. Activation Functions

The activation function determines what a neuron outputs. Sigmoid was the original but causes vanishing gradients in deep networks. ReLU is the modern default — simple, fast, and gradient-friendly.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    SIGMOID["Sigmoid: 1/(1 plus e to neg z)<br/>Range: 0 to 1<br/>Use: output layer (binary)<br/><i>Problem: vanishing gradients</i>"]
    RELU["ReLU: max(0, z)<br/>Range: 0 to infinity<br/>Use: hidden layers (default)<br/><i>Fast, no vanishing gradient</i>"]
    SOFTMAX["Softmax: e to zᵢ / Σ e to zⱼ<br/>Range: 0 to 1, sums to 1<br/>Use: output layer (multi class)<br/><i>Outputs probabilities per class</i>"]

    style SIGMOID fill:#252840,stroke:#f5b731,color:#c8cfe0
    style RELU fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style SOFTMAX fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
```

Yellow = sigmoid (classic, use for binary output only). Green = ReLU (modern default for hidden layers). Blue = softmax (for multi-class output). Rule of thumb: ReLU for hidden layers, sigmoid for binary output, softmax for multi-class output.

---

## 6. Optimizers — How Weights Get Updated

### Why we need smart optimizers

Basic gradient descent uses the same learning rate for every parameter. Adam adapts the rate per parameter — parameters with large gradients get smaller steps (prevents overshooting), parameters with small gradients get larger steps (prevents stalling).

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    SGD["SGD<br/>w = w minus η × gradient<br/>Same rate for all params<br/><i>Simple but slow</i>"]
    SGD -->|"Add momentum"| SGDM["SGD with Momentum<br/>Remembers past gradients<br/>Pushes through flat regions<br/><i>Better but still one rate</i>"]
    SGDM -->|"Add per-param scaling"| ADAM["Adam (default choice)<br/>Adapts rate per parameter<br/>momentum plus scaling<br/><i>Just use η=0.001</i>"]

    style SGD fill:#252840,stroke:#f5b731,color:#c8cfe0
    style SGDM fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style ADAM fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Each step adds an improvement. In practice: just use Adam with learning rate 0.001. It works for almost everything.

---

## 7. Overfitting Solutions

Neural networks have millions of parameters and can memorize training data. These techniques prevent that.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    PROBLEM["⚠️ Overfitting<br/>Train loss: 0.001<br/>Test loss: 2.500"] --> D & E & W & B

    D["Dropout<br/>Randomly disable neurons<br/>during training (p=0.5)<br/><i>Like training many sub-networks</i>"]
    E["Early Stopping<br/>Stop when validation loss<br/>starts increasing<br/><i>Simplest and most effective</i>"]
    W["Weight Decay (L2)<br/>Add λ × Σw² to loss<br/>Same as Ridge regression<br/><i>Penalizes large weights</i>"]
    B["Batch Normalization<br/>Normalize activations<br/>between layers<br/><i>Stabilizes and regularizes</i>"]

    style PROBLEM fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style D fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style E fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style W fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style B fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
```

Red = the problem. Green = early stopping (try first, simplest). Blue/Purple = other techniques to combine as needed. In practice, use early stopping always, add dropout for large networks, and weight decay for fine-tuning.

---

## 8. The Training Loop

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
flowchart TD
    START(["For each epoch:"]) --> BATCH["Get next batch<br/>(32 samples)"]
    BATCH --> FWD["Forward pass<br/>compute predictions"]
    FWD --> LOSS["Compute loss"]
    LOSS --> BWD["Backward pass<br/>compute gradients"]
    BWD --> UPD["Optimizer step<br/>update weights"]
    UPD --> MORE{"More batches?"}
    MORE -->|Yes| BATCH
    MORE -->|No| VAL["Check validation loss"]
    VAL --> STOP{"Improving?"}
    STOP -->|Yes| START
    STOP -->|No| DONE(["Stop training"])

    style START fill:#252840,stroke:#f5b731,color:#c8cfe0
    style LOSS fill:#2a1a1f,stroke:#f45d6d,color:#e2e8f0
    style BWD fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style DONE fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

The outer loop (epochs) repeats over the full dataset. The inner loop (batches) processes chunks of data. After each epoch, check validation loss — if it stops improving, stop training (early stopping).

---

## 9. Interview Decision Tree 🎯

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    Q1{"What is a<br/>neural network?"} -->|Answer| A1["Layers of neurons<br/>Each: weighted sum, activation<br/>Learns by backpropagation"]
    Q1 -->|Next Q| Q2{"Explain<br/>backpropagation?"}
    Q2 -->|Answer| A2["Chain rule through layers<br/>Loss gradient flows backward<br/>Each weight adjusted to reduce error"]
    Q2 -->|Next Q| Q3{"Why ReLU<br/>over sigmoid?"}
    Q3 -->|Answer| A3["Sigmoid: vanishing gradients<br/>ReLU: constant gradient for positive<br/>Faster, works in deep networks"]
    Q3 -->|Next Q| Q4{"What is dropout?"}
    Q4 -->|Answer| A4["Randomly disable neurons (p=0.5)<br/>Prevents co-adaptation<br/>Like ensemble of sub-networks"]

    style Q1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A4 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

---

> 💡 **How to view:** GitHub (native), VS Code (Mermaid extension), Obsidian (built-in), or [mermaid.live](https://mermaid.live)
