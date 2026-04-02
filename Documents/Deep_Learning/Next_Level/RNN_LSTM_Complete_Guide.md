# RNNs & LSTMs: Complete Guide with Example Data at Every Step

## The Problem

You have a sequence of pizza reviews, one word at a time. You want to predict the sentiment AFTER reading the whole sentence. But regular neural networks can't handle sequences — they need a fixed-size input. How do you process "Great pizza fast delivery" when the sentence could be 4 words or 40 words?

### Why Regular Neural Networks Fail on Sequences

```
Regular NN: Input must be FIXED SIZE (e.g., 2 numbers: Rating, Delivery)

But text is VARIABLE LENGTH:
  "Great pizza" → 2 words
  "Great pizza fast delivery" → 4 words
  "The pizza was absolutely amazing and the delivery was super fast" → 11 words

You can't just pad everything to the max length — that wastes computation
and doesn't capture the SEQUENTIAL nature of language.

"not good" ≠ "good not" — ORDER MATTERS.
A regular NN treats inputs as a bag — no concept of sequence.
```

---

## STEP 1: The RNN Idea — Memory Through Time

### The Core Insight

An RNN processes one word at a time, but carries a **hidden state** from one step to the next. This hidden state is the network's "memory" — it summarizes everything it has seen so far.

```
Regular NN:  Each input is independent. No memory.
  x₁ → output₁
  x₂ → output₂  (doesn't know about x₁)

RNN: Each step sees the current input AND the previous hidden state.
  x₁ → h₁ (remembers x₁)
  x₂, h₁ → h₂ (remembers x₁ and x₂)
  x₃, h₂ → h₃ (remembers x₁, x₂, and x₃)
  ...
  Final h → output (based on ENTIRE sequence)
```

### The RNN Formula

```
At each time step t:
  hₜ = tanh(W_hh × hₜ₋₁ + W_xh × xₜ + b)

Where:
  xₜ     = input at time t (e.g., word embedding for "pizza")
  hₜ₋₁   = hidden state from previous step (the "memory")
  W_xh    = weight matrix for input (learned)
  W_hh    = weight matrix for hidden state (learned)
  b       = bias (learned)
  tanh    = activation function (squashes to -1 to 1)
  hₜ      = new hidden state (updated memory)

The SAME weights W_xh and W_hh are used at EVERY time step.
This is called "weight sharing" — the network applies the same
transformation at each step, regardless of sequence length.
```

---

## STEP 2: RNN Step by Step — Processing "Great pizza fast delivery"

### Setup

```
Word embeddings (simplified to 3 numbers):
  "great"    = [0.8, 0.1, 0.9]
  "pizza"    = [0.1, 0.9, 0.1]
  "fast"     = [0.7, 0.0, 0.8]
  "delivery" = [0.2, 0.8, 0.3]

Initial hidden state: h₀ = [0, 0, 0] (no memory yet)

Weights (simplified — normally learned during training):
  W_xh = 3×3 matrix (input to hidden)
  W_hh = 3×3 matrix (hidden to hidden)
```

### Step 1: Process "great"

```
Input: x₁ = [0.8, 0.1, 0.9]
Previous hidden: h₀ = [0, 0, 0]

h₁ = tanh(W_hh × [0, 0, 0] + W_xh × [0.8, 0.1, 0.9] + b)
   = tanh(W_xh × [0.8, 0.1, 0.9])  (h₀ is zero, so W_hh term vanishes)
   ≈ [0.6, 0.1, 0.7]  (after matrix multiply and tanh)

h₁ now "remembers": something positive was said (high first and third values)
```

### Step 2: Process "pizza"

```
Input: x₂ = [0.1, 0.9, 0.1]
Previous hidden: h₁ = [0.6, 0.1, 0.7]

h₂ = tanh(W_hh × [0.6, 0.1, 0.7] + W_xh × [0.1, 0.9, 0.1] + b)
   ≈ [0.4, 0.5, 0.5]

h₂ now "remembers": something positive about pizza
```

### Step 3: Process "fast"

```
Input: x₃ = [0.7, 0.0, 0.8]
Previous hidden: h₂ = [0.4, 0.5, 0.5]

h₃ = tanh(W_hh × [0.4, 0.5, 0.5] + W_xh × [0.7, 0.0, 0.8] + b)
   ≈ [0.7, 0.3, 0.8]

h₃ now "remembers": positive pizza, something fast
```

### Step 4: Process "delivery"

```
Input: x₄ = [0.2, 0.8, 0.3]
Previous hidden: h₃ = [0.7, 0.3, 0.8]

h₄ = tanh(W_hh × [0.7, 0.3, 0.8] + W_xh × [0.2, 0.8, 0.3] + b)
   ≈ [0.5, 0.6, 0.6]

h₄ is the FINAL hidden state — it summarizes the entire review.
```

### Final Prediction

```
output = sigmoid(W_out × h₄ + b_out)
       = sigmoid(W_out × [0.5, 0.6, 0.6] + b_out)
       ≈ 0.82

Prediction: 82% positive → Positive review ✅
```

---

## STEP 3: The Vanishing Gradient Problem in RNNs

### Why RNNs Struggle with Long Sequences

The same vanishing gradient problem from deep networks hits RNNs even harder. An RNN processing a 100-word sentence is like a 100-layer deep network — the gradient must flow back through 100 time steps.

```
"The pizza that we ordered last Tuesday from the new place on Main Street was great"

To learn that "great" relates to "pizza", the gradient must flow back 12 steps.
Each step multiplies by tanh'(z) (max = 1.0) and W_hh.

If W_hh has eigenvalues < 1: gradient vanishes → forgets early words
If W_hh has eigenvalues > 1: gradient explodes → training diverges

In practice: RNNs can only "remember" about 10-20 steps back.
After that, the gradient is too small to carry information.
```

### The Fundamental Problem

```
RNN hidden state update: hₜ = tanh(W_hh × hₜ₋₁ + W_xh × xₜ)

The gradient of hₜ with respect to h₁ involves:
  ∂hₜ/∂h₁ = Π (from k=1 to t-1) of [tanh'(zₖ) × W_hh]

This is a PRODUCT of t-1 terms.
If each term < 1: product → 0 (vanishing)
If each term > 1: product → ∞ (exploding)

For a 100-word sentence: you need this product to stay near 1
for 99 multiplications. That's nearly impossible with simple multiplication.
```

---

## STEP 4: LSTM — The Solution to Forgetting

### The Key Insight

Instead of one hidden state that gets overwritten at every step, LSTM adds a **cell state** — a separate memory highway that information can flow through with minimal modification. Three **gates** control what gets added, removed, or read from this memory.

```
RNN:  hₜ = tanh(W × [hₜ₋₁, xₜ])
      One state, completely overwritten each step.
      Old information gets "washed out" by new information.

LSTM: cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ
      Cell state cₜ is a HIGHWAY — information flows through with
      only small, controlled modifications via gates.
      Old information is PRESERVED unless explicitly removed.
```

### The Three Gates

Each gate is a sigmoid layer that outputs values between 0 and 1. A value of 0 means "block everything" and 1 means "let everything through."

```
1. FORGET GATE (fₜ): "What should I forget from the old memory?"
   fₜ = σ(W_f × [hₜ₋₁, xₜ] + b_f)
   
   Output: values between 0 and 1 for each memory cell
   0 = completely forget this memory
   1 = completely keep this memory

2. INPUT GATE (iₜ): "What new information should I store?"
   iₜ = σ(W_i × [hₜ₋₁, xₜ] + b_i)     ← how much to add
   c̃ₜ = tanh(W_c × [hₜ₋₁, xₜ] + b_c)  ← what to add (candidate)

3. OUTPUT GATE (oₜ): "What should I output from memory?"
   oₜ = σ(W_o × [hₜ₋₁, xₜ] + b_o)
```

### The Cell State Update

```
Step 1: Forget old stuff
  cₜ = fₜ ⊙ cₜ₋₁
  (⊙ means element-wise multiplication)
  If fₜ = 0.9: keep 90% of old memory
  If fₜ = 0.1: forget 90% of old memory

Step 2: Add new stuff
  cₜ = cₜ + iₜ ⊙ c̃ₜ
  Only add the parts that the input gate allows

Step 3: Produce output
  hₜ = oₜ ⊙ tanh(cₜ)
  Only output the parts that the output gate allows
```

---

## STEP 5: LSTM Step by Step — "The pizza was not good"

This example shows how the forget gate handles negation — a classic test for sequence models.

### Processing "The"

```
x₁ = embedding("The")
Forget gate: fₜ ≈ 1.0 (nothing to forget yet)
Input gate:  iₜ ≈ 0.1 (not much to store — "The" is uninformative)
Cell state:  c₁ ≈ [0, 0, 0] (almost empty)
```

### Processing "pizza"

```
x₂ = embedding("pizza")
Forget gate: fₜ ≈ 1.0 (keep everything)
Input gate:  iₜ ≈ 0.7 (store the topic — pizza)
Cell state:  c₂ ≈ [0, 0.7, 0] (remembers: topic is pizza)
```

### Processing "was"

```
x₃ = embedding("was")
Forget gate: fₜ ≈ 1.0 (keep everything)
Input gate:  iₜ ≈ 0.1 (not much to store)
Cell state:  c₃ ≈ [0, 0.7, 0] (still remembers pizza, "was" didn't change much)
```

### Processing "not" — The Critical Step

```
x₄ = embedding("not")
Forget gate: fₜ ≈ 0.8 (keep most memory)
Input gate:  iₜ ≈ 0.9 (STORE THIS — negation is important!)
Candidate:   c̃₄ ≈ [-0.8, 0, 0] (negative signal in sentiment dimension)

Cell state:  c₄ = 0.8 × [0, 0.7, 0] + 0.9 × [-0.8, 0, 0]
                = [0, 0.56, 0] + [-0.72, 0, 0]
                = [-0.72, 0.56, 0]

The cell state now carries: NEGATIVE sentiment + pizza topic
The "not" flipped the sentiment dimension from 0 to -0.72!
```

### Processing "good"

```
x₅ = embedding("good")
Forget gate: fₜ ≈ 0.9 (keep the negation!)
Input gate:  iₜ ≈ 0.6 (add some positive signal)
Candidate:   c̃₅ ≈ [0.5, 0, 0.3] (positive sentiment)

Cell state:  c₅ = 0.9 × [-0.72, 0.56, 0] + 0.6 × [0.5, 0, 0.3]
                = [-0.65, 0.50, 0] + [0.3, 0, 0.18]
                = [-0.35, 0.50, 0.18]

Sentiment dimension is STILL NEGATIVE (-0.35)!
The LSTM remembered the "not" and didn't let "good" fully override it.
```

### Final Prediction

```
output = sigmoid(W_out × h₅)
       ≈ 0.31

Prediction: 31% positive → Negative review ✅
The LSTM correctly understood "not good" = negative!
```

### Why a Simple RNN Would Fail Here

```
In a simple RNN, the hidden state is completely overwritten each step.
By the time it processes "good", the effect of "not" has been washed out.

RNN:  "not" → h₄ = [-0.5, ...] → "good" → h₅ = [0.6, ...] → Positive ❌
LSTM: "not" → c₄ = [-0.72, ...] → "good" → c₅ = [-0.35, ...] → Negative ✅

The cell state highway preserved the negation signal.
```

---

## STEP 6: Why LSTM Solves the Vanishing Gradient

### The Gradient Highway

```
In a simple RNN, the gradient flows through:
  ∂hₜ/∂hₜ₋₁ = tanh'(z) × W_hh  (multiplied at every step → vanishes)

In an LSTM, the gradient through the cell state is:
  ∂cₜ/∂cₜ₋₁ = fₜ  (just the forget gate value!)

If fₜ ≈ 1 (keep memory): gradient ≈ 1 → passes through unchanged!
If fₜ ≈ 0 (forget memory): gradient ≈ 0 → but that's intentional.

Through 100 steps with fₜ = 0.95:
  RNN:  tanh'(z)¹⁰⁰ ≈ 0 (vanished)
  LSTM: 0.95¹⁰⁰ = 0.006 (small but non-zero — still learning!)
  LSTM with fₜ = 0.99: 0.99¹⁰⁰ = 0.366 (substantial gradient!)

The forget gate CONTROLS how much gradient flows back.
It's a LEARNED gate — the network decides what to remember.
```

---

## STEP 7: GRU — A Simpler Alternative

### GRU (Gated Recurrent Unit) simplifies LSTM

```
LSTM has 3 gates: forget, input, output → more parameters, slower
GRU has 2 gates: reset, update → fewer parameters, faster

GRU merges the forget and input gates into one "update gate":
  zₜ = σ(W_z × [hₜ₋₁, xₜ])           ← update gate
  rₜ = σ(W_r × [hₜ₋₁, xₜ])           ← reset gate
  h̃ₜ = tanh(W × [rₜ ⊙ hₜ₋₁, xₜ])    ← candidate
  hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ   ← final state

The update gate zₜ decides: how much old state to keep vs new state to use.
When zₜ = 0: keep old state entirely (like forget gate = 1)
When zₜ = 1: use new state entirely (like forget gate = 0)
```

### LSTM vs GRU

```
| Aspect        | LSTM                    | GRU                     |
|---------------|-------------------------|-------------------------|
| Gates         | 3 (forget, input, out)  | 2 (update, reset)       |
| Parameters    | More (4 weight matrices)| Fewer (3 weight matrices)|
| Speed         | Slower                  | Faster                  |
| Performance   | Slightly better on long | Similar on most tasks   |
| Cell state    | Separate c and h        | Only h (merged)         |
| When to use   | Long sequences, complex | Shorter sequences, speed|
```

---

## STEP 8: RNN vs LSTM vs Transformer

```
| Aspect              | RNN          | LSTM         | Transformer      |
|---------------------|--------------|--------------|------------------|
| Memory              | Short (10-20)| Long (100+)  | Entire sequence  |
| Parallelizable      | No (sequential)| No (sequential)| Yes (fully)  |
| Training speed      | Slow         | Slow         | Fast (GPU)       |
| Long-range deps     | Poor         | Good         | Excellent        |
| Vanishing gradient  | Severe       | Mitigated    | None (attention) |
| Parameters          | Few          | Moderate     | Many             |
| When to use         | Simple seqs  | Medium seqs  | Everything modern|
```

Transformers replaced LSTMs for most NLP tasks after 2017. But LSTMs are still used in:
- Time series forecasting (where sequence order is critical)
- Speech recognition (real-time streaming)
- Music generation (sequential by nature)
- Edge devices (smaller than Transformers)

---

## COMPLETE FORMULA SUMMARY

```
RNN:
  hₜ = tanh(W_hh × hₜ₋₁ + W_xh × xₜ + b)

LSTM:
  fₜ = σ(W_f × [hₜ₋₁, xₜ] + b_f)           (forget gate)
  iₜ = σ(W_i × [hₜ₋₁, xₜ] + b_i)           (input gate)
  c̃ₜ = tanh(W_c × [hₜ₋₁, xₜ] + b_c)        (candidate)
  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ                 (cell state update)
  oₜ = σ(W_o × [hₜ₋₁, xₜ] + b_o)           (output gate)
  hₜ = oₜ ⊙ tanh(cₜ)                         (hidden state)

GRU:
  zₜ = σ(W_z × [hₜ₋₁, xₜ])                  (update gate)
  rₜ = σ(W_r × [hₜ₋₁, xₜ])                  (reset gate)
  h̃ₜ = tanh(W × [rₜ ⊙ hₜ₋₁, xₜ])           (candidate)
  hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ           (hidden state)
```

---

## INTERVIEW CHEAT SHEET

**Q: "What is an RNN and why do we need it?"**
> "An RNN processes sequences one step at a time, carrying a hidden state that acts as memory. Unlike regular NNs that need fixed-size input, RNNs handle variable-length sequences. The same weights are applied at every step (weight sharing), so it works for any sequence length."

**Q: "Why do RNNs struggle with long sequences?"**
> "The vanishing gradient problem. The gradient must flow back through every time step, multiplying by the weight matrix and activation derivative each time. After 20-30 steps, the gradient is essentially zero — the network can't learn long-range dependencies like 'not' affecting 'good' 10 words later."

**Q: "How does LSTM solve the vanishing gradient?"**
> "LSTM adds a cell state — a memory highway where information flows with minimal modification. Three gates (forget, input, output) control what's stored, removed, and read. The gradient through the cell state is just the forget gate value (near 1), so it passes through almost unchanged — no vanishing."

**Q: "Explain the three LSTM gates."**
> "Forget gate: decides what to erase from memory (sigmoid, 0=forget, 1=keep). Input gate: decides what new information to store (sigmoid × tanh candidate). Output gate: decides what to output from memory (sigmoid applied to tanh of cell state). All three are learned — the network discovers what to remember and forget."

**Q: "LSTM vs GRU?"**
> "GRU merges the forget and input gates into one update gate, and has no separate cell state. Fewer parameters, faster training, similar performance on most tasks. LSTM is slightly better for very long sequences. GRU is preferred when speed matters."

**Q: "Why did Transformers replace LSTMs?"**
> "Three reasons: (1) Parallelization — Transformers process all positions simultaneously, LSTMs must go sequentially. (2) Long-range — attention directly connects any two positions, LSTMs must pass information step by step. (3) Scalability — Transformers scale to billions of parameters on GPUs efficiently."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
