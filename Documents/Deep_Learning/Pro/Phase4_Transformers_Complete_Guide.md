# Transformers & Attention: Complete Guide with Example Data at Every Step

## The Problem

You want to understand the architecture behind GPT, BERT, and every modern LLM.
We'll use our pizza reviews to explain every concept.

### Our Data

| Review | Text |
|--------|------|
| R1 | "The pizza was great but the delivery was slow" |

**The question:** How does a Transformer understand that "great" refers to "pizza" and "slow" refers to "delivery" — even though they're far apart in the sentence?

---

## STEP 1: Why Transformers? The Problem with Previous Approaches

### Bag of Words / TF-IDF: No Word Order

```
"The pizza was great" and "Great was the pizza"
→ Identical BoW vectors! Word order is completely lost.
```

### RNNs / LSTMs: Sequential but Slow

```
RNN processes words one at a time, left to right:
  "The" → "pizza" → "was" → "great" → "but" → "the" → "delivery" → "was" → "slow"

Problems:
1. SLOW — can't parallelize (each word depends on the previous)
2. FORGETS — by the time it reaches "slow", it may have forgotten "pizza"
3. Long-range dependencies are hard (word 1 affecting word 50)
```

### Transformers: Parallel and Attentive

```
Transformer processes ALL words simultaneously:
  ["The", "pizza", "was", "great", "but", "the", "delivery", "was", "slow"]
  
  Every word can directly attend to every other word.
  "slow" can directly look at "delivery" (and ignore "pizza").
  "great" can directly look at "pizza" (and ignore "delivery").
  
  No sequential bottleneck. Fully parallelizable. Handles long text.
```

---

## STEP 2: Self-Attention — The Core Mechanism

### The Intuition

For each word, ask: **"Which other words in this sentence should I pay attention to?"**

```
"The pizza was great but the delivery was slow"

For "great": pay attention to "pizza" (what was great?)
For "slow":  pay attention to "delivery" (what was slow?)
For "was":   could attend to "pizza" or "delivery" (depends on context)
```

### The Three Vectors: Query, Key, Value

Every word gets transformed into three vectors:

```
Query (Q): "What am I looking for?"
Key (K):   "What do I contain?"
Value (V): "What information do I provide?"

Analogy — searching a library:
  Query = your search question ("What was great?")
  Key   = book titles (each word's label)
  Value = book contents (each word's actual information)

You compare your Query against all Keys to find the best match,
then read the Values of the matching books.
```

### Computing Q, K, V

Each word starts as an embedding vector. We multiply by three weight matrices to get Q, K, V:

```
Word embedding for "pizza": e = [0.5, 0.3, 0.8, 0.1]  (4-dimensional)

Q = e × W_Q = [0.5, 0.3, 0.8, 0.1] × W_Q = [0.2, 0.9]  (2-dimensional)
K = e × W_K = [0.5, 0.3, 0.8, 0.1] × W_K = [0.7, 0.4]
V = e × W_V = [0.5, 0.3, 0.8, 0.1] × W_V = [0.3, 0.8]

W_Q, W_K, W_V are LEARNED during training.
The network learns WHAT to query, WHAT to advertise as keys, and WHAT to provide as values.
```

---

## STEP 3: Attention Scores — Step by Step

### Simplified Example (3 words)

Let's compute attention for: "pizza was great"

```
Word embeddings (simplified to 2D):
  pizza = [1.0, 0.0]
  was   = [0.0, 0.5]
  great = [0.8, 0.2]
```

### Step 3a: Compute Q, K, V for each word

Using simple weight matrices:
```
W_Q = [[1, 0], [0, 1]]  (identity for simplicity)
W_K = [[1, 0], [0, 1]]
W_V = [[1, 0], [0, 1]]

Q_pizza = [1.0, 0.0],  K_pizza = [1.0, 0.0],  V_pizza = [1.0, 0.0]
Q_was   = [0.0, 0.5],  K_was   = [0.0, 0.5],  V_was   = [0.0, 0.5]
Q_great = [0.8, 0.2],  K_great = [0.8, 0.2],  V_great = [0.8, 0.2]
```

### Step 3b: Compute Attention Scores (Q × K^T)

For "great" attending to all words:
```
Score(great→pizza) = Q_great · K_pizza = 0.8×1.0 + 0.2×0.0 = 0.80
Score(great→was)   = Q_great · K_was   = 0.8×0.0 + 0.2×0.5 = 0.10
Score(great→great) = Q_great · K_great = 0.8×0.8 + 0.2×0.2 = 0.68

Raw scores: [0.80, 0.10, 0.68]
"great" attends most to "pizza" (0.80) — it learned what was great!
```

### Step 3c: Scale by √d_k

```
d_k = dimension of key vectors = 2
√d_k = √2 = 1.414

Scaled scores: [0.80/1.414, 0.10/1.414, 0.68/1.414]
             = [0.566, 0.071, 0.481]

Why scale? Without scaling, dot products grow large with dimension,
pushing softmax into regions with tiny gradients (vanishing gradient problem).
```

### Step 3d: Apply Softmax (convert to probabilities)

```
Softmax([0.566, 0.071, 0.481]):
  e^0.566 = 1.761
  e^0.071 = 1.074
  e^0.481 = 1.618
  Sum = 4.453

  Attention weights:
    pizza: 1.761/4.453 = 0.395
    was:   1.074/4.453 = 0.241
    great: 1.618/4.453 = 0.363

"great" pays 39.5% attention to "pizza", 24.1% to "was", 36.3% to itself.
```

### Step 3e: Weighted Sum of Values

```
Output for "great" = 0.395 × V_pizza + 0.241 × V_was + 0.363 × V_great
                   = 0.395 × [1.0, 0.0] + 0.241 × [0.0, 0.5] + 0.363 × [0.8, 0.2]
                   = [0.395, 0.0] + [0.0, 0.121] + [0.290, 0.073]
                   = [0.685, 0.194]

The output for "great" is now a BLEND of information from all words,
weighted by how relevant each word is. It absorbed context from "pizza"!
```

### The Complete Attention Formula

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

This single formula is the entire self-attention mechanism.
Everything else in a Transformer is built around this.
```

---

## STEP 4: Multi-Head Attention — Looking at Multiple Things

### Why Multiple Heads?

```
One attention head might learn: "what was described?" (great→pizza)
Another head might learn: "what was the action?" (was→pizza)
Another might learn: "what's the sentiment?" (great→positive)

Multiple heads = multiple perspectives on the same text.
```

### How It Works

```
Instead of one set of Q, K, V:
  Head 1: Q₁, K₁, V₁ → Attention₁  (learns syntactic relationships)
  Head 2: Q₂, K₂, V₂ → Attention₂  (learns semantic relationships)
  Head 3: Q₃, K₃, V₃ → Attention₃  (learns positional relationships)
  ...

Concatenate all heads → Linear projection → Output

In BERT-base: 12 heads, each with d_k = 64 (total dimension = 768)
In GPT-3:    96 heads, each with d_k = 128 (total dimension = 12288)
```

---

## STEP 5: The Full Transformer Block

### One Transformer Block

```
Input
  ↓
[Multi-Head Self-Attention]
  ↓
[Add & Normalize]          ← Residual connection + Layer Norm
  ↓
[Feed-Forward Network]     ← Two linear layers with ReLU
  ↓
[Add & Normalize]          ← Another residual connection
  ↓
Output
```

### Why Residual Connections?

```
Without residual: output = f(input)
With residual:    output = f(input) + input

The "+ input" part means: "keep the original information and ADD new information."
This prevents the vanishing gradient problem in deep networks.
If f(input) learns nothing useful, the output is just the input — no harm done.

GPT-3 has 96 Transformer blocks stacked. Without residual connections,
gradients would vanish by the time they reach the first block.
```

### Why Layer Normalization?

```
Normalizes activations to have mean=0, variance=1 within each layer.
Keeps numbers in a reasonable range → stable training.
Similar to Batch Normalization but works better for sequences.
```

---

## STEP 6: Positional Encoding — Word Order Matters

### The Problem

```
Self-attention treats all words equally — it doesn't know word ORDER.
"Pizza great was" and "Was pizza great" would get identical attention scores.

But order matters! "The pizza was great" ≠ "Great was the pizza" (emphasis differs)
```

### The Solution: Add Position Information

```
Word embedding:     "pizza" → [0.5, 0.3, 0.8, 0.1]
Position encoding:  position 2 → [0.0, 1.0, 0.0, 0.1]  (computed from sine/cosine)

Final input: [0.5, 0.3, 0.8, 0.1] + [0.0, 1.0, 0.0, 0.1] = [0.5, 1.3, 0.8, 0.2]

Now "pizza" at position 2 has a different vector than "pizza" at position 5.
The model can learn that position matters.
```

### Why Sine/Cosine?

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Why this specific formula?
1. Each position gets a unique encoding
2. The model can learn relative positions (PE(pos+k) is a linear function of PE(pos))
3. Works for any sequence length (no maximum position)
```

---

## STEP 7: BERT vs GPT — Two Sides of the Transformer

### Encoder (BERT): Understands Text

```
Input:  "The pizza was [MASK] but the delivery was slow"
Output: "The pizza was GREAT but the delivery was slow"

BERT sees ALL words at once (bidirectional).
It's trained to fill in blanks (Masked Language Modeling).
Good for: classification, NER, question answering, embeddings.
```

### Decoder (GPT): Generates Text

```
Input:  "The pizza was"
Output: "The pizza was great"  (predicts next word)

GPT only sees words to the LEFT (autoregressive).
It's trained to predict the next word.
Good for: text generation, chatbots, code generation.
```

### The Key Difference

```
BERT (Encoder):
  "The pizza was [?] but the delivery was slow"
  Can see: ← ALL words → (bidirectional)
  Knows "slow" when predicting [?] — uses full context.

GPT (Decoder):
  "The pizza was [?]"
  Can see: ← only left words (autoregressive)
  Doesn't know what comes after — generates one word at a time.

BERT = understanding (reads the whole book, then answers questions)
GPT  = generation (writes the book one word at a time)
```

---

## STEP 8: How LLMs Scale

```
| Model      | Parameters | Layers | Heads | Dimension | Training Data |
|------------|-----------|--------|-------|-----------|---------------|
| BERT-base  | 110M      | 12     | 12    | 768       | 16GB text     |
| BERT-large | 340M      | 24     | 16    | 1024      | 16GB text     |
| GPT-2      | 1.5B      | 48     | 25    | 1600      | 40GB text     |
| GPT-3      | 175B      | 96     | 96    | 12288     | 570GB text    |
| GPT-4      | ~1.8T*    | ~120*  | ~128* | ~16384*   | ~13T tokens*  |

*estimated, not officially confirmed

The pattern: more layers, more heads, more dimensions, more data → better performance.
This is the "scaling law" — performance improves predictably with scale.
```

---

## COMPLETE FORMULA SUMMARY

```
1. Attention:       Attention(Q,K,V) = softmax(QK^T/√d_k) × V
2. Multi-head:      MultiHead = Concat(head₁,...,headₕ) × W_O
3. Each head:       headᵢ = Attention(QW_Qⁱ, KW_Kⁱ, VW_Vⁱ)
4. Position:        PE(pos,2i) = sin(pos/10000^(2i/d))
5. Transformer:     output = LayerNorm(x + Attention(x)) 
                    output = LayerNorm(output + FFN(output))
6. FFN:             FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

---

## INTERVIEW CHEAT SHEET

**Q: "Explain self-attention."**
> "Each word creates a Query (what am I looking for?), Key (what do I contain?), and Value (what info do I provide). We compute dot products between the Query of one word and Keys of all words to get attention scores, apply softmax to get weights, then take a weighted sum of Values. This lets each word gather relevant context from the entire sequence."

**Q: "Why scale by √d_k?"**
> "Without scaling, dot products grow proportionally to the dimension d_k. Large dot products push softmax into saturated regions where gradients are near zero. Dividing by √d_k keeps the variance of dot products at 1 regardless of dimension, ensuring softmax produces useful gradients."

**Q: "BERT vs GPT?"**
> "Both use Transformers but differently. BERT is an encoder — it sees all words bidirectionally and is trained with masked language modeling (fill in blanks). Good for understanding tasks like classification and NER. GPT is a decoder — it sees only left context and is trained to predict the next word. Good for generation tasks like chatbots and code."

**Q: "Why are Transformers better than RNNs?"**
> "Three reasons: (1) Parallelization — Transformers process all words simultaneously, RNNs process sequentially. (2) Long-range dependencies — attention directly connects any two words regardless of distance, RNNs struggle with long sequences. (3) Scalability — Transformers scale efficiently to billions of parameters on modern GPUs."

**Q: "What are positional encodings and why are they needed?"**
> "Self-attention is permutation-invariant — it doesn't know word order. Positional encodings add position information to word embeddings using sine/cosine functions. This lets the model distinguish 'pizza was great' from 'great was pizza'. The sine/cosine choice allows the model to learn relative positions and generalize to unseen sequence lengths."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
