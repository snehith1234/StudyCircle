# Transformers and Attention: Visual Guide with Mermaid Diagrams

> Visual companion to `Documents/Deep_Learning/Pro/Phase4_Transformers_Complete_Guide.md`.
> Every diagram has explanatory text — what it shows, why it matters, and how to read it.

---

## 1. Why Transformers — The Evolution of Sequence Models

Bag of Words loses word order entirely. RNNs process words one at a time — slow and forgetful over long sequences. Transformers process all words simultaneously, letting every word directly attend to every other word. No sequential bottleneck, no forgetting, fully parallelizable.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    BOW["Bag of Words / TFIDF<br/>No word order at all<br/><i>great pizza = pizza great</i>"] --> PROBLEM1["Loses structure"]
    RNN["RNN / LSTM<br/>Sequential, one word at a time<br/><i>Slow, forgets long range</i>"] --> PROBLEM2["Slow and forgetful"]
    TRANS["Transformer<br/>All words in parallel<br/><i>Every word sees every word</i>"] --> GOOD["Fast, parallel, long range"]

    style BOW fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style RNN fill:#252840,stroke:#f5b731,color:#c8cfe0
    style TRANS fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style PROBLEM1 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style PROBLEM2 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style GOOD fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Red = broken approaches (BoW loses order, RNN is slow). Yellow = RNN (works but limited). Green = Transformer (solves all three problems). The key breakthrough: attention lets "slow" directly look at "delivery" even if they are 5 words apart, without processing every word in between.

---

## 2. Self-Attention: Query, Key, Value

For each word, self-attention asks: "Which other words should I pay attention to?" Every word is transformed into three vectors — Query (what am I looking for?), Key (what do I contain?), Value (what information do I provide). The embedding (0.5, 0.3, 0.8, 0.1) is a simplified 4-dimensional example. In real models, embeddings are much larger — BERT uses 768 dimensions, GPT-3 uses 12,288. These numbers are learned during training: each word gets a unique vector that captures its meaning. Similar words (like 'pizza' and 'pasta') end up with similar vectors. Think of it like a library: your Query is the search question, Keys are book titles, Values are book contents. You match your question against all titles, then read the best-matching books.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    WORD["Word Embedding<br/>pizza = (0.5, 0.3, 0.8, 0.1)"] --> Q["Query (Q)<br/>What am I looking for?<br/>e x W_Q<br/><i>Search question</i>"]
    WORD --> K["Key (K)<br/>What do I contain?<br/>e x W_K<br/><i>Book title</i>"]
    WORD --> V["Value (V)<br/>What info do I provide?<br/>e x W_V<br/><i>Book contents</i>"]

    style WORD fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style K fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style V fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = the word embedding (starting point). Blue = Query (what this word is searching for). Purple = Key (what this word advertises about itself). Green = Value (the actual information this word carries). W_Q, W_K, W_V are learned weight matrices — the network learns what to query, what to advertise, and what to provide.

---

## 3. Attention Score Computation

The attention mechanism in four steps: (1) dot product of Query with all Keys to get raw scores, (2) scale by square root of dimension to prevent gradient issues, (3) softmax to convert to probabilities, (4) weighted sum of Values. The result: each word gets a new representation that blends information from the most relevant words.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    DOT["Step 1: Q dot K transpose<br/>Raw attention scores<br/><i>How relevant is each word?</i>"]
    DOT --> SCALE["Step 2: Divide by sqrt(d_k)<br/>Scaled scores<br/><i>Prevents gradient saturation</i>"]
    SCALE --> SOFT["Step 3: Softmax<br/>Convert to probabilities<br/><i>Weights sum to 1.0</i>"]
    SOFT --> WSUM["Step 4: Weighted sum of V<br/>Blend relevant information<br/><i>Context-aware output</i>"]

    style DOT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style SCALE fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style SOFT fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style WSUM fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = raw dot products (measure similarity). Blue = scaling (numerical stability). Purple = softmax (normalize to probabilities). Green = weighted sum (the final context-aware output). The complete formula: Attention(Q,K,V) = softmax(Q x K_T / sqrt(d_k)) x V.

### Example: "great" Attending to Other Words

When computing attention for the word "great" in "pizza was great", these attention weights (0.395 for pizza, 0.241 for was, 0.363 for self) come from the softmax of the scaled dot products computed in the steps above. They sum to 1.0 and represent how much each word contributes to the output for 'great'. The model learned that 'pizza' is most relevant to 'great' (39.5%) because they have the highest dot product similarity. The scores show it pays most attention to "pizza" — it learned what was great.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    GREAT["great<br/>Query word"] --> P["pizza<br/>weight: 0.395<br/><i>Highest attention</i>"]
    GREAT --> W["was<br/>weight: 0.241<br/><i>Lower attention</i>"]
    GREAT --> G["great<br/>weight: 0.363<br/><i>Self attention</i>"]

    style GREAT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style P fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style W fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style G fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
```

Yellow = the query word. Green = highest attention (pizza, 39.5%). Blue/Purple = lower attention words. The output for "great" becomes a blend: 39.5% pizza info, 24.1% was info, 36.3% self info. It absorbed context from "pizza" — now it knows what was great.

---

## 4. Multi-Head Attention

One attention head captures one type of relationship. But language has many simultaneous relationships — syntactic, semantic, positional. Multi-head attention runs several attention mechanisms in parallel, each with its own Q, K, V weights. The outputs are concatenated and projected. BERT uses 12 heads, GPT-3 uses 96.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    INPUT["Input Embeddings"] --> H1["Head 1<br/>Q1, K1, V1<br/><i>Syntactic relations</i>"]
    INPUT --> H2["Head 2<br/>Q2, K2, V2<br/><i>Semantic relations</i>"]
    INPUT --> H3["Head 3<br/>Q3, K3, V3<br/><i>Positional relations</i>"]
    INPUT --> HN["Head N...<br/>QN, KN, VN<br/><i>Other patterns</i>"]
    H1 --> CONCAT["Concatenate all heads<br/>then linear projection"]
    H2 --> CONCAT
    H3 --> CONCAT
    HN --> CONCAT
    CONCAT --> OUT["Multi-Head Output<br/><i>Rich, multi-perspective</i>"]

    style INPUT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style H1 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style H2 fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style H3 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style HN fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style CONCAT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style OUT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = input and concatenation. Blue/Purple = individual attention heads (each learns different relationships). Green = final multi-perspective output. Each head operates on a smaller dimension (BERT: 768/12 = 64 per head), so the total computation is similar to single-head attention but captures richer patterns.

---

## 5. Full Transformer Block

A Transformer block has four components: multi-head attention, add and normalize, feed-forward network, add and normalize again. The residual connections (add) prevent vanishing gradients — if a layer learns nothing useful, the signal passes through unchanged. Layer normalization keeps values stable. GPT-3 stacks 96 of these blocks.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    IN["Input"] --> ATTN["Multi-Head Self-Attention<br/><i>Each word attends to all words</i>"]
    ATTN --> AN1["Add and Normalize<br/>output = LayerNorm(x plus Attn(x))<br/><i>Residual connection</i>"]
    AN1 --> FFN["Feed-Forward Network<br/>Two linear layers with ReLU<br/><i>Process each position</i>"]
    FFN --> AN2["Add and Normalize<br/>output = LayerNorm(x plus FFN(x))<br/><i>Another residual</i>"]
    AN2 --> OUT["Output<br/><i>To next Transformer block</i>"]

    style IN fill:#252840,stroke:#f5b731,color:#c8cfe0
    style ATTN fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style AN1 fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style FFN fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style AN2 fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style OUT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = input/output. Blue = the two main operations (attention and FFN). Purple = add and normalize (residual connections with layer norm). Green = output to next block. The residual connections are critical — without them, gradients would vanish through 96 layers. With them, gradients flow directly through the skip paths.

---

## 6. Positional Encoding — Why Word Order Matters

Self-attention is permutation-invariant — it treats "pizza was great" and "great was pizza" identically because it has no concept of position. Positional encoding fixes this by adding a unique position signal to each word embedding using sine and cosine functions. Now the model knows that "pizza" at position 1 is different from "pizza" at position 5.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    EMB["Word Embedding<br/>pizza = (0.5, 0.3, 0.8, 0.1)<br/><i>No position info</i>"]
    POS["Position Encoding<br/>pos 2 = (0.0, 1.0, 0.0, 0.1)<br/><i>Sine and cosine values</i>"]
    EMB --> ADD["Element-wise Addition<br/>(0.5, 1.3, 0.8, 0.2)<br/><i>Now position-aware</i>"]
    POS --> ADD

    style EMB fill:#252840,stroke:#f5b731,color:#c8cfe0
    style POS fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style ADD fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = word embedding (meaning only). Blue = positional encoding (position only). Green = the sum (meaning plus position). The sine/cosine formula gives three properties: (1) each position gets a unique encoding, (2) the model can learn relative positions, (3) it generalizes to any sequence length.

### Why Sine and Cosine?

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    FORMULA["PE(pos, 2i) = sin(pos / 10000 to (2i/d))<br/>PE(pos, 2i plus 1) = cos(pos / 10000 to (2i/d))"]
    FORMULA --> P1["Unique per position<br/>No two positions share encoding"]
    FORMULA --> P2["Relative positions learnable<br/>PE(pos plus k) is linear fn of PE(pos)"]
    FORMULA --> P3["Generalizes to any length<br/>No maximum sequence limit"]

    style FORMULA fill:#252840,stroke:#f5b731,color:#c8cfe0
    style P1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style P2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style P3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = the formula. Green = the three key properties. The different frequencies (controlled by the dimension index i) create a unique "fingerprint" for each position, similar to how binary numbers use different bit positions.

---

## 7. BERT vs GPT — Encoder vs Decoder

Both use Transformers but in opposite directions. BERT is an encoder — it sees all words bidirectionally and fills in blanks (masked language modeling). GPT is a decoder — it sees only left context and predicts the next word (autoregressive). BERT understands text. GPT generates text.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    BERT["BERT (Encoder)<br/>Sees ALL words both directions<br/>Trained: fill in masked words<br/><i>Bidirectional</i>"]
    GPT["GPT (Decoder)<br/>Sees only LEFT context<br/>Trained: predict next word<br/><i>Autoregressive</i>"]
    BERT --> BU["Understanding tasks<br/>Classification, NER<br/>Question answering<br/>Sentence embeddings"]
    GPT --> GU["Generation tasks<br/>Chatbots, text completion<br/>Code generation<br/>Creative writing"]

    style BERT fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style GPT fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style BU fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style GU fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Blue = BERT (encoder, bidirectional). Purple = GPT (decoder, left-to-right). Green = their respective strengths.

### How They See Text

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    B1["BERT sees:<br/>The pizza was MASK but<br/>the delivery was slow<br/><i>Uses ALL words to predict MASK</i>"]
    G1["GPT sees:<br/>The pizza was ___<br/><i>Only left context</i><br/>Predicts: great"]

    style B1 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style G1 fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
```

Blue = BERT reads the whole sentence including "slow" when predicting the masked word. Purple = GPT only sees "The pizza was" and must guess what comes next. BERT is like reading a book then answering questions. GPT is like writing a book one word at a time.

---

## 8. How LLMs Scale

Larger models with more parameters, more layers, more attention heads, and more training data consistently perform better. This is the scaling law — performance improves predictably with scale. GPT-3 has 175 billion parameters across 96 layers with 96 attention heads, trained on 570GB of text.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    SMALL["BERT-base<br/>110M params<br/>12 layers, 12 heads<br/>dim 768"] --> MED["GPT-2<br/>1.5B params<br/>48 layers, 25 heads<br/>dim 1600"]
    MED --> LARGE["GPT-3<br/>175B params<br/>96 layers, 96 heads<br/>dim 12288"]
    LARGE --> HUGE["GPT-4 (estimated)<br/>approx 1.8T params<br/>approx 120 layers<br/>dim approx 16384"]

    style SMALL fill:#252840,stroke:#f5b731,color:#c8cfe0
    style MED fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style LARGE fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style HUGE fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = small (BERT-base, 110M). Blue = medium (GPT-2, 1.5B). Purple = large (GPT-3, 175B). Green = massive (GPT-4, estimated 1.8T). The pattern: more layers, more heads, larger dimensions, more data. Each jump is roughly 100x more parameters.

### What Scales

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    SCALE["Scaling Laws"] --> PARAMS["More Parameters<br/>Wider layers, more weights<br/><i>Model capacity</i>"]
    SCALE --> LAYERS["More Layers<br/>Deeper network<br/><i>More abstraction levels</i>"]
    SCALE --> DATA["More Training Data<br/>Larger text corpus<br/><i>More knowledge encoded</i>"]
    SCALE --> COMPUTE["More Compute<br/>More GPUs, longer training<br/><i>Better optimization</i>"]

    style SCALE fill:#252840,stroke:#f5b731,color:#c8cfe0
    style PARAMS fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style LAYERS fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style DATA fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style COMPUTE fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = the scaling principle. Blue/Purple = model architecture scaling. Green = data and compute scaling. All four dimensions must scale together — a huge model with little data overfits, and a small model with huge data plateaus.

---

## 9. Interview Decision Tree

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    Q1{"Explain self-attention?"} -->|Answer| A1["Q dot K for relevance scores<br/>Scale by sqrt(d_k), softmax<br/>Weighted sum of Values"]
    Q1 -->|Next Q| Q2{"Why scale by sqrt(d_k)?"}
    Q2 -->|Answer| A2["Large dot products saturate softmax<br/>Gradients vanish in saturated regions<br/>Scaling keeps variance at 1"]
    Q2 -->|Next Q| Q3{"BERT vs GPT?"}
    Q3 -->|Answer| A3["BERT: encoder, bidirectional<br/>GPT: decoder, autoregressive<br/>Understanding vs generation"]
    Q3 -->|Next Q| Q4{"Why Transformers<br/>over RNNs?"}
    Q4 -->|Answer| A4["Parallel processing, not sequential<br/>Direct long-range connections<br/>Scales to billions of params"]
    Q4 -->|Next Q| Q5{"What are positional<br/>encodings?"}
    Q5 -->|Answer| A5["Attention has no position sense<br/>Add sine/cosine per position<br/>Unique, relative, generalizable"]

    style Q1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q5 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A4 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A5 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

---

> 💡 **How to view:** GitHub (native), VS Code (Mermaid extension), Obsidian (built-in), or [mermaid.live](https://mermaid.live)
