# RNNs & LSTMs: Visual Guide with Mermaid Diagrams

> Visual companion to `Documents/Deep_Learning/Next_Level/RNN_LSTM_Complete_Guide.md`.
> Every diagram has explanatory text — what it shows, why it matters, and how to read it.

---

## 1. Why We Need RNNs — Regular NNs Can't Handle Sequences

Regular neural networks need fixed-size input. But sentences have variable length — 4 words or 40 words. Worse, regular NNs treat inputs as a bag with no order. "Not good" and "good not" would be identical. RNNs solve this by processing one word at a time, carrying a hidden state (memory) from step to step.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    NN["Regular NN<br/>Fixed size input only<br/>No concept of order<br/><i>not good = good not</i>"]
    RNN["RNN<br/>Variable length sequences<br/>Processes one step at a time<br/><i>Carries memory forward</i>"]

    NN --> BAD["Cannot handle text,<br/>time series, audio, video"]
    RNN --> GOOD["Handles any sequence length<br/>Remembers what came before"]

    style NN fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style RNN fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style BAD fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style GOOD fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Red = regular NN (can't do sequences). Green = RNN (designed for sequences). The hidden state is the key innovation — it's a vector that gets updated at each step, carrying information from all previous steps.

---

## 2. How an RNN Processes a Sequence

The RNN reads one word at a time, left to right. At each step, it combines the current word with its memory of all previous words to produce an updated memory. The same weights are used at every step (weight sharing) — so it works for any sequence length. The final hidden state summarizes the entire sentence.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    X1["great"] --> R1["RNN Cell<br/>h₁ = tanh(W×x₁)"]
    R1 -->|"h₁ = memory<br/>of 'great'"| R2["RNN Cell<br/>h₂ = tanh(W×h₁,x₂)"]
    X2["pizza"] --> R2
    R2 -->|"h₂ = memory of<br/>'great pizza'"| R3["RNN Cell<br/>h₃ = tanh(W×h₂,x₃)"]
    X3["fast"] --> R3
    R3 -->|"h₃ = memory of<br/>'great pizza fast'"| R4["RNN Cell<br/>h₄ = tanh(W×h₃,x₄)"]
    X4["delivery"] --> R4
    R4 --> OUT["h₄ → sigmoid<br/>= 82% positive ✅"]

    style X1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style X2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style X3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style X4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style R1 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style R2 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style R3 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style R4 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style OUT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = input words (one per step). Blue = RNN cells (same weights W at every step). The arrows between cells carry the hidden state h — each one accumulates more context. Green = final prediction using the last hidden state. The formula at each step: hₜ = tanh(W_hh × hₜ₋₁ plus W_xh × xₜ plus b).

---

## 3. The RNN's Fatal Flaw — Vanishing Gradients Over Time

An RNN processing a 100-word sentence is like a 100-layer deep network. The gradient must flow back through every time step. With tanh activation (max gradient = 1.0), the gradient shrinks at each step. After 20-30 steps, early words are effectively forgotten — the network can't learn that "not" (word 5) should affect "good" (word 8).

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    W5["Word 50<br/>gradient = 1.0<br/><i>Learns fine</i>"] --> W4["Word 40<br/>gradient shrinks"]
    W4 --> W3["Word 30<br/>gradient smaller"]
    W3 --> W2["Word 20<br/>gradient tiny"]
    W2 --> W1["Word 1<br/>gradient ≈ 0<br/><i>Forgotten!</i>"]

    style W5 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style W4 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style W3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style W2 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style W1 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

Green = recent words (strong gradient). Blue/Yellow = middle words (weakening). Red = early words (gradient dead). This is why a simple RNN can't understand "The pizza that we ordered last Tuesday from the new place was great" — by the time it reaches "great", it has forgotten "pizza."

---

## 4. LSTM — The Memory Highway

LSTM solves the vanishing gradient by adding a cell state — a separate memory highway. Information flows through the cell state with only small, controlled modifications via three gates. The key difference: in an RNN, the hidden state is completely overwritten each step. In an LSTM, the cell state is only slightly modified — old information is preserved unless explicitly erased.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    RNN_WAY["RNN: Hidden state<br/>Completely overwritten each step<br/>Old info washed out by new info<br/><i>Like erasing a whiteboard every time</i>"]
    LSTM_WAY["LSTM: Cell state highway<br/>Only small modifications via gates<br/>Old info preserved unless erased<br/><i>Like a notebook you can add to or erase from</i>"]

    RNN_WAY --> SHORT["Short memory: 10 to 20 steps"]
    LSTM_WAY --> LONG["Long memory: 100 plus steps"]

    style RNN_WAY fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style LSTM_WAY fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style SHORT fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style LONG fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Red = RNN approach (overwrite, forget). Green = LSTM approach (highway, preserve). The whiteboard vs notebook analogy captures it: an RNN erases and redraws every step, an LSTM writes in a notebook and only erases specific parts when needed.

---

## 5. The Three LSTM Gates

Each gate is a sigmoid layer outputting values between 0 (block everything) and 1 (let everything through). The forget gate decides what to erase from memory. The input gate decides what new information to write. The output gate decides what to read out. All three are learned during training — the network discovers what to remember and forget.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    INPUT["Current word xₜ<br/>Previous output hₜ₋₁"] --> FG & IG & OG

    FG["🚪 Forget Gate<br/>fₜ = σ(W_f × input)<br/>What to ERASE from memory<br/><i>0 = forget, 1 = keep</i>"]
    IG["🚪 Input Gate<br/>iₜ = σ(W_i × input)<br/>What NEW info to WRITE<br/><i>0 = ignore, 1 = store</i>"]
    OG["🚪 Output Gate<br/>oₜ = σ(W_o × input)<br/>What to READ from memory<br/><i>0 = hide, 1 = output</i>"]

    FG --> CELL["📝 Cell State Update<br/>cₜ = fₜ ⊙ cₜ₋₁ plus iₜ ⊙ candidate<br/><i>Erase old, write new</i>"]
    IG --> CELL
    CELL --> HIDDEN["Hidden State<br/>hₜ = oₜ ⊙ tanh(cₜ)<br/><i>Filtered output</i>"]
    OG --> HIDDEN

    style INPUT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style FG fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style IG fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style OG fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style CELL fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style HIDDEN fill:#252840,stroke:#f5b731,color:#c8cfe0
```

Yellow = inputs and output. Red = forget gate (erases). Green = input gate (writes). Blue = output gate (reads). Purple = cell state (the memory itself). The ⊙ symbol means element-wise multiplication — each gate controls each memory dimension independently.

---

## 6. LSTM in Action — "not good"

This traces how the LSTM handles negation. When it reads "not", the input gate stores a strong negative signal. When it then reads "good", the forget gate preserves the negation — the positive signal from "good" doesn't fully override the "not." A simple RNN would overwrite the "not" memory entirely.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    NOT["Word: 'not'<br/>Input gate: 0.9 (store this!)<br/>Candidate: negative signal<br/>Cell: sentiment = <b>negative</b>"]
    NOT --> GOOD["Word: 'good'<br/>Forget gate: 0.9 (keep negation!)<br/>Input gate: 0.6 (add some positive)<br/>Cell: sentiment = <b>still negative</b>"]
    GOOD --> PRED["Final: 31% positive<br/>= Negative review ✅<br/><i>LSTM remembered the 'not'</i>"]

    NOT --> RNN_FAIL["Simple RNN would:<br/>'not' → negative state<br/>'good' → overwrites to positive<br/>= Positive review ❌"]

    style NOT fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style GOOD fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style PRED fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style RNN_FAIL fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

Blue = "not" step (stores negation). Purple = "good" step (preserves negation via forget gate). Green = LSTM gets it right. Red = simple RNN fails (overwrites the negation). The forget gate value of 0.9 means "keep 90% of the old memory" — the negation survives.

---

## 7. Why LSTM Gradients Don't Vanish

The gradient through the cell state is just the forget gate value — near 1.0 for important memories. No tanh derivative, no weight matrix multiplication. The gradient flows through the cell state highway almost unchanged, even over 100 steps.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    RNN_GRAD["RNN gradient per step<br/>= tanh'(z) × W_hh<br/>Multiplied 100 times → 0<br/><i>Vanishes</i>"]
    LSTM_GRAD["LSTM gradient per step<br/>= forget gate fₜ ≈ 0.99<br/>0.99¹⁰⁰ = 0.366<br/><i>Still substantial!</i>"]

    style RNN_GRAD fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style LSTM_GRAD fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Red = RNN gradient (vanishes through multiplication of tanh derivatives and weights). Green = LSTM gradient (controlled by forget gate, stays near 1). The cell state acts as a gradient highway — information and gradients flow through with minimal loss.

---

## 8. GRU — The Simpler Alternative

GRU simplifies LSTM by merging the forget and input gates into one update gate, and removing the separate cell state. Fewer parameters, faster training, similar performance on most tasks. LSTM is slightly better for very long sequences where the separate cell state helps.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    LSTM_BOX["LSTM<br/>3 gates: forget, input, output<br/>Separate cell state and hidden state<br/>More parameters, slightly better<br/><i>Best for very long sequences</i>"]
    GRU_BOX["GRU<br/>2 gates: update, reset<br/>Only hidden state (merged)<br/>Fewer parameters, faster<br/><i>Best when speed matters</i>"]

    style LSTM_BOX fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style GRU_BOX fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
```

Blue = LSTM (more complex, slightly more powerful). Purple = GRU (simpler, faster, usually equivalent). In practice, try GRU first — if performance isn't good enough, switch to LSTM.

---

## 9. The Evolution: RNN → LSTM → Transformer

Each generation solved a limitation of the previous one. RNNs added memory but forgot long sequences. LSTMs added gates to control memory but were still sequential (slow). Transformers replaced sequential processing with parallel attention — every word sees every other word directly.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    RNN2["RNN (1986)<br/>Memory through hidden state<br/><i>Problem: forgets after 20 steps</i>"]
    RNN2 -->|"Add gates to<br/>control memory"| LSTM2["LSTM (1997)<br/>Cell state highway with gates<br/><i>Problem: still sequential, slow</i>"]
    LSTM2 -->|"Replace sequence<br/>with attention"| TRANS2["Transformer (2017)<br/>Parallel attention, all words at once<br/><i>Enabled GPT, BERT, all modern LLMs</i>"]

    style RNN2 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style LSTM2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style TRANS2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Red = RNN (limited memory). Yellow = LSTM (good memory, slow). Green = Transformer (unlimited memory, fast). Each arrow label tells you what problem was solved. Understanding this evolution is essential — interviewers love asking "why did Transformers replace LSTMs?"

---

## 10. Interview Decision Tree 🎯

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    Q1{"What is an RNN?"} -->|Answer| A1["Processes sequences step by step<br/>Hidden state carries memory forward<br/>Same weights at every step"]
    Q1 -->|Next Q| Q2{"Why do RNNs forget?"}
    Q2 -->|Answer| A2["Vanishing gradient through time<br/>Gradient multiplied at each step<br/>After 20 steps: gradient ≈ 0"]
    Q2 -->|Next Q| Q3{"How does LSTM fix this?"}
    Q3 -->|Answer| A3["Cell state = memory highway<br/>3 gates control read/write/erase<br/>Gradient = forget gate ≈ 1"]
    Q3 -->|Next Q| Q4{"LSTM vs GRU?"}
    Q4 -->|Answer| A4["LSTM: 3 gates, separate cell state<br/>GRU: 2 gates, merged state, faster<br/>Similar performance, GRU simpler"]
    Q4 -->|Next Q| Q5{"Why Transformers<br/>replaced LSTMs?"}
    Q5 -->|Answer| A5["Parallel not sequential<br/>Direct long range connections<br/>Scales to billions of parameters"]

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
