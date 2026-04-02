# Decision Trees: Visual Guide with Mermaid Diagrams

> Visual companion to `Documents/Decision_Trees_Complete_Guide.md`.
> Every diagram has explanatory text — what it shows, why it matters, and how to read it.

---

## 1. What Is a Decision Tree?

A decision tree is a flowchart of yes/no questions that splits data into pure groups. Unlike logistic regression (which draws a single line), a tree creates rectangular regions by asking one question at a time. The diagram below shows the core idea: start at the top, answer a question, follow the branch, repeat until you reach a prediction.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    ROOT{"🌳 Is Rating > 3.8?"} -->|"Yes"| RIGHT{"Is Delivery < 30?"}
    ROOT -->|"No"| LEFT["❌ Not Successful<br/><i>Low rating stores</i>"]
    RIGHT -->|"Yes"| SUCCESS["✅ Successful<br/><i>High rating and fast delivery</i>"]
    RIGHT -->|"No"| FAIL["❌ Not Successful<br/><i>High rating but slow</i>"]

    style ROOT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style LEFT fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style SUCCESS fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style FAIL fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

Diamonds = questions (split nodes). Rectangles = predictions (leaf nodes). Green = success, Red = failure. Every data point starts at the top and follows exactly one path down to a leaf.

Why 3.8? The tree algorithm tried every possible threshold (2.95, 3.1, 3.35, 3.8, 4.2, etc.) and found that 3.8 gives the best split — it perfectly separates all successful stores (Rating > 3.8) from all failures (Rating ≤ 3.8). The algorithm discovered this automatically by computing the Gini gain for each candidate.

---

## 2. Our Pizza Store Data

### 2.1 The Data Table

| Store | Rating ⭐ | Delivery 🚚 | Successful? |
|:-----:|:---------:|:-----------:|:-----------:|
| S1    | 4.5       | 20 min      | ✅ Yes (1)  |
| S2    | 3.2       | 45 min      | ❌ No (0)   |
| S3    | 4.8       | 18 min      | ✅ Yes (1)  |
| S4    | 2.9       | 50 min      | ❌ No (0)   |
| S5    | 4.1       | 25 min      | ✅ Yes (1)  |
| S6    | 3.5       | 35 min      | ❌ No (0)   |
| S7    | 4.3       | 22 min      | ✅ Yes (1)  |
| S8    | 3.0       | 40 min      | ❌ No (0)   |

### 2.2 Scatter Plot — The Two Groups

```
  Delivery
  (min)
     55 ┤
        │  ╔══════════════════╗
     50 ┤  ║  ❌ S4 (2.9,50)  ║
     45 ┤  ║  ❌ S2 (3.2,45)  ║    FAILURE ZONE
     40 ┤  ║  ❌ S8 (3.0,40)  ║    Rating < 3.8
     35 ┤  ║  ❌ S6 (3.5,35)  ║
        │  ╚══════════════════╝
     30 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ SPLIT BOUNDARY ─ ─
        │                  ╔══════════════════╗
     25 ┤                  ║  ✅ S5 (4.1,25)  ║
     22 ┤                  ║  ✅ S7 (4.3,22)  ║    SUCCESS ZONE
     20 ┤                  ║  ✅ S1 (4.5,20)  ║    Rating > 3.8
     18 ┤                  ║  ✅ S3 (4.8,18)  ║
        │                  ╚══════════════════╝
        └──┬─────┬─────┬─────┬─────┬─────┬──→ Rating
          2.5   3.0   3.5   4.0   4.5   5.0

  A single split at Rating = 3.8 perfectly separates the two groups.
  The tree's job is to FIND this boundary automatically.
```

---

## 3. Measuring Impurity — Gini vs Entropy

**Why we need impurity measures:** The tree needs to decide "is this split good?" To answer that, it needs a number that says "how mixed is this group?" A pure group (all same class) should score 0. A maximally mixed group (50/50) should score the highest. Gini and Entropy are two different ways to compute this number — they measure the same concept (disorder) from different angles.

**Where the formulas come from:** Gini Impurity = 1 - Σpᵢ² comes from probability theory: if you randomly pick two items from a group, what's the chance they're different classes? When all items are the same class, pᵢ² = 1 and Gini = 0 (no chance of a mismatch). When it's 50/50, Gini = 0.5 (maximum chance of mismatch). Entropy = -Σpᵢ log₂(pᵢ) comes from information theory: how many bits of information do you need to describe which class a random item belongs to? A pure group needs 0 bits (you already know), a 50/50 group needs 1 bit (one yes/no question). In practice, both give nearly identical split decisions — sklearn uses Gini by default because it's slightly faster to compute (no logarithm).

The tree needs a way to measure how "mixed" a group is. Two methods exist: Gini Impurity and Entropy. Both equal 0 for pure nodes and reach their maximum for 50/50 splits. The diagram below shows how they compare across different class proportions.

```
  Impurity
  1.0 │          ╱╲  ← Entropy
      │         ╱  ╲
      │        ╱    ╲
  0.5 │───────╱──────╲──── ← Gini (max = 0.5)
      │      ╱   ╲    ╲
      │     ╱     ╲    ╲
      │    ╱       ╲    ╲
  0.0 │───╱─────────╲────╲───
      └───┬─────┬─────┬───→ P(success)
         0.0   0.5   1.0

  Both are 0 at the edges (pure) and max in the middle (50/50).
  Entropy peaks at 1.0, Gini peaks at 0.5.
  In practice, they give nearly identical split decisions.
```

### 3.1 Computing Gini and Entropy for Our Data

Before any split, our data is 4 success + 4 failure = 50/50. This is maximum impurity. The diagram shows the calculation flowing from class proportions to the final impurity score.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    DATA["8 stores<br/>4 success, 4 failure"] --> PROPS["p(1) = 0.5<br/>p(0) = 0.5"]
    PROPS --> GINI["Gini = 1 minus (0.5² + 0.5²)<br/>= 1 - 0.5<br/>= <b>0.5</b> (max!)"]
    PROPS --> ENTROPY["Entropy = −(0.5×log₂0.5 + 0.5×log₂0.5)<br/>= −(0.5×(−1) + 0.5×(−1))<br/>= <b>1.0</b> (max!)"]

    style DATA fill:#252840,stroke:#f5b731,color:#c8cfe0
    style GINI fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style ENTROPY fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
```

Both metrics confirm: our unsplit data is maximally impure. Any good split should reduce these numbers toward 0.

---

## 4. Finding the Best Split

The tree tries every possible split on every feature and picks the one with the highest "gain" (biggest impurity reduction). This diagram shows three candidate splits and their resulting Gini values. The algorithm compares them all and picks the winner.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    PARENT["📊 Parent Node<br/>4✅ and 4❌ = Gini 0.5"] --> S1 & S2 & S3

    subgraph S1["Split: Rating ≤ 3.8"]
        S1L["Left: 0✅ 4❌<br/>Gini = 0.0 ✨"] --- S1R["Right: 4✅ 0❌<br/>Gini = 0.0 ✨"]
    end

    subgraph S2["Split: Rating ≤ 3.35"]
        S2L["Left: 0✅ 3❌<br/>Gini = 0.0"] --- S2R["Right: 4✅ 1❌<br/>Gini = 0.32"]
    end

    subgraph S3["Split: Delivery ≤ 30"]
        S3L["Left: 4✅ 0❌<br/>Gini = 0.0 ✨"] --- S3R["Right: 0✅ 4❌<br/>Gini = 0.0 ✨"]
    end

    S1 --> G1["Weighted Gini = 0.0<br/><b>Gain = 0.5</b> 🏆"]
    S2 --> G2["Weighted Gini = 0.2<br/>Gain = 0.3"]
    S3 --> G3["Weighted Gini = 0.0<br/><b>Gain = 0.5</b> 🏆"]

    style PARENT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style S1 fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style S2 fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style S3 fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style G1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style G2 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style G3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

The 🏆 marks the winners. Both "Rating ≤ 3.8" and "Delivery ≤ 30" achieve perfect separation (Gain = 0.5). "Rating ≤ 3.35" is decent (Gain = 0.3) but leaves one failure store (S6) mixed in with the successes. The algorithm picks the first perfect split it finds.

---

## 5. Information Gain — The Selection Criterion

**Why Information Gain:** We need a way to compare splits. A split is good if it reduces impurity — but we need to be precise about how we measure that reduction. Information Gain = parent impurity minus the weighted average of children's impurity. The weighting by child size is critical — a split that puts 7 items left and 1 right should weight the left child more heavily than the right. Without this weighting, the tree could game the metric by creating tiny pure leaves with just one data point (a form of overfitting). The weighted average ensures the tree considers the overall quality of the split, not just whether it can isolate a single point.

Information Gain is simply: how much did impurity decrease after the split? The diagram shows the formula and the calculation for our best split. Parent impurity minus weighted child impurity = the gain.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    FORMULA["Information Gain = Impurity(parent) - Weighted Impurity(children)"]
    FORMULA --> EXAMPLE["For Rating ≤ 3.8:"]
    EXAMPLE --> CALC["Gain = 0.5 − (4/8×0.0 + 4/8×0.0)<br/>= 0.5 − 0.0<br/>= <b>0.5</b>"]
    CALC --> MEANING["Maximum possible gain!<br/>This split perfectly separates the classes."]

    style FORMULA fill:#252840,stroke:#f5b731,color:#c8cfe0
    style CALC fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style MEANING fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

The weighted average accounts for the size of each child. A split that puts 7 stores left and 1 right is weighted differently than a 4/4 split. This prevents the tree from creating tiny pure leaves with just one data point.

---

## 6. The Final Tree

Our data is perfectly separable with just one split. The resulting tree is a "stump" (depth 1). The diagram shows the complete tree with the data flowing through it — left branch catches all failures, right branch catches all successes.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    ROOT{"🌳 Rating ≤ 3.8?<br/><i>8 stores: 4✅ 4❌</i><br/>Gini = 0.5"} -->|"Yes (≤ 3.8)"| LEFT["🍂 Predict: 0 ❌<br/>S2, S4, S6, S8<br/><i>4 stores, all failures</i><br/>Gini = 0.0"]
    ROOT -->|"No (> 3.8)"| RIGHT["🍂 Predict: 1 ✅<br/>S1, S3, S5, S7<br/><i>4 stores, all successes</i><br/>Gini = 0.0"]

    LEFT --> ACC1["4/4 correct = 100%"]
    RIGHT --> ACC2["4/4 correct = 100%"]
    ACC1 --> TOTAL["Overall: 8/8 = <b>100% accuracy</b>"]
    ACC2 --> TOTAL

    style ROOT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style LEFT fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style RIGHT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style TOTAL fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

The root node (yellow) shows the split question and the data distribution. Each leaf (red/green) shows the prediction, which stores landed there, and the Gini score (0.0 = pure). Perfect accuracy on training data — but would this hold on new data?

---

## 7. Making Predictions — Walking the Tree

To predict a new store, start at the root and follow the branches. The diagram traces three different stores through the tree, showing how each one takes a different path to reach its prediction.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph PRED1["New Store: Rating=4.2, Delivery=28"]
        P1A{"Rating ≤ 3.8?"} -->|"4.2 > 3.8 → No"| P1B["→ Right leaf<br/><b>Predict: 1 ✅</b>"]
    end

    subgraph PRED2["New Store: Rating=3.3, Delivery=32"]
        P2A{"Rating ≤ 3.8?"} -->|"3.3 ≤ 3.8 → Yes"| P2B["→ Left leaf<br/><b>Predict: 0 ❌</b>"]
    end

    subgraph PRED3["⚠️ New Store: Rating 3.9, Del 55"]
        P3A{"Rating ≤ 3.8?"} -->|"3.9 > 3.8 → No"| P3B["→ Right leaf<br/><b>Predict: 1 ✅</b><br/><i>But 55min delivery?!</i>"]
    end

    PRED3 --> WARNING["⚠️ The tree only checks Rating.<br/>It misses that 55min delivery is terrible.<br/>A deeper tree or more features would help."]

    style PRED1 fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style PRED2 fill:#0e1117,stroke:#f45d6d,color:#e2e8f0
    style PRED3 fill:#0e1117,stroke:#f5b731,color:#e2e8f0
    style WARNING fill:#252840,stroke:#f5b731,color:#c8cfe0
```

The first two predictions make sense. The third one exposes a limitation: our stump only uses Rating, so a store with great rating but terrible delivery still gets predicted as successful. This is why deeper trees (or ensembles) are often needed.

---

## 8. Overfitting — The Danger of Deep Trees

This is the most important concept in decision trees. A deep tree memorizes the training data perfectly but fails on new data. The diagram shows the progression from underfitting (too simple) to the sweet spot to overfitting (too complex).

```
  Error
    │
    │╲                              ╱
    │ ╲  Training Error            ╱ Test Error
    │  ╲                          ╱
    │   ╲                        ╱
    │    ╲         ╭────╮       ╱
    │     ╲       ╱      ╲     ╱
    │      ╲     ╱        ╲   ╱
    │       ╲   ╱          ╲ ╱
    │        ╲ ╱            ╳
    │         ╳            ╱ ╲
    │        ╱ ╲          ╱   ╲
    │       ╱   ╲────────╱     ╲── Training (→ 0)
    │      ╱
    └──────┬──────┬──────┬──────┬──→ Tree Depth
          1      3      5      10

    ◀─ Underfitting ─▶◀─ Sweet Spot ─▶◀─ Overfitting ─▶
```

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    subgraph UNDER["Depth 1 (Stump)"]
        U["One question<br/>High bias<br/>Low variance<br/><i>Too simple</i>"]
    end
    subgraph SWEET["Depth 3 to 5"]
        S["Good balance<br/>Moderate bias<br/>Moderate variance<br/><i>Just right</i>"]
    end
    subgraph OVER["Depth 10 and above"]
        O["Memorizes data<br/>Zero bias<br/>High variance<br/><i>Too complex</i>"]
    end

    UNDER -->|"Add depth"| SWEET -->|"Add more depth"| OVER

    style UNDER fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style SWEET fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style OVER fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

The green "sweet spot" in the middle is what we aim for. The error curve above shows training error always decreasing (the tree fits training data better), but test error first decreases then increases (overfitting kicks in).

---

## 9. Preventing Overfitting — Pruning Controls

These are the hyperparameters you tune to keep a tree from going too deep. Each one limits complexity in a different way. The diagram shows them as a control panel with their effects.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    TREE["🌳 Decision Tree"] --> CONTROLS

    subgraph CONTROLS["🎛️ Overfitting Controls"]
        C1["max_depth = 3<br/><i>Stop growing after 3 levels</i>"]
        C2["min_samples_split = 5<br/><i>Need ≥5 samples to split a node</i>"]
        C3["min_samples_leaf = 2<br/><i>Each leaf must have ≥2 samples</i>"]
        C4["max_leaf_nodes = 8<br/><i>Limit total number of leaves</i>"]
        C5["ccp_alpha > 0<br/><i>Post-pruning: remove weak branches</i>"]
    end

    CONTROLS --> RESULT["Simpler tree → Less overfitting<br/>→ Better generalization"]

    style TREE fill:#252840,stroke:#f5b731,color:#c8cfe0
    style CONTROLS fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style RESULT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Each control attacks overfitting differently: max_depth limits vertical growth, min_samples prevents splits on tiny groups, max_leaf_nodes limits horizontal spread, and ccp_alpha prunes after the tree is built. In practice, max_depth and min_samples_leaf are the most commonly tuned.

---

## 10. Classification vs Regression Trees

Decision trees work for both classification (predict a category) and regression (predict a number). The split criterion and leaf prediction differ. This diagram shows the two variants side by side.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph CLASS["🏷️ Classification Tree"]
        CC["Split criterion: Gini or Entropy<br/>Leaf prediction: Majority class<br/>Example: Successful? → Yes/No"]
        CC --> CL["Leaf: 3✅ 1❌<br/>Predict: ✅ (75% confidence)"]
    end

    subgraph REG["📈 Regression Tree"]
        RC["Split criterion: Variance reduction<br/>Leaf prediction: Mean of values<br/>Example: Revenue? → $72K"]
        RC --> RL["Leaf: $85K, $70K, $62K<br/>Predict: $72.3K (mean)"]
    end

    style CLASS fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style REG fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
```

Green = classification (uses Gini/Entropy, predicts the most common class). Blue = regression (uses variance, predicts the average value). Same tree structure, different math inside.

---

## 11. Decision Tree Algorithm Flowchart

The complete algorithm in one flowchart. This is what happens inside `sklearn.tree.DecisionTreeClassifier.fit()`. Start at the top, and the recursive loop builds the tree one split at a time until a stopping condition is met.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
flowchart TD
    START(["START: All data in root node"]) --> CHECK{"Stopping condition met?<br/><i>max_depth, min_samples,<br/>pure node, etc.</i>"}
    CHECK -->|"Yes"| LEAF["Create LEAF node<br/>Predict: majority class<br/>(or mean for regression)"]
    CHECK -->|"No"| TRY["Try EVERY feature<br/>at EVERY threshold"]
    TRY --> COMPUTE["Compute Gini Gain<br/>for each candidate split"]
    COMPUTE --> BEST["Pick split with<br/>HIGHEST gain"]
    BEST --> SPLIT["Split data into<br/>Left child and Right child"]
    SPLIT --> RECURSE_L["Recurse on Left child"]
    SPLIT --> RECURSE_R["Recurse on Right child"]
    RECURSE_L --> CHECK
    RECURSE_R --> CHECK

    style START fill:#252840,stroke:#f5b731,color:#c8cfe0
    style LEAF fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style BEST fill:#252840,stroke:#f5b731,color:#c8cfe0
    style SPLIT fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
```

The key insight: this is a greedy recursive algorithm. At each node, it picks the locally best split without considering future splits. This greedy nature is why a single tree can be suboptimal — and why ensembles (Random Forest, XGBoost) improve on it.

---

## 12. Pros, Cons, and When to Use

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph PROS["✅ Strengths"]
        P1["Easy to visualize & explain"]
        P2["No feature scaling needed"]
        P3["Handles numerical and categorical"]
        P4["Captures non-linear patterns"]
        P5["Built-in feature importance"]
    end

    subgraph CONS["❌ Weaknesses"]
        C1["Prone to overfitting"]
        C2["Unstable (high variance)"]
        C3["Greedy — not globally optimal"]
        C4["Can't extrapolate beyond data range"]
    end

    subgraph USE["🎯 Best Used For"]
        U1["Interpretable models (medical, legal)"]
        U2["Baseline before complex models"]
        U3["Building blocks for ensembles"]
    end

    style PROS fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style CONS fill:#0e1117,stroke:#f45d6d,color:#e2e8f0
    style USE fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
```

Green = reasons to use trees, Red = reasons to be cautious, Blue = ideal use cases. The biggest weakness (instability/high variance) is exactly what ensemble methods fix — which is why trees are rarely used alone in production.

---

## 13. Formula Summary

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph FORMULAS["📐 Key Formulas"]
        F1["Gini = 1 - Σ pᵢ²"]
        F2["Entropy = -Σ pᵢ × log₂(pᵢ)"]
        F3["Info Gain = H(parent) - Σ(nⱼ/n) × H(childⱼ)"]
        F4["Regression: Minimize weighted variance"]
        F5["Leaf: Classification → majority class<br/>Regression → mean value"]
    end

    style FORMULAS fill:#0e1117,stroke:#f5b731,color:#e2e8f0
```

---

## 14. Interview Decision Tree 🎯

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    Q1{"How does a tree<br/>decide where to split?"} -->|Answer| A1["Tries every feature × threshold<br/>Picks highest Gini gain<br/>Greedy, one split at a time"]
    Q1 -->|Next Q| Q2{"Gini vs Entropy?"}
    Q2 -->|Answer| A2["Both measure impurity<br/>Gini: faster, sklearn default<br/>Nearly identical results"]
    Q2 -->|Next Q| Q3{"How to prevent<br/>overfitting?"}
    Q3 -->|Answer| A3["max_depth, min_samples_leaf<br/>pruning, or use ensembles<br/>(Random Forest, XGBoost)"]
    Q3 -->|Next Q| Q4{"Why are trees<br/>unstable?"}
    Q4 -->|Answer| A4["Small data change → different root split<br/>→ cascades through entire tree<br/>This high variance is why we ensemble"]

    style Q1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A4 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Yellow = interview questions, Green = your answers. Follow the chain for a complete decision tree interview prep.

---

> 💡 **How to view:** GitHub (native), VS Code (Mermaid extension), Obsidian (built-in), or [mermaid.live](https://mermaid.live)
