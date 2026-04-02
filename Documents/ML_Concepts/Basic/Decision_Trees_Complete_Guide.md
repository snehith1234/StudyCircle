# Decision Trees: Complete Guide with Example Data at Every Step

## The Problem

You have 8 pizza stores. You want to predict: **Will a store be successful (1) or not (0)?**

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

## STEP 1: What Is a Decision Tree?

A decision tree is a flowchart of yes/no questions that splits data into groups.

```
Think of it like 20 Questions:
  "Is the rating above 3.8?"
    → Yes: "Is delivery under 30 min?"
      → Yes: Successful ✅
      → No:  Not Successful ❌
    → No: Not Successful ❌
```

Unlike logistic regression (which draws a single line), a decision tree creates
**rectangular regions** by splitting on one feature at a time.

### Key terminology
```
Root Node     = The very first question (top of the tree)
Internal Node = A question/split in the middle
Leaf Node     = A final prediction (no more questions)
Branch         = The path from one node to another
Depth          = How many levels of questions
```

---

## STEP 2: How Does the Tree Decide Where to Split?

The tree tries EVERY possible split on EVERY feature and picks the one that
creates the "purest" groups. But what does "pure" mean?

A **pure** node contains only one class:
```
Pure:     [1, 1, 1, 1]     → all successful
Pure:     [0, 0, 0, 0]     → all failures
Impure:   [1, 1, 0, 0]     → mixed
```

We need a way to MEASURE impurity. Two main methods:

---

## STEP 3: Gini Impurity

### The Formula

```
Gini(node) = 1 - Σ pᵢ²

Where pᵢ = proportion of class i in the node
```

### Intuition

Gini measures: "If I randomly pick two items from this node, what's the
probability they're DIFFERENT classes?"

```
Gini = 0   → perfectly pure (all same class)
Gini = 0.5 → maximum impurity (50/50 split for binary)
```

### Computing Gini for our full dataset (before any split)

```
Total: 8 stores → 4 successful (1), 4 not successful (0)

p(success) = 4/8 = 0.5
p(failure) = 4/8 = 0.5

Gini = 1 - (0.5² + 0.5²)
     = 1 - (0.25 + 0.25)
     = 1 - 0.5
     = 0.5  ← Maximum impurity! Makes sense — it's a 50/50 split.
```

---

## STEP 4: Entropy

### The Formula

```
Entropy(node) = -Σ pᵢ × log₂(pᵢ)
```

### Intuition

Entropy measures "surprise" or "uncertainty." If a node is pure, there's
no surprise (entropy = 0). If it's 50/50, maximum surprise (entropy = 1).

### Computing Entropy for our full dataset

```
p(success) = 0.5, p(failure) = 0.5

Entropy = -[0.5 × log₂(0.5) + 0.5 × log₂(0.5)]
        = -[0.5 × (-1) + 0.5 × (-1)]
        = -[-0.5 + (-0.5)]
        = -(-1.0)
        = 1.0  ← Maximum entropy for binary classification
```

### Useful log₂ values to remember
```
log₂(1.0) = 0       log₂(0.5)  = -1
log₂(0.75) ≈ -0.415  log₂(0.25) = -2
log₂(0.875) ≈ -0.193 log₂(0.125) = -3
```

---

## STEP 5: Finding the Best Split — Trying Every Possibility

The tree algorithm tries splitting on every feature at every possible threshold.

### Candidate splits for Rating

We sort the unique ratings and try midpoints:
```
Sorted ratings: 2.9, 3.0, 3.2, 3.5, 4.1, 4.3, 4.5, 4.8

Candidate thresholds (midpoints):
  2.95, 3.1, 3.35, 3.8, 4.2, 4.4, 4.65
```

### Let's try: Rating ≤ 3.8 vs Rating > 3.8

```
Left child (Rating ≤ 3.8):  S2(3.2,0), S4(2.9,0), S6(3.5,0), S8(3.0,0)
                             → Labels: [0, 0, 0, 0] → 4 stores, all failures

Right child (Rating > 3.8): S1(4.5,1), S3(4.8,1), S5(4.1,1), S7(4.3,1)
                             → Labels: [1, 1, 1, 1] → 4 stores, all successes
```

**Gini for left child:**
```
p(0) = 4/4 = 1.0, p(1) = 0/4 = 0.0
Gini_left = 1 - (1.0² + 0.0²) = 1 - 1.0 = 0.0  ← PURE!
```

**Gini for right child:**
```
p(1) = 4/4 = 1.0, p(0) = 0/4 = 0.0
Gini_right = 1 - (1.0² + 0.0²) = 1 - 1.0 = 0.0  ← PURE!
```

**Weighted Gini after split:**
```
Gini_split = (n_left/n_total) × Gini_left + (n_right/n_total) × Gini_right
           = (4/8) × 0.0 + (4/8) × 0.0
           = 0.0  ← PERFECT SPLIT!
```

**Gini Gain (Information Gain using Gini):**
```
Gain = Gini_parent - Gini_split
     = 0.5 - 0.0
     = 0.5  ← Maximum possible gain!
```

This is a perfect split — Rating ≤ 3.8 separates the classes completely.

### Let's also try: Rating ≤ 3.35

```
Left (Rating ≤ 3.35):  S2(3.2,0), S4(2.9,0), S8(3.0,0) → [0,0,0]
Right (Rating > 3.35): S1(4.5,1), S3(4.8,1), S5(4.1,1), S6(3.5,0), S7(4.3,1) → [1,1,1,0,1]

Gini_left = 1 - (1.0² + 0.0²) = 0.0
Gini_right = 1 - ((4/5)² + (1/5)²) = 1 - (0.64 + 0.04) = 0.32

Gini_split = (3/8)(0.0) + (5/8)(0.32) = 0 + 0.2 = 0.2
Gain = 0.5 - 0.2 = 0.3  ← Good but not as good as 3.8
```

### Let's try a Delivery split: Delivery ≤ 30

```
Left (Delivery ≤ 30):  S1(20,1), S3(18,1), S5(25,1), S7(22,1) → [1,1,1,1]
Right (Delivery > 30): S2(45,0), S4(50,0), S6(35,0), S8(40,0) → [0,0,0,0]

Gini_left = 0.0, Gini_right = 0.0
Gini_split = 0.0
Gain = 0.5 - 0.0 = 0.5  ← Also a perfect split!
```

Both Rating ≤ 3.8 and Delivery ≤ 30 give perfect splits. The algorithm picks
the first one it finds (or breaks ties arbitrarily). Let's use Rating ≤ 3.8.

---

## STEP 6: Information Gain (Entropy Version)

Information Gain using entropy works the same way:

```
Information Gain = Entropy_parent - Weighted_Entropy_children
```

### For the split Rating ≤ 3.8:

```
Entropy_parent = 1.0 (computed in Step 4)

Entropy_left = -[1.0 × log₂(1.0)] = -[1.0 × 0] = 0.0
Entropy_right = -[1.0 × log₂(1.0)] = 0.0

Weighted Entropy = (4/8)(0.0) + (4/8)(0.0) = 0.0

Information Gain = 1.0 - 0.0 = 1.0  ← Maximum possible!
```

### For the split Rating ≤ 3.35:

```
Entropy_left = 0.0 (pure)
Entropy_right = -[(4/5)log₂(4/5) + (1/5)log₂(1/5)]
              = -[0.8 × (-0.322) + 0.2 × (-2.322)]
              = -[-0.258 + (-0.464)]
              = -(-0.722)
              = 0.722

Weighted Entropy = (3/8)(0.0) + (5/8)(0.722) = 0 + 0.451 = 0.451

Information Gain = 1.0 - 0.451 = 0.549
```

Same conclusion: Rating ≤ 3.8 is the best split.

---

## STEP 7: Building the Full Tree

### Our dataset is perfectly separable with one split:

```
                    [All 8 stores]
                    4 success, 4 fail
                    Gini = 0.5
                         |
                  Rating ≤ 3.8?
                   /          \
                 YES            NO
                  |              |
          [S2,S4,S6,S8]    [S1,S3,S5,S7]
          0 success, 4 fail  4 success, 0 fail
          Gini = 0.0         Gini = 0.0
               |                   |
          Predict: 0          Predict: 1
          (Not Successful)    (Successful)
```

This is a **stump** — a tree with depth 1 (just one split).

### What if the data wasn't perfectly separable?

Let's say S5 (Rating=4.1) was actually a failure (y=0). Then:

```
Right child (Rating > 3.8): S1(1), S3(1), S5(0), S7(1) → [1,1,0,1]

Gini_right = 1 - ((3/4)² + (1/4)²) = 1 - (0.5625 + 0.0625) = 0.375

The tree would need a SECOND split on the right child:
  → Try Delivery ≤ 23: Left=[S1(20,1),S3(18,1),S7(22,1)] Right=[S5(25,0)]
  → Both pure! Gini = 0.0
```

This creates a deeper tree (depth 2).

---

## STEP 8: Making Predictions

To predict a new store, walk down the tree:

### New store: Rating=4.2, Delivery=28

```
Step 1: Is Rating ≤ 3.8?  → 4.2 > 3.8 → Go RIGHT
Step 2: Reach leaf node → Predict: 1 (Successful)
```

### New store: Rating=3.3, Delivery=32

```
Step 1: Is Rating ≤ 3.8?  → 3.3 ≤ 3.8 → Go LEFT
Step 2: Reach leaf node → Predict: 0 (Not Successful)
```

### New store: Rating=3.9, Delivery=55

```
Step 1: Is Rating ≤ 3.8?  → 3.9 > 3.8 → Go RIGHT
Step 2: Reach leaf node → Predict: 1 (Successful)

Hmm — 55 min delivery but predicted successful? This is a limitation of our
simple tree. A deeper tree with more splits could catch this edge case.
```

---

## STEP 9: Overfitting — The Danger of Deep Trees

### What happens if we let the tree grow without limits?

```
Depth 1: "Is Rating > 3.8?" → Simple, generalizable
Depth 5: "Is Rating > 4.37 AND Delivery < 21.5 AND Rating < 4.62?" → Memorizing!

A tree that perfectly fits training data will FAIL on new data.
```

### Example of overfitting:

```
Training data: 100% accuracy (memorized every store)
New data:      60% accuracy (can't generalize)

It's like a student who memorizes exam answers instead of understanding concepts.
They ace the practice test but fail the real exam.
```

### How to prevent overfitting:

```
1. Max Depth:        Limit how deep the tree can grow (e.g., max_depth=3)
2. Min Samples Split: Require at least N samples to make a split (e.g., min_samples_split=5)
3. Min Samples Leaf:  Require at least N samples in each leaf (e.g., min_samples_leaf=2)
4. Max Leaf Nodes:    Limit total number of leaves
5. Pruning:           Grow full tree, then remove branches that don't help
```

---

## STEP 10: Gini vs Entropy — When to Use Which?

```
| Aspect          | Gini                    | Entropy                    |
|-----------------|-------------------------|----------------------------|
| Range           | 0 to 0.5 (binary)      | 0 to 1.0 (binary)         |
| Computation     | Faster (no logarithm)  | Slower (needs log)         |
| Behavior        | Prefers larger partitions| Prefers balanced partitions|
| Default in sklearn | Yes (default)       | No (need criterion='entropy')|
| In practice     | Almost identical results| Almost identical results   |
```

**Rule of thumb:** Just use Gini (the default). The difference is negligible
in most real-world cases.

---

## STEP 11: Decision Trees for Regression

Decision trees can also predict continuous values (not just classes).

Instead of Gini/Entropy, regression trees use **variance reduction**:

```
Variance(node) = (1/n) × Σ (yᵢ - ȳ)²

The best split minimizes the weighted variance of the children.
```

### Example: Predicting monthly revenue

| Store | Rating | Revenue ($K) |
|-------|--------|-------------|
| S1    | 4.5    | 85          |
| S2    | 3.2    | 40          |
| S3    | 4.8    | 92          |
| S4    | 2.9    | 35          |
| S5    | 4.1    | 70          |
| S6    | 3.5    | 48          |

**Split: Rating ≤ 3.5**
```
Left:  [40, 35, 48] → mean = 41.0
  Variance = ((40-41)² + (35-41)² + (48-41)²) / 3 = (1+36+49)/3 = 28.67

Right: [85, 92, 70] → mean = 82.33
  Variance = ((85-82.33)² + (92-82.33)² + (70-82.33)²) / 3 = (7.13+93.51+152.15)/3 = 84.26

Weighted Variance = (3/6)(28.67) + (3/6)(84.26) = 14.33 + 42.13 = 56.47
```

**Parent variance:**
```
All: [85, 40, 92, 35, 70, 48] → mean = 61.67
Variance = ((85-61.67)² + ... ) / 6 = 544.22 / 6 = 544.22

Variance Reduction = 544.22 - 56.47 = 487.75
```

The leaf prediction is the **mean** of the values in that leaf.

---

## STEP 12: Pros, Cons, and When to Use

### Pros ✅
```
- Easy to understand and visualize (show it to non-technical stakeholders)
- No feature scaling needed (doesn't care about magnitude)
- Handles both numerical and categorical features
- Captures non-linear relationships
- Feature importance is built-in
- Fast prediction (just follow the path)
```

### Cons ❌
```
- Prone to overfitting (especially deep trees)
- Unstable — small data changes can create completely different trees
- Biased toward features with more levels
- Can't extrapolate (predictions bounded by training data range)
- Greedy algorithm — doesn't guarantee globally optimal tree
```

### When to use
```
- When interpretability is critical (medical, legal, finance)
- As a baseline model before trying complex methods
- When you need to explain decisions to non-technical people
- As building blocks for ensemble methods (Random Forest, XGBoost)
```

---

## COMPLETE FORMULA SUMMARY

```
1. Gini Impurity:      Gini = 1 - Σ pᵢ²
2. Entropy:            H = -Σ pᵢ × log₂(pᵢ)
3. Information Gain:   IG = H(parent) - Σ (nⱼ/n) × H(childⱼ)
4. Gini Gain:          GG = Gini(parent) - Σ (nⱼ/n) × Gini(childⱼ)
5. Regression split:   Minimize weighted variance of children
6. Leaf prediction:    Classification → majority class; Regression → mean value
```

---

## INTERVIEW CHEAT SHEET

**Q: "How does a decision tree decide where to split?"**
> "It tries every feature at every threshold, computes the impurity reduction (Gini gain or information gain) for each, and picks the split with the highest gain. This is a greedy approach — it optimizes one split at a time."

**Q: "Gini vs Entropy?"**
> "Both measure impurity. Gini is faster (no log computation) and is sklearn's default. Entropy tends to produce slightly more balanced trees. In practice, they give nearly identical results."

**Q: "How do you prevent overfitting in decision trees?"**
> "Limit tree depth, set minimum samples per split/leaf, limit max leaf nodes, or use post-pruning (cost-complexity pruning). Better yet, use ensemble methods like Random Forest or XGBoost."

**Q: "Why are decision trees unstable?"**
> "Small changes in data can change the root split, which cascades through the entire tree. This high variance is why we use ensembles — averaging many unstable trees gives a stable prediction."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
