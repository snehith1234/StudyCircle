# Bagging & Boosting: Complete Guide with Example Data at Every Step

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

## STEP 1: Why Ensembles? The Wisdom of Crowds

### The Problem with a Single Decision Tree

One decision tree is like asking ONE food critic to rate your restaurant:
- They might be biased (high variance)
- They might focus on the wrong things
- Small changes in data → completely different tree

### The Ensemble Idea

Instead of one critic, ask 100 critics and take the majority vote.

```
Single tree:   "Rating > 3.8 → Successful"  (might be wrong)

Ensemble:      Tree 1: "Rating > 3.8 → Successful"
               Tree 2: "Delivery < 30 → Successful"
               Tree 3: "Rating > 4.0 → Successful"
               ...
               Final: Majority vote of all trees
```

**Two main strategies:**
```
Bagging  = Train trees INDEPENDENTLY on different data samples → VOTE
Boosting = Train trees SEQUENTIALLY, each fixing the previous one's mistakes
```

---

## PART A: BAGGING (Bootstrap Aggregating)

---

## STEP 2: What Is Bootstrap Sampling?

Bootstrap = sampling WITH REPLACEMENT from the original data.

```
Original data: [S1, S2, S3, S4, S5, S6, S7, S8]  (8 stores)

Bootstrap sample 1: [S1, S3, S3, S5, S6, S6, S7, S8]  (8 stores, some repeated)
Bootstrap sample 2: [S2, S2, S4, S5, S5, S6, S7, S8]  (8 stores, some repeated)
Bootstrap sample 3: [S1, S1, S3, S4, S5, S7, S7, S8]  (8 stores, some repeated)
```

**Key insight:** Each bootstrap sample:
- Has the same SIZE as the original (8 stores)
- Some stores appear MULTIPLE times
- Some stores are LEFT OUT (~37% on average)

### Why ~37% are left out?

```
Probability a specific store is NOT picked in one draw = 7/8
Probability it's NOT picked in 8 draws = (7/8)^8 = 0.344 ≈ 34.4%

So about 1/3 of data is left out of each sample.
These are called "Out-of-Bag" (OOB) samples — free validation data!
```

---

## STEP 3: Bagging Step-by-Step

### Step 3a: Create 3 bootstrap samples

**Sample 1:** [S1, S3, S3, S5, S6, S7, S7, S8]
```
| Store | Rating | Delivery | Successful |
|-------|--------|----------|------------|
| S1    | 4.5    | 20       | 1          |
| S3    | 4.8    | 18       | 1          |
| S3    | 4.8    | 18       | 1          |
| S5    | 4.1    | 25       | 1          |
| S6    | 3.5    | 35       | 0          |
| S7    | 4.3    | 22       | 1          |
| S7    | 4.3    | 22       | 1          |
| S8    | 3.0    | 40       | 0          |

Missing: S2, S4 (these are OOB for this tree)
```

**Sample 2:** [S1, S2, S2, S4, S5, S6, S8, S8]
```
| Store | Rating | Delivery | Successful |
|-------|--------|----------|------------|
| S1    | 4.5    | 20       | 1          |
| S2    | 3.2    | 45       | 0          |
| S2    | 3.2    | 45       | 0          |
| S4    | 2.9    | 50       | 0          |
| S5    | 4.1    | 25       | 1          |
| S6    | 3.5    | 35       | 0          |
| S8    | 3.0    | 40       | 0          |
| S8    | 3.0    | 40       | 0          |

Missing: S3, S7 (OOB)
```

**Sample 3:** [S1, S1, S3, S4, S5, S6, S7, S8]
```
Missing: S2 (OOB)
```

### Step 3b: Train a decision tree on each sample

**Tree 1 (from Sample 1):**
```
Best split: Rating ≤ 3.25
  Left:  [S8(3.0,0)] → Predict 0
  Right: [S1,S3,S3,S5,S6,S7,S7] → 5 success, 1 fail
    Split: Delivery ≤ 30
      Left:  [S1,S3,S3,S5,S7,S7] → Predict 1
      Right: [S6] → Predict 0
```

**Tree 2 (from Sample 2):**
```
Best split: Rating ≤ 3.8
  Left:  [S2,S2,S4,S6,S8,S8] → all 0 → Predict 0
  Right: [S1,S5] → all 1 → Predict 1
```

**Tree 3 (from Sample 3):**
```
Best split: Delivery ≤ 30
  Left:  [S1,S1,S3,S5,S7] → all 1 → Predict 1
  Right: [S4,S6,S8] → all 0 → Predict 0
```

### Step 3c: Aggregate predictions (majority vote)

**Predict for a new store: Rating=3.6, Delivery=32**

```
Tree 1: Rating=3.6 > 3.25 → Right → Delivery=32 > 30 → Predict 0
Tree 2: Rating=3.6 ≤ 3.8 → Left → Predict 0
Tree 3: Delivery=32 > 30 → Right → Predict 0

Vote: 0, 0, 0 → Final prediction: 0 (Not Successful)
Confidence: 3/3 = 100%
```

**Predict for: Rating=4.2, Delivery=28**

```
Tree 1: Rating=4.2 > 3.25 → Delivery=28 ≤ 30 → Predict 1
Tree 2: Rating=4.2 > 3.8 → Predict 1
Tree 3: Delivery=28 ≤ 30 → Predict 1

Vote: 1, 1, 1 → Final prediction: 1 (Successful)
```

**Predict for: Rating=3.9, Delivery=33**

```
Tree 1: Rating=3.9 > 3.25 → Delivery=33 > 30 → Predict 0
Tree 2: Rating=3.9 > 3.8 → Predict 1
Tree 3: Delivery=33 > 30 → Predict 0

Vote: 0, 1, 0 → Final prediction: 0 (Not Successful)
Confidence: 2/3 = 67%
```

---

## STEP 4: Why Bagging Works — Variance Reduction

### The Math Behind It

```
Single tree error = Bias² + Variance + Noise

Bagging doesn't change bias (each tree is trained the same way).
Bagging REDUCES variance by averaging.

If you have B trees, each with variance σ²:
  - If trees are independent: Var(average) = σ²/B
  - If trees are correlated (ρ): Var(average) = ρσ² + (1-ρ)σ²/B

As B → ∞: Variance → ρσ² (limited by correlation between trees)
```

**Key insight:** Bagging works best when trees are UNCORRELATED.
This is why Random Forest adds random feature selection — to decorrelate trees.

### Bagging vs Random Forest

```
Bagging:        Each tree sees ALL features at each split
Random Forest:  Each tree sees a RANDOM SUBSET of features at each split

With 2 features (Rating, Delivery):
  Bagging:        Can split on Rating OR Delivery at every node
  Random Forest:  Might only see Rating at one node, only Delivery at another
                  (max_features = sqrt(p) for classification)
```

---

## PART B: BOOSTING

---

## STEP 5: The Boosting Idea

Bagging trains trees independently (parallel).
Boosting trains trees sequentially — each new tree focuses on what the previous ones got WRONG.

```
Bagging:   Tree1 ──→ ┐
           Tree2 ──→ ├──→ Vote
           Tree3 ──→ ┘

Boosting:  Tree1 ──→ errors ──→ Tree2 ──→ errors ──→ Tree3 ──→ ... ──→ Weighted Sum
```

**Analogy:**
```
Bagging  = 3 independent tutors teach you, you take the average advice
Boosting = Tutor 1 teaches you, Tutor 2 focuses on what you still don't get,
           Tutor 3 focuses on what you STILL don't get after Tutor 2
```

---

## STEP 6: AdaBoost (Adaptive Boosting) — Step by Step

### Step 6a: Initialize sample weights

Every store starts with equal weight:

```
w_i = 1/N = 1/8 = 0.125 for each store

| Store | Rating | Delivery | y | Weight |
|-------|--------|----------|---|--------|
| S1    | 4.5    | 20       | 1 | 0.125  |
| S2    | 3.2    | 45       | 0 | 0.125  |
| S3    | 4.8    | 18       | 1 | 0.125  |
| S4    | 2.9    | 50       | 0 | 0.125  |
| S5    | 4.1    | 25       | 1 | 0.125  |
| S6    | 3.5    | 35       | 0 | 0.125  |
| S7    | 4.3    | 22       | 1 | 0.125  |
| S8    | 3.0    | 40       | 0 | 0.125  |
Sum = 1.0
```

### Step 6b: Train Stump 1 (weak learner)

A "stump" is a tree with just ONE split (depth=1).

Best split: Rating ≤ 3.8 (perfectly separates with equal weights)

```
Stump 1: Rating ≤ 3.8 → Predict 0, else → Predict 1

Predictions:
S1(4.5→1) ✅  S2(3.2→0) ✅  S3(4.8→1) ✅  S4(2.9→0) ✅
S5(4.1→1) ✅  S6(3.5→0) ✅  S7(4.3→1) ✅  S8(3.0→0) ✅

All correct! Weighted error = 0
```

Since this perfectly classifies, let's make it more interesting. Suppose our
data had S5 with Rating=3.7 (borderline case):

| Store | Rating | Delivery | y |
|-------|--------|----------|---|
| S5    | 3.7    | 25       | 1 |

Now with split Rating ≤ 3.8:
```
S5(3.7→0) but actual=1 → ❌ WRONG!

Weighted error ε₁ = Σ wᵢ × I(wrong) = 0.125 × 1 = 0.125
(only S5 is wrong, weight = 0.125)
```

### Step 6c: Compute stump weight (α)

```
α₁ = 0.5 × ln((1 - ε₁) / ε₁)
   = 0.5 × ln((1 - 0.125) / 0.125)
   = 0.5 × ln(0.875 / 0.125)
   = 0.5 × ln(7.0)
   = 0.5 × 1.946
   = 0.973

Higher α = more accurate stump = more influence in final vote
```

### Step 6d: Update sample weights

```
For correctly classified samples:
  w_new = w_old × e^(-α) = 0.125 × e^(-0.973) = 0.125 × 0.378 = 0.0473

For misclassified samples (S5):
  w_new = w_old × e^(+α) = 0.125 × e^(0.973) = 0.125 × 2.646 = 0.331

Before normalization:
  7 correct stores: 7 × 0.0473 = 0.331
  1 wrong store (S5): 0.331
  Total = 0.331 + 0.331 = 0.662

Normalize (divide by total so weights sum to 1):
  Correct stores: 0.0473 / 0.662 = 0.0714 each
  S5 (wrong):     0.331 / 0.662 = 0.500

| Store | Old Weight | New Weight | Change    |
|-------|-----------|------------|-----------|
| S1    | 0.125     | 0.0714     | ↓ decreased |
| S2    | 0.125     | 0.0714     | ↓ decreased |
| S3    | 0.125     | 0.0714     | ↓ decreased |
| S4    | 0.125     | 0.0714     | ↓ decreased |
| S5    | 0.125     | 0.500      | ↑ INCREASED (was wrong!) |
| S6    | 0.125     | 0.0714     | ↓ decreased |
| S7    | 0.125     | 0.0714     | ↓ decreased |
| S8    | 0.125     | 0.0714     | ↓ decreased |
```

**S5 now has 50% of the total weight!** The next stump MUST get S5 right.

### Step 6e: Train Stump 2 (with updated weights)

Now the algorithm tries to find a split that correctly classifies S5.

Best split considering weights: Delivery ≤ 30

```
Stump 2: Delivery ≤ 30 → Predict 1, else → Predict 0

S5(Delivery=25 ≤ 30 → 1) ✅ Gets S5 right!

But S6(Delivery=35 > 30 → 0) ✅
All stores classified correctly with this split too!

Weighted error ε₂ = 0 (or very small)
```

### Step 6f: Final prediction (combine stumps)

```
Final(x) = sign(α₁ × Stump1(x) + α₂ × Stump2(x) + ...)

For a new store (Rating=3.7, Delivery=22):
  Stump 1: Rating=3.7 ≤ 3.8 → Predict -1 (encoding: 0→-1, 1→+1)
  Stump 2: Delivery=22 ≤ 30 → Predict +1

  Score = 0.973 × (-1) + α₂ × (+1)
  If α₂ > 0.973: Final = +1 (Successful)
```

---

## STEP 7: Gradient Boosting — The General Framework

AdaBoost adjusts sample weights. Gradient Boosting takes a more general approach:
each new tree fits the **residuals** (errors) of the previous model.

### Step 7a: Start with a simple prediction

```
Initial prediction: F₀(x) = mean(y) = 4/8 = 0.5 for all stores
(For classification, we'd use log-odds, but let's keep it simple with regression)
```

### Step 7b: Compute residuals

```
Residual = Actual - Predicted

| Store | Actual (y) | Predicted F₀ | Residual |
|-------|-----------|-------------|----------|
| S1    | 1         | 0.5         | +0.5     |
| S2    | 0         | 0.5         | -0.5     |
| S3    | 1         | 0.5         | +0.5     |
| S4    | 0         | 0.5         | -0.5     |
| S5    | 1         | 0.5         | +0.5     |
| S6    | 0         | 0.5         | -0.5     |
| S7    | 1         | 0.5         | +0.5     |
| S8    | 0         | 0.5         | -0.5     |
```

### Step 7c: Train Tree 1 on residuals

```
Tree 1 learns to predict the residuals:
  Rating ≤ 3.8 → predict -0.5 (these stores had negative residuals)
  Rating > 3.8 → predict +0.5 (these stores had positive residuals)
```

### Step 7d: Update predictions with learning rate

```
Learning rate η = 0.1 (small steps to avoid overfitting)

F₁(x) = F₀(x) + η × Tree1(x)

For S1 (Rating=4.5):
  F₁ = 0.5 + 0.1 × (+0.5) = 0.5 + 0.05 = 0.55

For S2 (Rating=3.2):
  F₁ = 0.5 + 0.1 × (-0.5) = 0.5 - 0.05 = 0.45
```

### Step 7e: Compute new residuals and repeat

```
| Store | Actual | F₁    | New Residual |
|-------|--------|-------|-------------|
| S1    | 1      | 0.55  | +0.45       |
| S2    | 0      | 0.45  | -0.45       |
| S3    | 1      | 0.55  | +0.45       |
| S4    | 0      | 0.45  | -0.45       |
| ...   | ...    | ...   | ...         |

Residuals are smaller! Each iteration reduces the error a bit more.
After 100 trees: F₁₀₀ ≈ actual values
```

---

## STEP 8: Bagging vs Boosting — Head to Head

```
| Aspect              | Bagging                    | Boosting                     |
|---------------------|----------------------------|------------------------------|
| Training            | Parallel (independent)     | Sequential (dependent)       |
| Focus               | Reduce VARIANCE            | Reduce BIAS                  |
| Base learners       | Full trees (complex)       | Stumps/shallow trees (weak)  |
| Sample weights      | Equal (bootstrap)          | Adjusted (focus on errors)   |
| Tree weights        | Equal vote                 | Weighted by accuracy         |
| Overfitting risk    | Low                        | Higher (can overfit noise)   |
| Sensitivity to noise| Robust                     | Sensitive                    |
| Speed               | Faster (parallelizable)    | Slower (sequential)          |
| Typical performance | Good                       | Often better (if tuned well) |
```

### When to use which?

```
Bagging (Random Forest):
  - Noisy data with outliers
  - When you want a "set it and forget it" model
  - When interpretability matters (feature importance)
  - When you have limited time for hyperparameter tuning

Boosting (XGBoost, LightGBM):
  - Clean data, well-defined problem
  - When you need maximum predictive performance
  - Kaggle competitions (boosting wins most tabular data competitions)
  - When you have time to tune hyperparameters
```

---

## STEP 9: Out-of-Bag (OOB) Error — Free Validation

Since each bootstrap sample leaves out ~37% of data, we can use those
left-out samples for validation WITHOUT a separate test set.

```
Tree 1 trained on: [S1,S3,S3,S5,S6,S7,S7,S8]
  OOB samples: S2, S4
  Predict S2 with Tree 1: 0 ✅
  Predict S4 with Tree 1: 0 ✅

Tree 2 trained on: [S1,S2,S2,S4,S5,S6,S8,S8]
  OOB samples: S3, S7
  Predict S3 with Tree 2: 1 ✅
  Predict S7 with Tree 2: 1 ✅

OOB Error = average error across all OOB predictions
         = 0/4 = 0% (perfect on this small example)
```

This is essentially free cross-validation — no need to set aside a test set.

---

## COMPLETE FORMULA SUMMARY

```
BAGGING:
1. Bootstrap sample:    Draw n samples with replacement from n data points
2. Train B trees:       Each on a different bootstrap sample
3. Aggregate:           Classification → majority vote; Regression → average
4. OOB error:           Test each tree on its ~37% left-out samples

ADABOOST:
1. Initialize weights:  wᵢ = 1/N
2. Weighted error:      εₜ = Σ wᵢ × I(ŷᵢ ≠ yᵢ)
3. Stump weight:        αₜ = 0.5 × ln((1-εₜ)/εₜ)
4. Update weights:      wᵢ = wᵢ × exp(-αₜ × yᵢ × ŷᵢ), then normalize
5. Final prediction:    F(x) = sign(Σ αₜ × hₜ(x))

GRADIENT BOOSTING:
1. Initialize:          F₀(x) = mean(y) or log(odds)
2. Compute residuals:   rᵢ = yᵢ - Fₘ₋₁(xᵢ)
3. Fit tree to residuals: hₘ(x) ≈ residuals
4. Update:              Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
5. Repeat M times
```

---

## INTERVIEW CHEAT SHEET

**Q: "What's the difference between bagging and boosting?"**
> "Bagging trains models independently on bootstrap samples and averages them — it reduces variance. Boosting trains models sequentially, each correcting the previous one's errors — it reduces bias. Bagging is more robust to noise; boosting often achieves higher accuracy but can overfit."

**Q: "Why does bagging reduce variance?"**
> "Averaging multiple noisy estimates reduces noise. If each tree has variance σ², averaging B uncorrelated trees gives variance σ²/B. Bootstrap sampling creates different training sets, making trees somewhat uncorrelated."

**Q: "How does AdaBoost handle misclassified samples?"**
> "It increases the weight of misclassified samples so the next weak learner focuses on them. The weight update is exponential: correctly classified samples get downweighted by e^(-α), misclassified ones get upweighted by e^(+α), where α reflects the learner's accuracy."

**Q: "What's the learning rate in gradient boosting?"**
> "It's a shrinkage parameter (0 < η ≤ 1) that scales each tree's contribution. Smaller η means each tree contributes less, requiring more trees but usually better generalization. It's a regularization technique — trading computation for accuracy."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
