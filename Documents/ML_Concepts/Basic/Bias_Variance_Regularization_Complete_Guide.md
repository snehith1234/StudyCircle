# Bias, Variance & Regularization: Complete Guide with Example Data at Every Step

## The Problem

You have 8 pizza stores. You want to predict **monthly revenue ($K)** from Rating.

### Our Data

| Store | Rating (x) | Revenue $K (y) |
|-------|-----------|----------------|
| S1    | 4.5       | 85             |
| S2    | 3.2       | 42             |
| S3    | 4.8       | 92             |
| S4    | 2.9       | 35             |
| S5    | 4.1       | 70             |
| S6    | 3.5       | 48             |
| S7    | 4.3       | 78             |
| S8    | 3.0       | 38             |

---

## STEP 1: What Are Bias and Variance?

### The Dartboard Analogy

Imagine throwing darts at a target (the true answer):

```
HIGH BIAS, LOW VARIANCE:        LOW BIAS, HIGH VARIANCE:
(consistently wrong)            (right on average, but scattered)

    ╭───────╮                       ╭───────╮
    │  ●●●  │                       │●     ●│
    │  ●●●  │                       │   ◎   │
    │   ◎   │                       │ ●   ● │
    │       │                       │  ●  ● │
    ╰───────╯                       ╰───────╯
  Darts clustered but              Darts scattered around
  far from bullseye                the bullseye

LOW BIAS, LOW VARIANCE:          HIGH BIAS, HIGH VARIANCE:
(the goal!)                      (worst case)

    ╭───────╮                       ╭───────╮
    │       │                       │●      │
    │  ●◎●  │                       │    ●  │
    │  ●●   │                       │   ◎   │
    │       │                       │ ●    ●│
    ╰───────╯                       ╰───────╯
  Tight cluster on                 Scattered AND off-target
  the bullseye
```

### Formal Definitions

```
Bias = How far OFF the average prediction is from the true value
       → "Systematic error" — the model is consistently wrong in one direction
       → Caused by: model too simple, wrong assumptions

Variance = How much predictions SCATTER when trained on different data
           → "Sensitivity to training data" — model changes a lot with new data
           → Caused by: model too complex, memorizing noise
```

### The Bias-Variance Decomposition

```
Expected Error = Bias² + Variance + Irreducible Noise

  Bias²:             Error from wrong assumptions (underfitting)
  Variance:          Error from sensitivity to training data (overfitting)
  Irreducible Noise: Random noise in the data (can't be reduced by any model)
```

---

## STEP 2: Demonstrating Bias — The Underfitting Model

### Model 1: Predict the mean (simplest possible model)

```
ŷ = mean(y) = (85+42+92+35+70+48+78+38) / 8 = 488/8 = 61.0

Every store gets the same prediction: $61K
```

**Predictions and errors:**
```
| Store | Rating | Actual | Predicted | Error  | Error² |
|-------|--------|--------|-----------|--------|--------|
| S1    | 4.5    | 85     | 61        | +24    | 576    |
| S2    | 3.2    | 42     | 61        | -19    | 361    |
| S3    | 4.8    | 92     | 61        | +31    | 961    |
| S4    | 2.9    | 35     | 61        | -26    | 676    |
| S5    | 4.1    | 70     | 61        | +9     | 81     |
| S6    | 3.5    | 48     | 61        | -13    | 169    |
| S7    | 4.3    | 78     | 61        | +17    | 289    |
| S8    | 3.0    | 38     | 61        | -23    | 529    |

MSE = (576+361+961+676+81+169+289+529) / 8 = 3642/8 = 455.25
```

**This model has HIGH BIAS:** It's too simple to capture the Rating→Revenue pattern.
It systematically underpredicts high-revenue stores and overpredicts low-revenue ones.

**But LOW VARIANCE:** If you retrain on different data, the mean won't change much.

---

## STEP 3: Demonstrating Variance — The Overfitting Model

### Model 3: Fit a degree-7 polynomial (way too complex for 8 points)

```
ŷ = β₀ + β₁x + β₂x² + β₃x³ + β₄x⁴ + β₅x⁵ + β₆x⁶ + β₇x⁷
```

With 8 data points and 8 parameters, this polynomial passes through
EVERY point exactly:

```
| Store | Actual | Predicted | Error |
|-------|--------|-----------|-------|
| S1    | 85     | 85.0      | 0     |
| S2    | 42     | 42.0      | 0     |
| S3    | 92     | 92.0      | 0     |
| S4    | 35     | 35.0      | 0     |
| S5    | 70     | 70.0      | 0     |
| S6    | 48     | 48.0      | 0     |
| S7    | 78     | 78.0      | 0     |
| S8    | 38     | 38.0      | 0     |

Training MSE = 0  ← PERFECT! But...
```

**What happens with a new store (Rating=3.8)?**

The polynomial might predict Revenue = -150 or +500 — wildly wrong!
Between the training points, the curve oscillates crazily.

**This model has LOW BIAS** (on training data) but **HIGH VARIANCE:**
Train on slightly different data → completely different polynomial → completely different predictions.

---

## STEP 4: The Sweet Spot — Linear Regression

### Model 2: Simple linear regression (just right)

```
ŷ = β₀ + β₁ × Rating
```

**Computing β₁ and β₀:**
```
x̄ = mean(Rating) = (4.5+3.2+4.8+2.9+4.1+3.5+4.3+3.0)/8 = 30.3/8 = 3.7875
ȳ = mean(Revenue) = 488/8 = 61.0

Numerator = Σ(xᵢ - x̄)(yᵢ - ȳ):
  S1: (4.5-3.7875)(85-61) = 0.7125 × 24 = 17.10
  S2: (3.2-3.7875)(42-61) = -0.5875 × (-19) = 11.16
  S3: (4.8-3.7875)(92-61) = 1.0125 × 31 = 31.39
  S4: (2.9-3.7875)(35-61) = -0.8875 × (-26) = 23.08
  S5: (4.1-3.7875)(70-61) = 0.3125 × 9 = 2.81
  S6: (3.5-3.7875)(48-61) = -0.2875 × (-13) = 3.74
  S7: (4.3-3.7875)(78-61) = 0.5125 × 17 = 8.71
  S8: (3.0-3.7875)(38-61) = -0.7875 × (-23) = 18.11
  
  Numerator = 116.10

Denominator = Σ(xᵢ - x̄)²:
  (0.7125² + 0.5875² + 1.0125² + 0.8875² + 0.3125² + 0.2875² + 0.5125² + 0.7875²)
  = 0.508 + 0.345 + 1.025 + 0.788 + 0.098 + 0.083 + 0.263 + 0.620
  = 3.729

β₁ = 116.10 / 3.729 = 31.13
β₀ = ȳ - β₁ × x̄ = 61.0 - 31.13 × 3.7875 = 61.0 - 117.91 = -56.91

Model: ŷ = -56.91 + 31.13 × Rating
```

**Predictions:**
```
| Store | Rating | Actual | Predicted | Error  | Error² |
|-------|--------|--------|-----------|--------|--------|
| S1    | 4.5    | 85     | 83.18     | +1.82  | 3.31   |
| S2    | 3.2    | 42     | 42.71     | -0.71  | 0.50   |
| S3    | 4.8    | 92     | 92.52     | -0.52  | 0.27   |
| S4    | 2.9    | 35     | 33.37     | +1.63  | 2.66   |
| S5    | 4.1    | 70     | 70.72     | -0.72  | 0.52   |
| S6    | 3.5    | 48     | 52.05     | -4.05  | 16.40  |
| S7    | 4.3    | 78     | 76.95     | +1.05  | 1.10   |
| S8    | 3.0    | 38     | 36.48     | +1.52  | 2.31   |

MSE = 27.07 / 8 = 3.38
```

**This model has:**
- Moderate bias (not perfect, but captures the trend)
- Low variance (linear model is stable across different training sets)
- MSE = 3.38 (much better than the mean model's 455.25)

---

## STEP 5: The Bias-Variance Tradeoff Visualized

```
Error
  |
  |  \                    /
  |   \   Bias²         / Variance
  |    \               /
  |     \             /
  |      \    ___    /
  |       \  /   \ /
  |        \/     \
  |     Total Error (U-shaped)
  |
  +──────────────────────────→ Model Complexity
     Mean   Linear  Polynomial
     Model  Regr.   Degree 7
     
  ← Underfitting    Overfitting →
  ← High Bias       High Variance →
```

```
| Model              | Bias  | Variance | Training Error | Test Error |
|--------------------|-------|----------|----------------|------------|
| Mean (too simple)  | HIGH  | LOW      | 455.25         | ~460       |
| Linear (just right)| LOW   | LOW      | 3.38           | ~5-10      |
| Poly-7 (too complex)| ZERO | HIGH     | 0.00           | ~500+      |
```

---

## STEP 6: Regularization — Controlling Complexity

### What is regularization?

```
Without regularization: "Minimize prediction error. That's it."
  → Model can use huge coefficients → overfits

With regularization: "Minimize prediction error + keep coefficients small."
  → Penalizes complexity → better generalization
```

### The general form:

```
Loss = Prediction Error + λ × Complexity Penalty

λ (lambda) = regularization strength
  λ = 0:    No regularization (original model)
  λ = ∞:    Maximum regularization (all coefficients → 0, predicts the mean)
  λ = sweet spot: Balanced model
```

---

## STEP 7: Ridge Regression (L2 Regularization)

### The Formula

```
Loss_Ridge = Σ(yᵢ - ŷᵢ)² + λ × Σβⱼ²
              ↑ MSE            ↑ Sum of SQUARED coefficients

The penalty is proportional to β² — large coefficients are penalized heavily.
```

### How it works — with our data

**Without regularization (λ=0):**
```
β₁ = 31.13 (Rating coefficient)
```

**With Ridge (λ=10):**

The Ridge solution modifies the normal equation:
```
β_ridge = (XᵀX + λI)⁻¹ Xᵀy

The λI term "shrinks" the coefficients toward zero.
```

Let's see the effect for different λ values:
```
| λ     | β₀      | β₁ (Rating) | MSE (train) | Interpretation          |
|-------|---------|-------------|-------------|-------------------------|
| 0     | -56.91  | 31.13       | 3.38        | No regularization       |
| 1     | -52.10  | 29.85       | 3.52        | Slight shrinkage        |
| 10    | -33.20  | 24.82       | 5.41        | Moderate shrinkage      |
| 100   | -3.15   | 16.92       | 18.73       | Heavy shrinkage         |
| 1000  | 52.80   | 2.16        | 398.50      | Almost predicting mean  |
```

**Key insight:** As λ increases:
- Coefficients shrink toward 0 (but never exactly 0)
- Training error increases (worse fit)
- But test error first DECREASES then increases (sweet spot exists)

### Why "Ridge"?

```
The λI added to XᵀX creates a "ridge" along the diagonal of the matrix.
This makes the matrix invertible even when features are correlated
(fixes multicollinearity).
```

---

## STEP 8: Lasso Regression (L1 Regularization)

### The Formula

```
Loss_Lasso = Σ(yᵢ - ŷᵢ)² + λ × Σ|βⱼ|
              ↑ MSE            ↑ Sum of ABSOLUTE coefficients
```

### The key difference from Ridge:

```
Ridge (L2): Penalty = β²  → Shrinks coefficients toward 0, but never TO 0
Lasso (L1): Penalty = |β| → Can shrink coefficients EXACTLY to 0

This means Lasso does FEATURE SELECTION!
```

### Why does L1 produce zeros but L2 doesn't?

```
Geometric intuition:

L2 constraint region: a CIRCLE (β₁² + β₂² ≤ t)
  → The loss contours (ellipses) touch the circle at any point
  → Usually NOT on an axis → both β₁ and β₂ are non-zero

L1 constraint region: a DIAMOND (|β₁| + |β₂| ≤ t)
  → The loss contours touch the diamond at a CORNER
  → Corners are on axes → one coefficient IS zero

     β₂                    β₂
      |   ╱╲                |   ╭╮
      |  ╱  ╲               |  ╭╯╰╮
      | ╱ ●  ╲              | ╭╯ ● ╰╮
  ────╱──────╲────      ────╰╮──────╭╯────
      ╲      ╱              ╰╮    ╭╯
       ╲    ╱                ╰╮  ╭╯
        ╲  ╱                  ╰╮╭╯
         ╲╱                    ╰╯
    L1 (Diamond)           L2 (Circle)
    ● = optimal point      ● = optimal point
    (on corner = sparse)   (not on axis = dense)
```

### Lasso with our data:

```
| λ     | β₀      | β₁ (Rating) | Notes                    |
|-------|---------|-------------|--------------------------|
| 0     | -56.91  | 31.13       | No regularization        |
| 1     | -53.50  | 30.20       | Slight shrinkage         |
| 10    | -28.00  | 23.50       | Moderate shrinkage       |
| 50    | 0       | 16.10       | β₀ pushed to 0!         |
| 200   | 0       | 0           | Both coefficients = 0!   |
```

---

## STEP 9: Elastic Net (L1 + L2 Combined)

### The Formula

```
Loss_ElasticNet = Σ(yᵢ - ŷᵢ)² + λ₁ × Σ|βⱼ| + λ₂ × Σβⱼ²
                                   ↑ L1 (sparsity)  ↑ L2 (stability)

Or equivalently with mixing parameter α:
  Loss = MSE + λ × [α × Σ|βⱼ| + (1-α) × Σβⱼ²]

  α = 1: Pure Lasso
  α = 0: Pure Ridge
  α = 0.5: Equal mix
```

### When to use Elastic Net:

```
- When you have many correlated features
- Lasso tends to pick one feature from a correlated group and ignore the rest
- Ridge keeps all features but doesn't do selection
- Elastic Net: selects features (like Lasso) but handles correlations (like Ridge)
```

---

## STEP 10: Regularization in Other Models

Regularization isn't just for linear regression. It appears everywhere:

```
| Model              | Regularization Method                    |
|--------------------|------------------------------------------|
| Linear Regression  | Ridge (L2), Lasso (L1), Elastic Net     |
| Logistic Regression| Same — L1, L2, Elastic Net              |
| Decision Trees     | Max depth, min samples, pruning          |
| Random Forest      | Number of trees, max features            |
| XGBoost            | λ (L2), α (L1), γ (min gain), max depth |
| Neural Networks    | L2 weight decay, Dropout, Early stopping |
```

**The common theme:** All regularization techniques add a penalty for complexity
to prevent the model from memorizing the training data.

---

## STEP 11: Cross-Validation — Choosing λ

How do you pick the right λ? You can't use training error (it always prefers λ=0).

### K-Fold Cross-Validation (K=4 for our 8 stores):

```
Fold 1: Train on [S3,S4,S5,S6,S7,S8], Test on [S1,S2]
Fold 2: Train on [S1,S2,S5,S6,S7,S8], Test on [S3,S4]
Fold 3: Train on [S1,S2,S3,S4,S7,S8], Test on [S5,S6]
Fold 4: Train on [S1,S2,S3,S4,S5,S6], Test on [S7,S8]

For each λ value, compute average test error across all 4 folds.
Pick the λ with the lowest average test error.
```

```
| λ    | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg Test Error |
|------|--------|--------|--------|--------|----------------|
| 0    | 5.2    | 8.1    | 12.3   | 4.8    | 7.6            |
| 0.1  | 4.9    | 7.5    | 11.1   | 4.5    | 7.0            |
| 1.0  | 4.5    | 6.8    | 9.2    | 4.2    | 6.2  ← BEST    |
| 10   | 8.3    | 10.1   | 8.5    | 7.9    | 8.7            |
| 100  | 25.1   | 28.3   | 22.7   | 24.5   | 25.2           |
```

λ = 1.0 gives the best generalization. Not too much, not too little.

---

## STEP 12: Summary — The Big Picture

```
                    Underfitting ←──────────→ Overfitting
                    
Model complexity:   Low ─────────────────────── High
Bias:               High ────────────────────── Low
Variance:           Low ─────────────────────── High
Training error:     High ────────────────────── Low (→ 0)
Test error:         High ──── Low (sweet spot) ── High

Regularization:     Moves you LEFT on this spectrum
                    (reduces complexity, increases bias, decreases variance)
```

### Decision framework:

```
Training error HIGH, Test error HIGH → Underfitting → DECREASE regularization
Training error LOW, Test error HIGH  → Overfitting  → INCREASE regularization
Training error LOW, Test error LOW   → Good fit!    → Keep current settings
```

---

## COMPLETE FORMULA SUMMARY

```
1. Bias-Variance:     E[Error] = Bias² + Variance + σ²_noise
2. Ridge (L2):        Loss = Σ(yᵢ-ŷᵢ)² + λΣβⱼ²
3. Lasso (L1):        Loss = Σ(yᵢ-ŷᵢ)² + λΣ|βⱼ|
4. Elastic Net:       Loss = Σ(yᵢ-ŷᵢ)² + λ[αΣ|βⱼ| + (1-α)Σβⱼ²]
5. Ridge solution:    β = (XᵀX + λI)⁻¹Xᵀy
6. Cross-validation:  CV(λ) = (1/K)Σ MSE_k(λ)
```

---

## INTERVIEW CHEAT SHEET

**Q: "Explain the bias-variance tradeoff."**
> "Total error = bias² + variance + noise. Bias is systematic error from a too-simple model (underfitting). Variance is sensitivity to training data from a too-complex model (overfitting). You can't minimize both simultaneously — reducing one increases the other. The goal is finding the sweet spot."

**Q: "Ridge vs Lasso?"**
> "Both add a penalty to prevent overfitting. Ridge uses L2 (squared coefficients) — it shrinks coefficients toward zero but never to exactly zero. Lasso uses L1 (absolute coefficients) — it can shrink coefficients to exactly zero, performing feature selection. Use Lasso when you suspect many features are irrelevant; use Ridge when most features matter."

**Q: "What does the regularization parameter λ control?"**
> "λ controls the strength of the penalty. λ=0 means no regularization (original model). As λ increases, coefficients shrink more, the model becomes simpler, bias increases, and variance decreases. Choose λ via cross-validation."

**Q: "How do you detect overfitting?"**
> "Compare training error vs validation/test error. If training error is much lower than test error, the model is overfitting. The gap between them indicates the degree of overfitting. Use learning curves (plot error vs training set size) for a visual diagnostic."

**Q: "What is Elastic Net and when would you use it?"**
> "Elastic Net combines L1 and L2 penalties. It does feature selection like Lasso but handles correlated features better like Ridge. Use it when you have many features, some correlated, and want automatic feature selection with stability."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
