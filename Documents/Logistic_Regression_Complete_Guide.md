# Logistic Regression: Complete Guide with Example Data at Every Step

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

## STEP 1: Why Not Linear Regression?

Linear regression predicts: ŷ = β₀ + β₁x₁ + β₂x₂

Problem: This can give values like -0.3 or 1.5 — but probability must be between 0 and 1!

```
Example: If linear regression learned β₀=-2, β₁=0.8, β₂=-0.01
  Store S4 (Rating=2.9, Delivery=50):
  ŷ = -2 + 0.8(2.9) + (-0.01)(50)
  ŷ = -2 + 2.32 - 0.50
  ŷ = -0.18  ← NEGATIVE! Can't be a probability!
```

**Solution:** We need a function that squashes any number into [0, 1].

---

## STEP 2: The Sigmoid Function

### The Formula

```
σ(z) = 1 / (1 + e^(-z))
```

### What it does

Takes ANY number z (from -∞ to +∞) and outputs a number between 0 and 1.

### Computing sigmoid for specific values

```
z = -3:  σ(-3) = 1 / (1 + e^3)  = 1 / (1 + 20.086) = 1 / 21.086 = 0.047
z = -2:  σ(-2) = 1 / (1 + e^2)  = 1 / (1 + 7.389)  = 1 / 8.389  = 0.119
z = -1:  σ(-1) = 1 / (1 + e^1)  = 1 / (1 + 2.718)  = 1 / 3.718  = 0.269
z =  0:  σ(0)  = 1 / (1 + e^0)  = 1 / (1 + 1)      = 1 / 2      = 0.500
z =  1:  σ(1)  = 1 / (1 + e^-1) = 1 / (1 + 0.368)  = 1 / 1.368  = 0.731
z =  2:  σ(2)  = 1 / (1 + e^-2) = 1 / (1 + 0.135)  = 1 / 1.135  = 0.881
z =  3:  σ(3)  = 1 / (1 + e^-3) = 1 / (1 + 0.050)  = 1 / 1.050  = 0.953
```

### Key properties
- z = 0 → exactly 0.5 (50/50)
- z > 0 → probability > 0.5 → predict YES
- z < 0 → probability < 0.5 → predict NO
- z very large → probability ≈ 1
- z very negative → probability ≈ 0

---

## STEP 3: The Logistic Regression Model

### Combine linear equation + sigmoid

```
Step 1: Calculate z (linear combination)
  z = β₀ + β₁x₁ + β₂x₂

Step 2: Apply sigmoid to get probability
  P(y=1) = σ(z) = 1 / (1 + e^(-z))

Step 3: Classify
  If P ≥ 0.5 → predict 1 (Successful)
  If P < 0.5 → predict 0 (Not Successful)
```

### The model needs to LEARN: β₀, β₁, β₂

These are the weights (parameters). We don't know them yet.
Let's say after training, the model learned:

```
β₀ = -4.0  (intercept/bias)
β₁ = +1.2  (weight for Rating)
β₂ = -0.04 (weight for Delivery_Min)
```

---

## STEP 4: Making Predictions (Forward Pass)

### Predict for EVERY store using learned weights

**Store S1: Rating=4.5, Delivery=20**
```
z = β₀ + β₁(Rating) + β₂(Delivery)
z = -4.0 + 1.2(4.5) + (-0.04)(20)
z = -4.0 + 5.4 - 0.8
z = 0.6

P = σ(0.6) = 1 / (1 + e^(-0.6))
P = 1 / (1 + 0.549)
P = 1 / 1.549
P = 0.646 = 64.6%

P = 0.646 ≥ 0.5 → Predict: 1 (Successful) ✅ Correct! (actual=1)
```

**Store S2: Rating=3.2, Delivery=45**
```
z = -4.0 + 1.2(3.2) + (-0.04)(45)
z = -4.0 + 3.84 - 1.8
z = -1.96

P = σ(-1.96) = 1 / (1 + e^(1.96))
P = 1 / (1 + 7.099)
P = 1 / 8.099
P = 0.123 = 12.3%

P = 0.123 < 0.5 → Predict: 0 (Not Successful) ✅ Correct! (actual=0)
```

**Store S3: Rating=4.8, Delivery=18**
```
z = -4.0 + 1.2(4.8) + (-0.04)(18)
z = -4.0 + 5.76 - 0.72
z = 1.04

P = σ(1.04) = 1 / (1 + e^(-1.04))
P = 1 / (1 + 0.354)
P = 1 / 1.354
P = 0.739 = 73.9%

Predict: 1 ✅ Correct!
```

**Store S4: Rating=2.9, Delivery=50**
```
z = -4.0 + 1.2(2.9) + (-0.04)(50)
z = -4.0 + 3.48 - 2.0
z = -2.52

P = σ(-2.52) = 1 / (1 + e^(2.52))
P = 1 / (1 + 12.429)
P = 1 / 13.429
P = 0.074 = 7.4%

Predict: 0 ✅ Correct!
```

**Store S5: Rating=4.1, Delivery=25**
```
z = -4.0 + 1.2(4.1) + (-0.04)(25)
z = -4.0 + 4.92 - 1.0
z = -0.08

P = σ(-0.08) = 1 / (1 + e^(0.08))
P = 1 / (1 + 1.083)
P = 1 / 2.083
P = 0.480 = 48.0%

P = 0.480 < 0.5 → Predict: 0 ❌ WRONG! (actual=1)
```

**Store S6: Rating=3.5, Delivery=35**
```
z = -4.0 + 1.2(3.5) + (-0.04)(35)
z = -4.0 + 4.2 - 1.4
z = -1.2

P = σ(-1.2) = 1 / (1 + e^(1.2))
P = 1 / (1 + 3.320)
P = 1 / 4.320
P = 0.231 = 23.1%

Predict: 0 ✅ Correct!
```

**Store S7: Rating=4.3, Delivery=22**
```
z = -4.0 + 1.2(4.3) + (-0.04)(22)
z = -4.0 + 5.16 - 0.88
z = 0.28

P = σ(0.28) = 1 / (1 + e^(-0.28))
P = 1 / (1 + 0.756)
P = 1 / 1.756
P = 0.570 = 57.0%

Predict: 1 ✅ Correct!
```

**Store S8: Rating=3.0, Delivery=40**
```
z = -4.0 + 1.2(3.0) + (-0.04)(40)
z = -4.0 + 3.6 - 1.6
z = -2.0

P = σ(-2.0) = 1 / (1 + e^(2.0))
P = 1 / (1 + 7.389)
P = 1 / 8.389
P = 0.119 = 11.9%

Predict: 0 ✅ Correct!
```

### Summary Table

| Store | Rating | Delivery | z     | P(success) | Predict | Actual | Correct? |
|-------|--------|----------|-------|-----------|---------|--------|----------|
| S1    | 4.5    | 20       | +0.60 | 64.6%     | 1       | 1      | ✅       |
| S2    | 3.2    | 45       | -1.96 | 12.3%     | 0       | 0      | ✅       |
| S3    | 4.8    | 18       | +1.04 | 73.9%     | 1       | 1      | ✅       |
| S4    | 2.9    | 50       | -2.52 | 7.4%      | 0       | 0      | ✅       |
| S5    | 4.1    | 25       | -0.08 | 48.0%     | 0       | 1      | ❌       |
| S6    | 3.5    | 35       | -1.20 | 23.1%     | 0       | 0      | ✅       |
| S7    | 4.3    | 22       | +0.28 | 57.0%     | 1       | 1      | ✅       |
| S8    | 3.0    | 40       | -2.00 | 11.9%     | 0       | 0      | ✅       |

**Accuracy: 7/8 = 87.5%** (only S5 is wrong — it's a borderline case at 48%)

---

## STEP 5: The Loss Function — How the Model Learns

### Why we need a loss function

The model starts with RANDOM weights (e.g., β₀=0, β₁=0, β₂=0).
It needs to figure out that β₀=-4, β₁=+1.2, β₂=-0.04 are good values.

The loss function tells the model: "How wrong are you?"

### Log Loss (Binary Cross-Entropy)

```
For one data point:
  Loss = -[y × log(p) + (1-y) × log(1-p)]

Where:
  y = actual label (0 or 1)
  p = predicted probability
  log = natural logarithm (ln)
```

### Where does this formula come from?

**Step A: Bernoulli Distribution**

Each store is a yes/no outcome. The probability of observing what we actually saw:

```
If y=1 (success): we want p to be high → probability of this observation = p
If y=0 (failure): we want p to be low → probability of this observation = (1-p)

Combined: L = p^y × (1-p)^(1-y)

Check:
  y=1: p^1 × (1-p)^0 = p × 1 = p ✅
  y=0: p^0 × (1-p)^1 = 1 × (1-p) = (1-p) ✅
```

**Step B: Likelihood of ALL data**

Multiply likelihoods for all 8 stores (assuming independence):

```
L = Π p_i^y_i × (1-p_i)^(1-y_i)    for i = 1 to 8
```

**Step C: Take the log (Log-Likelihood)**

Products of small numbers → numerical underflow. Log fixes this:

```
log(L) = Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]
```

**Step D: Negate (minimize loss instead of maximize likelihood)**

```
Loss = -log(L) = -Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]
```

### Computing Loss for Each Store

Using our predictions from Step 4:

**Store S1: y=1, p=0.646**
```
Loss = -[1 × log(0.646) + 0 × log(0.354)]
     = -[1 × (-0.437) + 0]
     = -(-0.437)
     = 0.437
```

**Store S2: y=0, p=0.123**
```
Loss = -[0 × log(0.123) + 1 × log(0.877)]
     = -[0 + 1 × (-0.131)]
     = -(-0.131)
     = 0.131
```

**Store S3: y=1, p=0.739**
```
Loss = -[1 × log(0.739)]
     = -(-0.302)
     = 0.302
```

**Store S4: y=0, p=0.074**
```
Loss = -[1 × log(0.926)]
     = -(-0.077)
     = 0.077
```

**Store S5: y=1, p=0.480** ← the wrong prediction
```
Loss = -[1 × log(0.480)]
     = -(-0.734)
     = 0.734  ← Highest loss! Model is most wrong here.
```

**Store S6: y=0, p=0.231**
```
Loss = -[1 × log(0.769)]
     = -(-0.263)
     = 0.263
```

**Store S7: y=1, p=0.570**
```
Loss = -[1 × log(0.570)]
     = -(-0.562)
     = 0.562
```

**Store S8: y=0, p=0.119**
```
Loss = -[1 × log(0.881)]
     = -(-0.127)
     = 0.127
```

### Total Loss

```
Total Loss = (1/8) × (0.437 + 0.131 + 0.302 + 0.077 + 0.734 + 0.263 + 0.562 + 0.127)
           = (1/8) × 2.633
           = 0.329
```

**Average Log Loss = 0.329** (lower is better, 0 = perfect)

---

## STEP 6: Gradient Descent — Updating the Weights

### The gradient (how much to adjust each weight)

```
∂Loss/∂βⱼ = (1/n) × Σ (pᵢ - yᵢ) × xᵢⱼ
```

This is just: **average of (prediction error × feature value)**

### Computing gradients with our data

**Errors (pᵢ - yᵢ) for each store:**
```
S1: 0.646 - 1 = -0.354  (underpredicted)
S2: 0.123 - 0 = +0.123  (slightly overpredicted)
S3: 0.739 - 1 = -0.261
S4: 0.074 - 0 = +0.074
S5: 0.480 - 1 = -0.520  (biggest error!)
S6: 0.231 - 0 = +0.231
S7: 0.570 - 1 = -0.430
S8: 0.119 - 0 = +0.119
```

**Gradient for β₀ (intercept):**
```
∂Loss/∂β₀ = (1/8) × Σ (pᵢ - yᵢ) × 1    (x₀ = 1 always)
           = (1/8) × (-0.354 + 0.123 - 0.261 + 0.074 - 0.520 + 0.231 - 0.430 + 0.119)
           = (1/8) × (-1.018)
           = -0.127
```

**Gradient for β₁ (Rating):**
```
∂Loss/∂β₁ = (1/8) × Σ (pᵢ - yᵢ) × Rating_i

= (1/8) × [(-0.354)(4.5) + (0.123)(3.2) + (-0.261)(4.8) + (0.074)(2.9)
          + (-0.520)(4.1) + (0.231)(3.5) + (-0.430)(4.3) + (0.119)(3.0)]

= (1/8) × [-1.593 + 0.394 - 1.253 + 0.215 - 2.132 + 0.809 - 1.849 + 0.357]

= (1/8) × (-5.052)
= -0.632
```

**Gradient for β₂ (Delivery_Min):**
```
∂Loss/∂β₂ = (1/8) × Σ (pᵢ - yᵢ) × Delivery_i

= (1/8) × [(-0.354)(20) + (0.123)(45) + (-0.261)(18) + (0.074)(50)
          + (-0.520)(25) + (0.231)(35) + (-0.430)(22) + (0.119)(40)]

= (1/8) × [-7.08 + 5.535 - 4.698 + 3.70 - 13.00 + 8.085 - 9.46 + 4.76]

= (1/8) × (-12.158)
= -1.520
```

### Update the weights

```
Learning rate η = 0.1

β₀_new = β₀ - η × ∂Loss/∂β₀ = -4.0 - 0.1 × (-0.127) = -4.0 + 0.013 = -3.987
β₁_new = β₁ - η × ∂Loss/∂β₁ = +1.2 - 0.1 × (-0.632) = +1.2 + 0.063 = +1.263
β₂_new = β₂ - η × ∂Loss/∂β₂ = -0.04 - 0.1 × (-1.520) = -0.04 + 0.152 = +0.112
```

Wait — β₂ went positive? That means the model thinks more delivery time = MORE successful.
That's wrong! But this is just ONE step. Over many iterations, the weights converge to correct values.

### After many iterations (e.g., 1000 steps)

The weights settle to values that minimize the total loss.
Each iteration: compute predictions → compute loss → compute gradients → update weights.

---

## STEP 7: Interpreting the Coefficients

### What the final weights mean

```
β₀ = -4.0  (intercept)
β₁ = +1.2  (Rating)
β₂ = -0.04 (Delivery_Min)
```

### Odds and Odds Ratio

**Odds = P / (1-P)**

```
If P = 0.75: Odds = 0.75/0.25 = 3.0 ("3 to 1 in favor")
If P = 0.50: Odds = 0.50/0.50 = 1.0 ("even odds")
If P = 0.25: Odds = 0.25/0.75 = 0.33 ("1 to 3 against")
```

**Log-Odds = log(Odds) = log(P/(1-P)) = z = β₀ + β₁x₁ + β₂x₂**

This is the key insight: logistic regression is LINEAR in log-odds!

**Odds Ratio = e^β**

```
β₁ = +1.2 (Rating):
  Odds Ratio = e^1.2 = 3.32
  "Each +1 point in rating multiplies the odds of success by 3.32×"

  Example: Store with Rating=3 has odds = 2.0
           Store with Rating=4 has odds = 2.0 × 3.32 = 6.64
           That's 3.32× more likely to succeed!

β₂ = -0.04 (Delivery):
  Odds Ratio = e^(-0.04) = 0.961
  "Each +1 minute delivery multiplies odds by 0.961 (reduces by 3.9%)"

  Example: Store with 20min delivery has odds = 5.0
           Store with 30min delivery has odds = 5.0 × 0.961^10 = 5.0 × 0.664 = 3.32
           10 extra minutes → odds drop by 33.6%!
```

---

## STEP 8: Decision Boundary

### Where does the model switch from 0 to 1?

At the decision boundary, P = 0.5, which means z = 0:

```
0 = β₀ + β₁(Rating) + β₂(Delivery)
0 = -4.0 + 1.2(Rating) - 0.04(Delivery)

Solving for Delivery:
0.04(Delivery) = -4.0 + 1.2(Rating)
Delivery = (-4.0 + 1.2 × Rating) / 0.04
Delivery = -100 + 30 × Rating
```

**Decision boundary examples:**
```
Rating = 3.0: Delivery = -100 + 30(3.0) = -100 + 90 = -10 min (impossible → always predict 0)
Rating = 3.5: Delivery = -100 + 30(3.5) = -100 + 105 = 5 min
Rating = 4.0: Delivery = -100 + 30(4.0) = -100 + 120 = 20 min
Rating = 4.5: Delivery = -100 + 30(4.5) = -100 + 135 = 35 min
Rating = 5.0: Delivery = -100 + 30(5.0) = -100 + 150 = 50 min
```

**Reading the boundary:**
- A store with Rating=4.0 is predicted successful if Delivery < 20 min
- A store with Rating=4.5 is predicted successful if Delivery < 35 min
- Higher rating → can tolerate longer delivery and still be successful

---

## STEP 9: Model Evaluation

### Confusion Matrix

```
                    Predicted
                    0       1
Actual  0          4       0       ← True Negatives, False Positives
        1          1       3       ← False Negatives, True Positives
```

### Metrics

```
Accuracy  = (TP + TN) / Total = (3 + 4) / 8 = 7/8 = 87.5%
Precision = TP / (TP + FP) = 3 / (3 + 0) = 3/3 = 100%
Recall    = TP / (TP + FN) = 3 / (3 + 1) = 3/4 = 75%
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
          = 2 × (1.0 × 0.75) / (1.0 + 0.75)
          = 2 × 0.75 / 1.75
          = 0.857 = 85.7%
```

---

## STEP 10: Predicting a New Store

### New store: Rating=4.0, Delivery=28 min

```
Step 1: Calculate z
  z = -4.0 + 1.2(4.0) + (-0.04)(28)
  z = -4.0 + 4.8 - 1.12
  z = -0.32

Step 2: Apply sigmoid
  P = 1 / (1 + e^(0.32))
  P = 1 / (1 + 1.377)
  P = 1 / 2.377
  P = 0.421 = 42.1%

Step 3: Classify
  P = 0.421 < 0.5 → Predict: 0 (Not Successful)

Interpretation: "This store has a 42% chance of being successful.
The rating is decent (4.0) but delivery is a bit slow (28 min).
If they can reduce delivery to under 20 min, they'd cross the threshold."
```

---

## COMPLETE FORMULA SUMMARY

```
1. Linear combination:     z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
2. Sigmoid:                P = 1 / (1 + e^(-z))
3. Classification:         ŷ = 1 if P ≥ 0.5, else 0
4. Loss (single point):    L = -[y log(p) + (1-y) log(1-p)]
5. Total loss:             J = -(1/n) Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
6. Gradient:               ∂J/∂βⱼ = (1/n) Σ (pᵢ - yᵢ) × xᵢⱼ
7. Weight update:          βⱼ = βⱼ - η × ∂J/∂βⱼ
8. Odds ratio:             OR = e^βⱼ
9. Decision boundary:      z = 0 → β₀ + β₁x₁ + β₂x₂ = 0
```

---

## INTERVIEW CHEAT SHEET

**Q: "Explain logistic regression."**
> "It predicts the probability of a binary outcome using the sigmoid function applied to a linear combination of features. The sigmoid squashes any value into [0,1]. We train it by minimizing log loss using gradient descent."

**Q: "Why log loss and not MSE?"**
> "MSE with sigmoid creates a non-convex loss surface with local minima. Log loss is derived from maximum likelihood estimation of the Bernoulli distribution, and it's convex — guaranteed to find the global minimum."

**Q: "How do you interpret coefficients?"**
> "Each coefficient β represents the change in log-odds for a one-unit increase in that feature. The odds ratio e^β tells you how much the odds multiply. For example, β=1.2 means each unit increase multiplies odds by e^1.2 = 3.32×."

**Q: "What assumptions does logistic regression make?"**
> "1) Linear relationship between features and log-odds. 2) Independence of observations. 3) No multicollinearity among features. 4) Large sample size relative to number of features."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
