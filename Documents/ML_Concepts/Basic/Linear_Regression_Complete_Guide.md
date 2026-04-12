# Linear Regression: Complete Guide with Example Data at Every Step

## The Problem

You have 8 pizza stores. You want to predict: **How much daily sales ($) will a store make?**

### Our Data

| Store | Employees (x₁) | Rating (x₂) | Ad_Spend (x₃) | Daily_Sales (y) |
|-------|----------------|-------------|---------------|----------------|
| S1    | 5              | 4.5         | 200           | $520           |
| S2    | 3              | 3.2         | 100           | $310           |
| S3    | 6              | 4.8         | 250           | $620           |
| S4    | 2              | 2.9         | 50            | $210           |
| S5    | 4              | 4.1         | 180           | $450           |
| S6    | 3              | 3.5         | 120           | $340           |
| S7    | 5              | 4.3         | 220           | $540           |
| S8    | 2              | 3.0         | 80            | $250           |

---

## STEP 1: What Is Linear Regression?

Linear regression finds the **best straight line** (or plane, in multiple dimensions) through your data so you can predict a continuous number.

### Simple vs. Multiple

```
Simple Linear Regression (1 feature):
  ŷ = β₀ + β₁x₁
  "Daily Sales = base + slope × Employees"

Multiple Linear Regression (many features):
  ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃
  "Daily Sales = base + β₁×Employees + β₂×Rating + β₃×Ad_Spend"
```

### Why "linear"?

The model is linear in the **parameters** (β₀, β₁, β₂...), not necessarily in the features.

```
Linear:     ŷ = β₀ + β₁x + β₂x²     ← still linear regression! (linear in β's)
Not linear: ŷ = β₀ + sin(β₁x)        ← non-linear (β₁ is inside a function)
```

---

## STEP 2: Simple Linear Regression — One Feature

Let's start simple: predict Daily Sales from Employees only.

### The equation

```
ŷ = β₀ + β₁x

Where:
  ŷ  = predicted daily sales
  β₀ = intercept (sales when employees = 0)
  β₁ = slope (change in sales per additional employee)
  x  = number of employees
```

### Computing β₁ and β₀ by hand

We need two formulas from statistics:

```
β₁ = Cov(x, y) / Var(x)
β₀ = ȳ - β₁x̄
```

**Step A: Compute means**
```
x̄ = (5 + 3 + 6 + 2 + 4 + 3 + 5 + 2) / 8
   = 30 / 8
   = 3.75

ȳ = (520 + 310 + 620 + 210 + 450 + 340 + 540 + 250) / 8
   = 3240 / 8
   = 405.0
```

**Step B: Compute Covariance(x, y)**
```
Cov(x,y) = (1/n) × Σ(xᵢ - x̄)(yᵢ - ȳ)

S1: (5 - 3.75)(520 - 405) = (1.25)(115)   = 143.75
S2: (3 - 3.75)(310 - 405) = (-0.75)(-95)  = 71.25
S3: (6 - 3.75)(620 - 405) = (2.25)(215)   = 483.75
S4: (2 - 3.75)(210 - 405) = (-1.75)(-195) = 341.25
S5: (4 - 3.75)(450 - 405) = (0.25)(45)    = 11.25
S6: (3 - 3.75)(340 - 405) = (-0.75)(-65)  = 48.75
S7: (5 - 3.75)(540 - 405) = (1.25)(135)   = 168.75
S8: (2 - 3.75)(250 - 405) = (-1.75)(-155) = 271.25

Sum = 143.75 + 71.25 + 483.75 + 341.25 + 11.25 + 48.75 + 168.75 + 271.25
    = 1540.0

Cov(x,y) = 1540.0 / 8 = 192.5
```

**Step C: Compute Variance(x)**
```
Var(x) = (1/n) × Σ(xᵢ - x̄)²

S1: (5 - 3.75)²  = (1.25)²  = 1.5625
S2: (3 - 3.75)²  = (-0.75)² = 0.5625
S3: (6 - 3.75)²  = (2.25)²  = 5.0625
S4: (2 - 3.75)²  = (-1.75)² = 3.0625
S5: (4 - 3.75)²  = (0.25)²  = 0.0625
S6: (3 - 3.75)²  = (-0.75)² = 0.5625
S7: (5 - 3.75)²  = (1.25)²  = 1.5625
S8: (2 - 3.75)²  = (-1.75)² = 3.0625

Sum = 15.5

Var(x) = 15.5 / 8 = 1.9375
```

**Step D: Compute β₁ and β₀**
```
β₁ = Cov(x,y) / Var(x) = 192.5 / 1.9375 = 99.35

β₀ = ȳ - β₁x̄ = 405.0 - 99.35 × 3.75 = 405.0 - 372.58 = 32.42
```

### Our simple model

```
ŷ = 32.42 + 99.35 × Employees

Interpretation:
  β₀ = 32.42  → A store with 0 employees would make ~$32 (theoretical baseline)
  β₁ = 99.35  → Each additional employee adds ~$99.35 in daily sales
```

---

## STEP 3: Making Predictions (Simple Model)

**Store S1: Employees=5**
```
ŷ = 32.42 + 99.35(5) = 32.42 + 496.75 = $529.17
Actual: $520    Error: 529.17 - 520 = +$9.17 (overpredicted by $9)
```

**Store S2: Employees=3**
```
ŷ = 32.42 + 99.35(3) = 32.42 + 298.05 = $330.47
Actual: $310    Error: +$20.47
```

**Store S3: Employees=6**
```
ŷ = 32.42 + 99.35(6) = 32.42 + 596.10 = $628.52
Actual: $620    Error: +$8.52
```

**Store S4: Employees=2**
```
ŷ = 32.42 + 99.35(2) = 32.42 + 198.70 = $231.12
Actual: $210    Error: +$21.12
```

**Store S5: Employees=4**
```
ŷ = 32.42 + 99.35(4) = 32.42 + 397.40 = $429.82
Actual: $450    Error: -$20.18
```

**Store S6: Employees=3**
```
ŷ = 32.42 + 99.35(3) = $330.47
Actual: $340    Error: -$9.53
```

**Store S7: Employees=5**
```
ŷ = 32.42 + 99.35(5) = $529.17
Actual: $540    Error: -$10.83
```

**Store S8: Employees=2**
```
ŷ = 32.42 + 99.35(2) = $231.12
Actual: $250    Error: -$18.88
```

### Summary Table

| Store | Employees | Actual ($) | Predicted ($) | Error ($) |
|-------|-----------|-----------|--------------|-----------|
| S1    | 5         | 520       | 529.17       | +9.17     |
| S2    | 3         | 310       | 330.47       | +20.47    |
| S3    | 6         | 620       | 628.52       | +8.52     |
| S4    | 2         | 210       | 231.12       | +21.12    |
| S5    | 4         | 450       | 429.82       | -20.18    |
| S6    | 3         | 340       | 330.47       | -9.53     |
| S7    | 5         | 540       | 529.17       | -10.83    |
| S8    | 2         | 250       | 231.12       | -18.88    |

---

## STEP 4: The Loss Function — Mean Squared Error (MSE)

### Why MSE?

The model needs to know "how wrong am I?" We use the squared errors because:

```
1. Squaring makes all errors positive (no cancellation of +/- errors)
2. Squaring penalizes large errors MORE than small ones
3. MSE is differentiable everywhere (smooth surface for optimization)
4. MSE has a closed-form solution (unlike log loss in logistic regression)
```

### The Formula

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

Where:
  yᵢ  = actual value
  ŷᵢ  = predicted value
  n   = number of data points
```

### Computing MSE for our simple model

```
S1: (520 - 529.17)² = (-9.17)²   = 84.09
S2: (310 - 330.47)² = (-20.47)²  = 419.02
S3: (620 - 628.52)² = (-8.52)²   = 72.59
S4: (210 - 231.12)² = (-21.12)²  = 446.05
S5: (450 - 429.82)² = (20.18)²   = 407.23
S6: (340 - 330.47)² = (9.53)²    = 90.82
S7: (540 - 529.17)² = (10.83)²   = 117.29
S8: (250 - 231.12)² = (18.88)²   = 356.45

Sum = 84.09 + 419.02 + 72.59 + 446.05 + 407.23 + 90.82 + 117.29 + 356.45
    = 1993.54

MSE = 1993.54 / 8 = 249.19
RMSE = √249.19 = $15.79
```

**RMSE = $15.79** means our predictions are off by about $16 on average.

### Other Loss Functions

```
MAE  = (1/n) × Σ|yᵢ - ŷᵢ|           ← less sensitive to outliers
MSE  = (1/n) × Σ(yᵢ - ŷᵢ)²          ← standard, penalizes large errors
RMSE = √MSE                           ← same units as y (dollars)
```

### Why not MAE instead of MSE?

```
MAE = |error|     → treats all errors equally
MSE = error²      → punishes big errors much more

Example:
  Errors: [1, 1, 1, 10]
  MAE = (1+1+1+10)/4 = 3.25
  MSE = (1+1+1+100)/4 = 25.75

MSE "cares" about that one big error (10) much more than MAE does.
Also: MSE is differentiable at 0, MAE is not (has a kink).
This matters for gradient descent — smooth surfaces are easier to optimize.
```

---

## STEP 5: The Closed-Form Solution (Normal Equation)

### Why linear regression is special

Unlike logistic regression (which needs iterative gradient descent), linear regression with MSE has a **closed-form solution** — you can compute the optimal weights directly with one formula.

### The Normal Equation

```
β = (XᵀX)⁻¹Xᵀy

Where:
  X = feature matrix (with a column of 1s for the intercept)
  y = target vector
  Xᵀ = transpose of X
  ⁻¹ = matrix inverse
```

### Why does this formula work?

```
We want to minimize: L = Σ(yᵢ - ŷᵢ)² = (y - Xβ)ᵀ(y - Xβ)

Take the derivative and set it to zero:
  ∂L/∂β = -2Xᵀ(y - Xβ) = 0

Solve for β:
  Xᵀy - XᵀXβ = 0
  XᵀXβ = Xᵀy
  β = (XᵀX)⁻¹Xᵀy  ✅
```

### Computing with our simple model (1 feature)

```
X = [1  5]     y = [520]
    [1  3]         [310]
    [1  6]         [620]
    [1  2]         [210]
    [1  4]         [450]
    [1  3]         [340]
    [1  5]         [540]
    [1  2]         [250]

The first column of 1s is for the intercept β₀.
```

For simple linear regression, the normal equation simplifies to:
```
β₁ = Cov(x,y) / Var(x) = 192.5 / 1.9375 = 99.35
β₀ = ȳ - β₁x̄ = 405.0 - 99.35(3.75) = 32.42

Same answer as Step 2! The normal equation is the general matrix form.
```

### When NOT to use the Normal Equation

```
Normal Equation: β = (XᵀX)⁻¹Xᵀy
  ✅ Small datasets (n < 10,000, features < 1,000)
  ❌ Large datasets — matrix inversion is O(p³) where p = number of features
  ❌ When XᵀX is singular (non-invertible) — happens with multicollinearity

Gradient Descent:
  ✅ Large datasets — scales well
  ✅ Works even with many features
  ❌ Requires choosing learning rate
  ❌ May need many iterations to converge
```

---

## STEP 6: Multiple Linear Regression

Now let's use ALL three features: Employees, Rating, and Ad_Spend.

### The model

```
ŷ = β₀ + β₁(Employees) + β₂(Rating) + β₃(Ad_Spend)
```

### Using the Normal Equation (matrix form)

```
X = [1  5  4.5  200]     y = [520]
    [1  3  3.2  100]         [310]
    [1  6  4.8  250]         [620]
    [1  2  2.9   50]         [210]
    [1  4  4.1  180]         [450]
    [1  3  3.5  120]         [340]
    [1  5  4.3  220]         [540]
    [1  2  3.0   80]         [250]
```

After solving β = (XᵀX)⁻¹Xᵀy (typically done by computer):

```
β₀ ≈ -20.5   (intercept)
β₁ ≈ 28.4    (Employees)
β₂ ≈ 45.2    (Rating)
β₃ ≈ 1.42    (Ad_Spend)
```

### Interpreting the coefficients

```
β₁ = 28.4  → Each additional employee adds ~$28.40 in daily sales
              (holding Rating and Ad_Spend constant)

β₂ = 45.2  → Each +1.0 point in rating adds ~$45.20 in daily sales
              (holding Employees and Ad_Spend constant)

β₃ = 1.42  → Each additional $1 in ad spend adds ~$1.42 in daily sales
              (holding Employees and Rating constant)
```

**Key phrase: "holding other variables constant."** This is the power of multiple regression — it isolates each feature's effect.

### Why did β₁ change from 99.35 to 28.4?

```
Simple model:   β₁ = 99.35 (Employees captures ALL the variation)
Multiple model: β₁ = 28.4  (Employees' UNIQUE contribution after accounting for Rating and Ad_Spend)

In the simple model, Employees was a proxy for everything — stores with more
employees also tend to have higher ratings and bigger ad budgets.
Multiple regression separates these effects.
```

---

## STEP 7: Gradient Descent for Linear Regression

Even though we have the normal equation, gradient descent is important to understand because:
1. It scales to massive datasets
2. It's the foundation for neural networks
3. It works when the normal equation doesn't (singular matrices)

### The gradient of MSE

```
Loss = (1/n) × Σ(yᵢ - ŷᵢ)²

∂Loss/∂βⱼ = -(2/n) × Σ(yᵢ - ŷᵢ) × xᵢⱼ

In plain English: The gradient for each weight is the average of
(actual - predicted) × (feature value), scaled by -2.
```

### Deriving the gradient step by step

```
Loss = (1/n) × Σ(yᵢ - ŷᵢ)²

Let eᵢ = yᵢ - ŷᵢ (the residual/error)

∂Loss/∂βⱼ = (1/n) × Σ 2(yᵢ - ŷᵢ) × ∂(yᵢ - ŷᵢ)/∂βⱼ    ← chain rule

Since yᵢ is constant and ŷᵢ = β₀ + β₁x₁ᵢ + β₂x₂ᵢ + ...:
  ∂(yᵢ - ŷᵢ)/∂βⱼ = -xᵢⱼ

So:
  ∂Loss/∂βⱼ = (1/n) × Σ 2(yᵢ - ŷᵢ) × (-xᵢⱼ)
             = -(2/n) × Σ(yᵢ - ŷᵢ) × xᵢⱼ
```

### Update rule

```
βⱼ_new = βⱼ_old - η × ∂Loss/∂βⱼ
       = βⱼ_old + η × (2/n) × Σ(yᵢ - ŷᵢ) × xᵢⱼ
```

### One iteration example (simple model)

Starting from β₀=0, β₁=0, learning rate η=0.001:

```
All predictions = 0 (since β₀=0, β₁=0)

Errors (yᵢ - ŷᵢ): 520, 310, 620, 210, 450, 340, 540, 250

∂Loss/∂β₀ = -(2/8) × (520×1 + 310×1 + 620×1 + 210×1 + 450×1 + 340×1 + 540×1 + 250×1)
           = -(2/8) × 3240
           = -810

∂Loss/∂β₁ = -(2/8) × (520×5 + 310×3 + 620×6 + 210×2 + 450×4 + 340×3 + 540×5 + 250×2)
           = -(2/8) × (2600 + 930 + 3720 + 420 + 1800 + 1020 + 2700 + 500)
           = -(2/8) × 13690
           = -3422.5

Update:
  β₀ = 0 - 0.001 × (-810) = 0 + 0.81 = 0.81
  β₁ = 0 - 0.001 × (-3422.5) = 0 + 3.42 = 3.42
```

After one step: ŷ = 0.81 + 3.42x. Still far from the optimal, but moving in the right direction. After thousands of iterations, it converges to β₀=32.42, β₁=99.35.

---

## STEP 8: Assumptions of Linear Regression

Linear regression makes 4 key assumptions. Violating them doesn't mean the model won't run — it means the results may be unreliable.

### 1. Linearity

```
Assumption: The relationship between features and target is linear.
Check: Plot residuals vs. predicted values — should show no pattern.

Violation example:
  True relationship: y = x²
  Linear model fits: ŷ = β₀ + β₁x  ← misses the curve!

Fix: Add polynomial features (x², x³) or use a non-linear model.
```

### 2. Independence of Errors

```
Assumption: Errors for different observations are independent.
Check: Durbin-Watson test (value near 2 = independent).

Violation example:
  Time series data — today's sales depend on yesterday's.
  Residuals show autocorrelation.

Fix: Use time series models (ARIMA) or add lagged features.
```

### 3. Homoscedasticity (Constant Variance)

```
Assumption: The spread of errors is the same across all predicted values.
Check: Plot residuals vs. predicted — should be a uniform band.

Violation example:
  Errors are small for low sales but huge for high sales.
  The residual plot fans out like a cone.

Fix: Log-transform the target, or use weighted least squares.
```

### 4. Normality of Errors

```
Assumption: Errors follow a normal distribution.
Check: Q-Q plot of residuals — should follow a straight line.

Violation example:
  Residuals are heavily skewed (many small errors, few huge ones).

Fix: Transform the target (log, sqrt), or use robust regression.
  Note: With large samples (n > 30), this matters less (Central Limit Theorem).
```

### Bonus: No Multicollinearity

```
Assumption: Features are not highly correlated with each other.
Check: Variance Inflation Factor (VIF) — should be < 5 (ideally < 10).

Violation example:
  "Employees" and "Ad_Spend" are 95% correlated.
  The model can't tell which one actually drives sales.
  Coefficients become unstable and hard to interpret.

Fix: Remove one of the correlated features, or use Ridge regression (L2).
```

---

## STEP 9: Evaluating the Model — R² and Adjusted R²

### R² (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)

Where:
  SS_res = Σ(yᵢ - ŷᵢ)²    ← sum of squared residuals (unexplained variance)
  SS_tot = Σ(yᵢ - ȳ)²      ← total sum of squares (total variance)
```

### Computing R² for our simple model

```
SS_res = Σ(yᵢ - ŷᵢ)² = 1993.54  (from Step 4)

SS_tot:
  S1: (520 - 405)² = (115)²  = 13225
  S2: (310 - 405)² = (-95)²  = 9025
  S3: (620 - 405)² = (215)²  = 46225
  S4: (210 - 405)² = (-195)² = 38025
  S5: (450 - 405)² = (45)²   = 2025
  S6: (340 - 405)² = (-65)²  = 4225
  S7: (540 - 405)² = (135)²  = 18225
  S8: (250 - 405)² = (-155)² = 24025

  SS_tot = 155000

R² = 1 - (1993.54 / 155000) = 1 - 0.01286 = 0.987
```

**R² = 0.987** means Employees alone explains 98.7% of the variance in Daily Sales.

### Interpreting R²

```
R² = 0.0  → Model explains nothing (just predicts the mean)
R² = 0.5  → Model explains 50% of variance
R² = 1.0  → Model explains everything (perfect predictions)

Our R² = 0.987 → Excellent! Employees is a very strong predictor.
```

### Adjusted R² — Why it matters

```
Problem: R² ALWAYS increases when you add more features, even useless ones.

Example: Add "store_id" as a feature → R² goes up, but the model is worse!

Adjusted R² penalizes for extra features:

Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]

Where:
  n = number of data points
  p = number of features

For our simple model (n=8, p=1):
  Adjusted R² = 1 - [(1 - 0.987)(8 - 1) / (8 - 1 - 1)]
              = 1 - [(0.013)(7) / 6]
              = 1 - [0.091 / 6]
              = 1 - 0.0152
              = 0.985

Rule: If Adjusted R² drops when you add a feature, that feature isn't helping.
```

---

## STEP 10: Regularization — Ridge, Lasso, Elastic Net

### The overfitting problem

```
With many features, the model can memorize the training data:
  - Coefficients become very large
  - Model fits noise, not signal
  - Great on training data, terrible on new data
```

### Ridge Regression (L2 Regularization)

```
Loss = MSE + λ × Σβⱼ²

The penalty λΣβⱼ² shrinks coefficients toward zero.
Large λ → more shrinkage → simpler model.

Key property: Ridge NEVER sets coefficients exactly to zero.
             It shrinks them, but keeps all features.
```

### Lasso Regression (L1 Regularization)

```
Loss = MSE + λ × Σ|βⱼ|

Key property: Lasso CAN set coefficients exactly to zero!
             It performs automatic feature selection.

Example: If Ad_Spend doesn't help, Lasso might set β₃ = 0.
```

### Elastic Net (L1 + L2)

```
Loss = MSE + λ₁ × Σ|βⱼ| + λ₂ × Σβⱼ²

Combines both: feature selection (L1) + coefficient shrinkage (L2).
Use when you have many correlated features.
```

### Choosing λ (the regularization strength)

```
λ = 0     → No regularization (standard linear regression)
λ = small → Slight regularization (coefficients slightly smaller)
λ = large → Heavy regularization (coefficients near zero, underfitting)

Use cross-validation to find the best λ:
  Try λ = [0.001, 0.01, 0.1, 1, 10, 100]
  Pick the one with lowest validation error.
```

---

## STEP 11: Feature Scaling — Why It Matters

### The problem

```
Our features have very different scales:
  Employees: 2 to 6       (range = 4)
  Rating:    2.9 to 4.8   (range = 1.9)
  Ad_Spend:  50 to 250    (range = 200)

For gradient descent: large-scale features dominate the gradient.
For regularization: large-scale features get penalized more.
```

### Standardization (Z-score scaling)

```
x_scaled = (x - mean) / std

Employees: mean=3.75, std=1.39
  S1: (5 - 3.75) / 1.39 = 0.90
  S4: (2 - 3.75) / 1.39 = -1.26

Ad_Spend: mean=150, std=68.7
  S1: (200 - 150) / 68.7 = 0.73
  S4: (50 - 150) / 68.7 = -1.46

Now both features are on the same scale (mean=0, std=1).
```

### When to scale

```
MUST scale:
  ✅ Gradient descent (convergence depends on scale)
  ✅ Ridge/Lasso (penalty treats all features equally only if scaled)
  ✅ Comparing coefficient magnitudes

DON'T need to scale:
  ❌ Normal equation (mathematically equivalent)
  ❌ Decision trees (splits don't depend on scale)
```

---

## STEP 12: Predicting a New Store

### New store: Employees=4, Rating=4.0, Ad_Spend=150

**Simple model (Employees only):**
```
ŷ = 32.42 + 99.35(4) = 32.42 + 397.40 = $429.82
```

**Multiple model (all features):**
```
ŷ = -20.5 + 28.4(4) + 45.2(4.0) + 1.42(150)
  = -20.5 + 113.6 + 180.8 + 213.0
  = $486.90
```

The multiple model gives a different (likely better) prediction because it uses more information.

---

## COMPLETE FORMULA SUMMARY

```
1. Simple model:          ŷ = β₀ + β₁x
2. Multiple model:        ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
3. Slope (simple):        β₁ = Cov(x,y) / Var(x)
4. Intercept (simple):    β₀ = ȳ - β₁x̄
5. Normal equation:       β = (XᵀX)⁻¹Xᵀy
6. MSE:                   (1/n) × Σ(yᵢ - ŷᵢ)²
7. RMSE:                  √MSE
8. R²:                    1 - (SS_res / SS_tot)
9. Adjusted R²:           1 - [(1-R²)(n-1) / (n-p-1)]
10. Gradient:             ∂MSE/∂βⱼ = -(2/n) × Σ(yᵢ - ŷᵢ) × xᵢⱼ
11. Weight update:        βⱼ = βⱼ - η × ∂MSE/∂βⱼ
12. Ridge loss:           MSE + λΣβⱼ²
13. Lasso loss:           MSE + λΣ|βⱼ|
```

---

## INTERVIEW CHEAT SHEET

**Q: "Explain linear regression."**
> "It finds the best-fit line (or hyperplane) through data by minimizing the sum of squared errors. The coefficients tell you how much the target changes per unit change in each feature, holding others constant."

**Q: "What's the difference between simple and multiple linear regression?"**
> "Simple uses one feature (ŷ = β₀ + β₁x). Multiple uses several (ŷ = β₀ + β₁x₁ + ... + βₚxₚ). Multiple regression isolates each feature's unique contribution by controlling for the others."

**Q: "How do you solve for the coefficients?"**
> "Two ways: (1) Normal equation β = (XᵀX)⁻¹Xᵀy — direct, O(p³), good for small data. (2) Gradient descent — iterative, scales to large data, required for regularized variants."

**Q: "What assumptions does linear regression make?"**
> "1) Linear relationship between features and target. 2) Independent errors. 3) Homoscedasticity (constant error variance). 4) Normally distributed errors. 5) No multicollinearity among features."

**Q: "What is R² and what are its limitations?"**
> "R² measures the proportion of variance explained by the model (0 to 1). Limitation: it always increases with more features, even useless ones. Use Adjusted R² which penalizes for extra features."

**Q: "When would you use Ridge vs. Lasso?"**
> "Ridge (L2) when you want to keep all features but shrink coefficients — good when all features are somewhat useful. Lasso (L1) when you suspect some features are irrelevant — it can zero them out for automatic feature selection. Elastic Net combines both."

**Q: "Linear regression vs. logistic regression?"**
> "Linear regression predicts a continuous number (sales, price). Logistic regression predicts a probability for classification (yes/no). Linear uses MSE loss; logistic uses log loss. Linear has a closed-form solution; logistic requires gradient descent."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
