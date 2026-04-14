# Linear Regression: Visual Guide

> Visual companion to `Documents/ML_Concepts/Basic/Linear_Regression_Complete_Guide.md`.

---

## 1. The Big Idea — Finding the Best Line

```mermaid
graph LR
    INPUT["📊 Features: Employees, Rating, Ad Spend"] --> LINEAR["ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃"] --> OUTPUT["💰 Daily Sales = $486.90"]
    style INPUT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style LINEAR fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style OUTPUT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Features (yellow) → weighted sum (blue) → continuous prediction (green).

---

## 2. Simple vs. Multiple Linear Regression

```mermaid
graph TD
    subgraph SIMPLE["Simple: 1 Feature"]
        S_IN["Employees"] --> S_EQ["ŷ = 32.42 + 99.35x, β₁ = 99.35"] --> S_R2["R² = 0.987"]
    end
    subgraph MULTIPLE["Multiple: 3 Features"]
        M_IN1["Employees"] --> M_EQ["ŷ = −20.5 + 28.4x₁ + 45.2x₂ + 1.42x₃"]
        M_IN2["Rating"] --> M_EQ
        M_IN3["Ad Spend"] --> M_EQ
        M_EQ --> M_R2["R² ≈ 0.998"]
    end
    S_R2 --> COMPARE["β₁ dropped: 99.35 → 28.4 — isolates UNIQUE effect"]
    M_R2 --> COMPARE
    style SIMPLE fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style MULTIPLE fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style COMPARE fill:#252840,stroke:#f5b731,color:#c8cfe0
```

---

## 3. The Best-Fit Line — Residuals

```
  Daily Sales ($)
  650 │                              ● S3 ($620)
      │                            ╱ ↕ residual
  550 │                     ● S7 ╱($540)
      │                   ● S1 ╱($520)
  450 │              ● S5╱($450)
  350 │         ● S6 ╱($340)
      │        ● S2╱($310)
  250 │     ● S8╱($250)
      │   ● S4╱($210)
  150 │   ╱  ŷ = 32.42 + 99.35x
      └──┬────┬────┬────┬────┬────┬──→ Employees
         1    2    3    4    5    6
```

---

## 4. Loss Functions Compared

```
  Loss
  100│ ●                              MSE = error²
  80 │   ╲
  60 │     ╲
  40 │       ╲
  20 │         ╲    ╱ MAE = |error|
   0 │───────────●───────────→ Error
    -10     -5     0     5     10
```

---

## 5. Normal Equation vs. Gradient Descent

```mermaid
graph TD
    DATA["Training Data: X, y"] --> NE_FORMULA
    DATA --> GD_INIT
    subgraph NE["Normal Equation"]
        NE_FORMULA["β = inv of XᵀX times Xᵀy"] --> NE_PRO["✅ Exact, no hyperparameters"]
        NE_FORMULA --> NE_CON["❌ Slow for large p, fails if singular"]
    end
    subgraph GD["Gradient Descent"]
        GD_INIT["Initialize β = 0"] --> GD_GRAD["Compute gradient"] --> GD_UPDATE["Update weights"] --> GD_CHECK{"Converged?"}
        GD_CHECK -->|No| GD_GRAD
        GD_CHECK -->|Yes| GD_PRO["✅ Scales to big data"]
        GD_CHECK -->|Yes| GD_CON["❌ Needs learning rate"]
    end
    NE_PRO --> SAME["Both find the SAME optimal β"]
    GD_PRO --> SAME
    style DATA fill:#252840,stroke:#f5b731,color:#c8cfe0
    style SAME fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

---

## 6. Gradient Descent — Walking Downhill

```
  MSE Loss
  800 │●
  600 │  ╲
  400 │    ╲
  200 │      ╲──────────────────
      │                         ●  converged
    0 └────┬────┬────┬────┬────┬──→ Iterations
           0   200  400  600  800
```

---

## 7. The Four Assumptions — Visual Diagnostics

```mermaid
graph TD
    A1["Linearity"] --> A1G["Good: Random scatter"]
    A1 --> A1B["Bad: Curved pattern"]
    A1B --> A2["Independence"]
    A2 --> A2G["Good: DW close to 2"]
    A2 --> A2B["Bad: DW near 0 or 4, errors are correlated"]
    A2B --> A3["Homoscedasticity"]
    A3 --> A3G["Good: Uniform band"]
    A3 --> A3B["Bad: Cone shape"]
    A3B --> A4["Normality of Errors"]
    A4 --> A4G["Good: Points on diagonal"]
    A4 --> A4B["Bad: S-curve or tails"]
```

Yellow = assumptions. Green = good. Red = violations and fixes.

---

## 8. R² — What It Means Visually

```
  R² = 1 − SS_res / SS_tot

  Our model: R² = 0.987
  ┌─────────────────────────────────────────────────┐
  │ ██████████████████████████████████████████████░░│
  │ 98.7% explained                          1.3%  │
  └─────────────────────────────────────────────────┘
```

---

## 9. Regularization — Ridge vs. Lasso vs. Elastic Net

```mermaid
graph TD
    OLS["Standard OLS — Loss = MSE, no penalty"] --> RIDGE
    OLS --> LASSO
    OLS --> ELASTIC
    RIDGE["Ridge L2: MSE + λΣβⱼ²"] --> RIDGE_D["Shrinks all, never zeros"] --> USE_R["Use when: all features matter"]
    LASSO["Lasso L1: MSE + λΣ❘βⱼ❘"] --> LASSO_D["Can zero out = feature selection"] --> USE_L["Use when: many irrelevant features"]
    ELASTIC["Elastic Net: L1 + L2 combined"] --> ELASTIC_D["Selection + shrinkage"] --> USE_E["Use when: correlated features"]
    style OLS fill:#252840,stroke:#f5b731,color:#c8cfe0
    style RIDGE fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style LASSO fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style ELASTIC fill:#1a1d2e,stroke:#22d3a7,color:#e2e8f0
    style USE_R fill:#1a2a1f,stroke:#5eaeff,color:#c8d8c0
    style USE_L fill:#1a2a1f,stroke:#7c6aff,color:#c8d8c0
    style USE_E fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

---

## 10. Linear vs. Logistic Regression

```mermaid
graph TD
    subgraph LIN["Linear Regression"]
        L1["Output: continuous"] --> L2["Loss: MSE"] --> L3["Closed-form solution"] --> L4["No activation"]
    end
    subgraph LOG["Logistic Regression"]
        G1["Output: probability 0-1"] --> G2["Loss: Log Loss"] --> G3["Gradient descent only"] --> G4["Sigmoid activation"]
    end
    LIN --> SHARED["Shared: z = β₀ + β₁x₁ + ..., linear in parameters"]
    LOG --> SHARED
    style LIN fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style LOG fill:#0e1117,stroke:#7c6aff,color:#e2e8f0
    style SHARED fill:#252840,stroke:#f5b731,color:#c8cfe0
```

---

## 11. Complete Pipeline

```mermaid
flowchart TD
    START(["Raw Data"]) --> SPLIT["Train/Test Split"] --> SCALE["Feature Scaling"]
    SCALE --> CHOOSE{"Method?"}
    CHOOSE -->|Small data| NE["Normal Equation"]
    CHOOSE -->|Large data| GD_ITER["Gradient Descent"]
    NE --> TRAINED["Trained Model"]
    GD_ITER --> TRAINED
    TRAINED --> EVAL["Evaluate: MSE, R², Adj R²"]
    EVAL --> CHECK{"Good enough?"}
    CHECK -->|Yes| DEPLOY(["Deploy"])
    CHECK -->|No| FIX["Improve: features, regularize, transform"]
    FIX --> SCALE
    style START fill:#252840,stroke:#f5b731,color:#c8cfe0
    style TRAINED fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style DEPLOY fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style FIX fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

---

## 12. Interview Decision Tree 🎯

```mermaid
graph TD
    Q1{"Explain linear regression?"} -->|Answer| A1["Best-fit line minimizing MSE"]
    Q1 -->|Next| Q2{"How solve for β?"}
    Q2 -->|Answer| A2["Normal equation or gradient descent"]
    Q2 -->|Next| Q3{"What is R²?"}
    Q3 -->|Answer| A3["Variance explained: 1 − SS_res/SS_tot"]
    Q3 -->|Next| Q4{"Assumptions?"}
    Q4 -->|Answer| A4["Linearity, independence, homoscedasticity, normality"]
    style Q1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A4 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```
