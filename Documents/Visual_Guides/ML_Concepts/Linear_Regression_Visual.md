# Linear Regression: Visual Guide with Mermaid Diagrams

> Visual companion to `Documents/ML_Concepts/Basic/Linear_Regression_Complete_Guide.md`.
> Every diagram has explanatory text — what it shows, why it matters, and how to read it.

---

## 1. The Big Idea — Finding the Best Line

Linear regression finds the line (or hyperplane) that minimizes the total squared distance between the data points and the line. The diagram shows the full pipeline: features go in, a linear combination is computed, and a continuous prediction comes out.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    INPUT["📊 Features<br/>Employees, Rating,<br/>Ad Spend"] --> LINEAR["ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃<br/><i>Weighted sum of features<br/>Can be any number</i>"]
    LINEAR --> OUTPUT["💰 Prediction<br/>Daily Sales = $486.90<br/><i>Continuous value</i>"]

    style INPUT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style LINEAR fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style OUTPUT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Read left-to-right: raw features (yellow) → linear combination (blue) → continuous prediction (green). Unlike logistic regression, there's no sigmoid — the output can be any real number.

---

## 2. Simple vs. Multiple Linear Regression

Simple regression uses one feature and fits a line. Multiple regression uses several features and fits a hyperplane. The diagram contrasts the two approaches and shows how adding features changes the model.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph SIMPLE["Simple: 1 Feature"]
        S_IN["Employees"] --> S_EQ["ŷ = 32.42 + 99.35x<br/>β₁ = 99.35"]
        S_EQ --> S_R2["R² = 0.987"]
    end

    subgraph MULTIPLE["Multiple: 3 Features"]
        M_IN1["Employees"] --> M_EQ["ŷ = −20.5 + 28.4x₁<br/>+ 45.2x₂ + 1.42x₃"]
        M_IN2["Rating"] --> M_EQ
        M_IN3["Ad Spend"] --> M_EQ
        M_EQ --> M_R2["R² ≈ 0.998"]
    end

    S_R2 --> COMPARE["β₁ dropped from 99.35 → 28.4<br/><i>Simple model: Employees captured<br/>ALL variation (proxy for everything)<br/>Multiple model: isolates UNIQUE effect</i>"]
    M_R2 --> COMPARE

    style SIMPLE fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style MULTIPLE fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style COMPARE fill:#252840,stroke:#f5b731,color:#c8cfe0
    style S_R2 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style M_R2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

The key insight is in the yellow box: when you add more features, each coefficient shrinks because it now represents only that feature's unique contribution, not a proxy for everything else.

---

## 3. The Best-Fit Line — Residuals

The best-fit line minimizes the sum of squared residuals (vertical distances from points to the line). The ASCII plot shows the line through our data with residuals marked.

```
  Daily Sales ($)
  650 │                              ● S3 ($620)
      │                            ╱ ↕ residual
  550 │                     ● S7 ╱($540)
      │                   ● S1 ╱($520)
  450 │              ● S5╱($450)
      │                ╱
  350 │         ● S6 ╱($340)
      │        ● S2╱($310)
  250 │     ● S8╱($250)
      │   ● S4╱($210)
      │     ╱
  150 │   ╱  ŷ = 32.42 + 99.35x
      │ ╱
      └──┬────┬────┬────┬────┬────┬──→ Employees
         1    2    3    4    5    6

  Residuals (vertical distances):
  S1: +$9    S3: +$9    S5: -$20   S7: -$11
  S2: +$20   S4: +$21   S6: -$10   S8: -$19
```

The line passes through the center of the data. Points above the line have positive residuals (overpredicted), points below have negative residuals (underpredicted). The model minimizes the sum of these squared distances.

---

## 4. Loss Functions Compared

Linear regression typically uses MSE, but there are alternatives. This diagram shows how MSE, MAE, and Huber loss respond differently to errors — especially outliers.

```
  Loss
  100│ ●                              MSE = error²
     │  ╲                             (penalizes large errors heavily)
  80 │   ╲
     │    ╲
  60 │     ╲
     │      ╲
  40 │       ╲
     │        ╲
  20 │         ╲    ╱ MAE = |error|
     │          ╲  ╱  (linear penalty)
   0 │───────────●───────────→ Error
    -10  -8  -6  -4  -2  0  2  4  6  8  10

  Error = 2:   MSE = 4,    MAE = 2
  Error = 10:  MSE = 100,  MAE = 10

  MSE cares 25× more about the big error.
  MAE treats them proportionally (5× more).
```

---

## 5. Two Ways to Solve — Normal Equation vs. Gradient Descent

Linear regression is unique among ML models: it has both a closed-form solution and an iterative one. The diagram compares the two approaches.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    DATA["Training Data<br/>X, y"] --> NE & GD

    subgraph NE["Normal Equation"]
        NE1["β = (XᵀX)⁻¹Xᵀy"] --> NE2["One-shot computation<br/>O(p³) complexity"]
        NE2 --> NE3["✅ Exact solution<br/>✅ No hyperparameters<br/>❌ Slow for large p<br/>❌ Fails if XᵀX singular"]
    end

    subgraph GD["Gradient Descent"]
        GD1["Initialize β = 0"] --> GD2["∂MSE/∂βⱼ = -(2/n)Σ(yᵢ-ŷᵢ)xᵢⱼ"]
        GD2 --> GD3["βⱼ = βⱼ - η × gradient"]
        GD3 --> GD4{"Converged?"}
        GD4 -->|No| GD2
        GD4 -->|Yes| GD5["✅ Scales to big data<br/>✅ Works always<br/>❌ Needs learning rate<br/>❌ Many iterations"]
    end

    NE3 --> SAME["Both find the<br/>SAME optimal β"]
    GD5 --> SAME

    style DATA fill:#252840,stroke:#f5b731,color:#c8cfe0
    style NE fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style GD fill:#0e1117,stroke:#7c6aff,color:#e2e8f0
    style SAME fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Both paths lead to the same answer (green box). The normal equation is a direct computation (blue) — plug in data, get weights. Gradient descent is iterative (purple) — start somewhere, improve step by step. Choose based on dataset size and feature count.

---

## 6. Gradient Descent — Walking Downhill

The MSE loss surface for linear regression is a convex bowl — there's exactly one minimum. Gradient descent starts at a random point and follows the slope downhill. The ASCII plot shows the loss decreasing over iterations.

```
  MSE Loss
  800 │●
      │ ╲
  600 │  ╲
      │   ╲
  400 │    ╲
      │     ╲
  200 │      ╲──────────────────
      │                         ●  converged
    0 │
      └────┬────┬────┬────┬────┬──→ Iterations
           0   200  400  600  800

  Iteration 0:   β₀=0, β₁=0       MSE = 164,025
  Iteration 100: β₀=15, β₁=60     MSE = 1,200
  Iteration 500: β₀=31, β₁=97     MSE = 260
  Iteration 800: β₀=32.4, β₁=99.3 MSE = 249  ← converged!
```

The curve drops steeply at first (big improvements) then flattens (fine-tuning). This is typical of convex optimization — you make fast progress early, then slow down near the minimum.

---

## 7. The Four Assumptions — Visual Diagnostics

Each assumption of linear regression can be checked with a specific plot. The diagram maps assumptions to their diagnostic plots and what violations look like.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    A1["1. Linearity<br/><i>Residuals vs. Predicted</i>"] --> A1G["✅ Random scatter<br/>around zero"]
    A1 --> A1B["❌ Curved pattern<br/>→ Add polynomial terms"]

    A2["2. Independence<br/><i>Durbin-Watson test</i>"] --> A2G["✅ DW ≈ 2.0"]
    A2 --> A2B["❌ DW far from 2<br/>→ Autocorrelation"]

    A3["3. Homoscedasticity<br/><i>Residuals vs. Predicted</i>"] --> A3G["✅ Uniform band"]
    A3 --> A3B["❌ Cone/fan shape<br/>→ Log-transform y"]

    A4["4. Normality of Errors<br/><i>Q-Q Plot</i>"] --> A4G["✅ Points on diagonal"]
    A4 --> A4B["❌ S-curve or tails<br/>→ Transform y"]

    style A1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A1G fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A2G fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A3G fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A4G fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A1B fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style A2B fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style A3B fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style A4B fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

Yellow boxes = assumptions. Green = what "good" looks like. Red = violations and fixes. Always check these plots after fitting a model — they tell you whether to trust your results.

---

## 8. R² — What It Means Visually

R² measures how much of the total variance your model explains. The diagram shows the decomposition: total variance = explained variance + unexplained variance.

```
  Total Variance (SS_tot)
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │   Explained by model (SS_reg)     Unexplained   │
  │   ████████████████████████████    ░░ (SS_res)   │
  │   ████████████████████████████    ░░            │
  │                                                 │
  └─────────────────────────────────────────────────┘

  R² = SS_reg / SS_tot = 1 - SS_res / SS_tot

  Our model: R² = 0.987
  ┌─────────────────────────────────────────────────┐
  │ ██████████████████████████████████████████████░░│
  │ 98.7% explained                          1.3%  │
  └─────────────────────────────────────────────────┘

  Employees explains 98.7% of sales variation!
```

---

## 9. Regularization — Ridge vs. Lasso vs. Elastic Net

Regularization adds a penalty to prevent overfitting. The diagram shows how each method constrains the coefficients differently.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    OLS["Standard Linear Regression<br/>Loss = MSE<br/><i>No penalty — can overfit</i>"] --> RIDGE & LASSO & ELASTIC

    RIDGE["Ridge (L2)<br/>Loss = MSE + λΣβⱼ²<br/><i>Shrinks all coefficients<br/>Never zeros them out</i>"]
    LASSO["Lasso (L1)<br/>Loss = MSE + λΣ|βⱼ|<br/><i>Can zero out coefficients<br/>= feature selection</i>"]
    ELASTIC["Elastic Net (L1+L2)<br/>Loss = MSE + λ₁Σ|βⱼ| + λ₂Σβⱼ²<br/><i>Best of both worlds</i>"]

    RIDGE --> USE_R["Use when:<br/>All features matter<br/>Correlated features"]
    LASSO --> USE_L["Use when:<br/>Many irrelevant features<br/>Want sparse model"]
    ELASTIC --> USE_E["Use when:<br/>Many features, some correlated<br/>Want selection + shrinkage"]

    style OLS fill:#252840,stroke:#f5b731,color:#c8cfe0
    style RIDGE fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style LASSO fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style ELASTIC fill:#1a1d2e,stroke:#22d3a7,color:#e2e8f0
    style USE_R fill:#1a2a1f,stroke:#5eaeff,color:#c8d8c0
    style USE_L fill:#1a2a1f,stroke:#7c6aff,color:#c8d8c0
    style USE_E fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Top (yellow) = standard regression with no penalty. Middle row = three regularization methods with their loss functions. Bottom row = when to use each. The key difference: Lasso can eliminate features entirely (sparse), Ridge just shrinks them (dense).

---

## 10. Linear Regression vs. Logistic Regression

A common interview question. The diagram highlights the key differences between the two models.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    subgraph LIN["Linear Regression"]
        L1["Output: continuous<br/>(sales, price, temp)"]
        L2["Loss: MSE"]
        L3["Solution: closed-form<br/>β = (XᵀX)⁻¹Xᵀy"]
        L4["Activation: none<br/>ŷ = z directly"]
    end

    subgraph LOG["Logistic Regression"]
        G1["Output: probability<br/>(0 to 1, then threshold)"]
        G2["Loss: Log Loss"]
        G3["Solution: gradient<br/>descent only"]
        G4["Activation: sigmoid<br/>P = σ(z)"]
    end

    LIN --- SHARED["Shared:<br/>Both compute z = β₀ + β₁x₁ + ...<br/>Both use gradient descent<br/>Both are linear in parameters"]
    LOG --- SHARED

    style LIN fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style LOG fill:#0e1117,stroke:#7c6aff,color:#e2e8f0
    style SHARED fill:#252840,stroke:#f5b731,color:#c8cfe0
```

Blue = linear regression. Purple = logistic regression. Yellow = what they share. The core difference: linear regression outputs z directly, logistic regression wraps z in a sigmoid to get a probability.

---

## 11. Complete Pipeline Flowchart

The end-to-end linear regression pipeline from raw data to prediction.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
flowchart TD
    START(["Raw Data"]) --> SPLIT["Train/Test Split"]
    SPLIT --> SCALE["Feature Scaling<br/>(if using gradient descent<br/>or regularization)"]
    SCALE --> CHOOSE{"Method?"}
    CHOOSE -->|"Small data"| NE["Normal Equation<br/>β = (XᵀX)⁻¹Xᵀy"]
    CHOOSE -->|"Large data"| GD["Gradient Descent<br/>Iterate until converged"]
    NE --> TRAINED["Trained Model<br/>ŷ = β₀ + β₁x₁ + ..."]
    GD --> TRAINED
    TRAINED --> EVAL["Evaluate on Test Set<br/>MSE, RMSE, R², Adj R²"]
    EVAL --> CHECK{"R² good?<br/>Assumptions met?"}
    CHECK -->|"Yes"| DEPLOY(["Deploy / Predict"])
    CHECK -->|"No"| FIX["Add features, regularize,<br/>transform, or try<br/>non-linear model"]
    FIX --> SCALE

    style START fill:#252840,stroke:#f5b731,color:#c8cfe0
    style TRAINED fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style DEPLOY fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style FIX fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

Follow top-to-bottom: data → split → scale → solve → evaluate → deploy or iterate. The red box is the feedback loop — if the model isn't good enough, you go back and improve it.

---

## 12. Interview Decision Tree 🎯

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    Q1{"Explain linear<br/>regression?"} -->|Answer| A1["Best-fit line minimizing MSE<br/>ŷ = β₀ + β₁x₁ + ...<br/>Coefficients = per-unit change in y"]
    Q1 -->|Next Q| Q2{"How do you<br/>solve for β?"}
    Q2 -->|Answer| A2["Normal equation: β = (XᵀX)⁻¹Xᵀy<br/>Or gradient descent for large data<br/>Both give the same optimal β"]
    Q2 -->|Next Q| Q3{"What is R²?"}
    Q3 -->|Answer| A3["Proportion of variance explained<br/>R² = 1 − SS_res/SS_tot<br/>Use Adjusted R² to penalize extra features"]
    Q3 -->|Next Q| Q4{"Assumptions?"}
    Q4 -->|Answer| A4["Linearity, independence,<br/>homoscedasticity, normal errors,<br/>no multicollinearity"]

    style Q1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A4 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

---

> 💡 **How to view:** GitHub (native), VS Code (Mermaid extension), Obsidian (built-in), or [mermaid.live](https://mermaid.live)
