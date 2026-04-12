# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="📈 Linear Regression", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
.block-container, [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #000000 !important;
}
.stApp, .stMarkdown, p, span, label { color: #e2e8f0 !important; }
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }

.stApp { font-family: 'Inter', sans-serif; }
.concept-card {
    background: linear-gradient(135deg, #1a1d2e, #1f2235);
    border: 1px solid #2d3148; border-radius: 14px; padding: 1.1rem 1.3rem;
    margin: 0.4rem 0; line-height: 1.85; font-size: 0.91rem; color: #c8cfe0;
}
.concept-card b { color: #e2e8f0; }
.output-label {
    display: inline-block; background: #22d3a722; color: #22d3a7;
    font-size: 0.72rem; padding: 3px 10px; border-radius: 8px;
    font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px;
}
.concept-label {
    display: inline-block; background: #7c6aff22; color: #7c6aff;
    font-size: 0.72rem; padding: 3px 10px; border-radius: 8px;
    font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px;
}
.insight-box {
    background: linear-gradient(135deg, #1a2a1f, #1f3528);
    border: 1px solid #2a5a3a; border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.87rem; color: #c8d8c0; line-height: 1.7;
}
.insight-box b { color: #d0f0e0; }
.warn-box {
    background: linear-gradient(135deg, #2a1a1e, #351f25);
    border: 1px solid #5a2a3a; border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.87rem; color: #d8a8b8; line-height: 1.7;
}
.warn-box b { color: #f0c8d8; }
.math-box {
    background: #252840; border-left: 4px solid #f5b731;
    border-radius: 0 10px 10px 0; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.87rem; color: #c8cfe0; line-height: 1.8;
}
.math-box b { color: #f5b731; }
.iq-card {
    background: #181a27; border: 1px solid #2d3148; border-radius: 14px;
    padding: 1.1rem 1.4rem; margin-bottom: 0.5rem; border-left: 4px solid;
}
.iq-title { font-size: 0.93rem; font-weight: 700; color: #e2e8f0; line-height: 1.5; }
.iq-meta { font-size: 0.7rem; margin-top: 5px; }
.iq-tag { display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 0.67rem; font-weight: 600; }
.iq-answer { background: #1a2a1f; border: 1px solid #2a5a3a; border-radius: 12px; padding: 1rem 1.3rem; margin-top: 0.3rem; color: #c8d8c0; font-size: 0.88rem; line-height: 1.8; }
.iq-answer b { color: #d0f0e0; }
.iq-tip { background: #252840; border-left: 4px solid #f5b731; border-radius: 0 10px 10px 0; padding: 0.6rem 1rem; margin-top: 0.3rem; font-size: 0.84rem; color: #c8cfe0; line-height: 1.6; }
.iq-tip b { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

DL = dict(
    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    xaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'), title_font=dict(color='#8892b0')),
    yaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'), title_font=dict(color='#8892b0')),
    font=dict(color='#e2e8f0'), legend=dict(font=dict(color='#8892b0')),
    margin=dict(t=40, b=40, l=40, r=40),
)
DIFF_C = {"Easy": "#22d3a7", "Medium": "#f5b731", "Hard": "#f45d6d"}


def iq(questions):
    st.markdown("---")
    st.markdown("### 🎤 Interview Questions")
    for i, q in enumerate(questions, 1):
        d = q.get("d", "Medium"); c = DIFF_C.get(d, "#7c6aff")
        cos = q.get("c", [])
        tags = f'<span class="iq-tag" style="background:{c}22;color:{c}">{d}</span> '
        tags += " ".join(f'<span class="iq-tag" style="background:#7c6aff15;color:#8892b0">{x}</span>' for x in cos)
        st.markdown(f'<div class="iq-card" style="border-left-color:{c}"><div class="iq-title">Q{i}. {q["q"]}</div><div class="iq-meta">{tags}</div></div>', unsafe_allow_html=True)
        with st.expander(f"💡 Answer — Q{i}"):
            st.markdown(f'<div class="iq-answer">{q["a"]}</div>', unsafe_allow_html=True)
            if q.get("t"):
                st.markdown(f'<div class="iq-tip">🎯 <b>Tip:</b> {q["t"]}</div>', unsafe_allow_html=True)


def split_row(concept_html, code_str, output_func, concept_title="", output_title=""):
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<span class="concept-label">📖 CONCEPT</span>', unsafe_allow_html=True)
        if concept_title:
            st.markdown(f"#### {concept_title}")
        st.markdown(concept_html, unsafe_allow_html=True)
    with right:
        st.markdown('<span class="output-label">💻 CODE + OUTPUT</span>', unsafe_allow_html=True)
        if output_title:
            st.markdown(f"#### {output_title}")
        with st.expander("📝 Show Code", expanded=False):
            st.code(code_str, language="python")
        output_func()
    st.divider()


# ── Data ──
np.random.seed(42)
n = 200
employees = np.random.randint(2, 12, n)
rating = np.round(np.random.uniform(2.5, 5.0, n), 1)
ad_spend = np.random.randint(50, 300, n)
daily_sales = (30 + 25 * employees + 50 * rating + 1.5 * ad_spend
               + np.random.normal(0, 30, n)).astype(int)
df = pd.DataFrame({'employees': employees, 'rating': rating,
                    'ad_spend': ad_spend, 'daily_sales': daily_sales})

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 📈 Linear Regression")
    st.caption("Concept | Code + Output")
    st.divider()
    module = st.radio("Module:", [
        "🏠 Overview",
        "📐 Simple Regression",
        "📊 Multiple Regression",
        "🧮 Normal Eq vs Gradient Descent",
        "✅ Assumptions & Diagnostics",
        "🛡️ Regularization",
        "📏 Evaluation Metrics",
    ], label_visibility="collapsed")


# ═══════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════
if module == "🏠 Overview":
    st.markdown("# 📈 Linear Regression Deep Dive")
    st.caption("Concept on Left · Code + Output on Right")

    st.markdown("""<div class="concept-card">
    Linear regression is the <b>foundation of predictive modeling</b>. It finds the best straight line
    (or hyperplane) through your data to predict a continuous number — like daily sales, house prices, or temperature.
    <br><br>This module covers everything from computing coefficients by hand to regularization techniques used in production.
    </div>""", unsafe_allow_html=True)

    modules = [
        ("📐", "Simple Regression", "Core", "#5eaeff", "One feature, one line — β₁ = Cov(x,y)/Var(x)"),
        ("📊", "Multiple Regression", "Core", "#22d3a7", "Many features, isolating each one's unique effect"),
        ("🧮", "Normal Eq vs Gradient Descent", "Math", "#f5b731", "Two ways to find optimal weights — and when to use each"),
        ("✅", "Assumptions & Diagnostics", "Practice", "#7c6aff", "The 4 assumptions and how to check them visually"),
        ("🛡️", "Regularization", "Advanced", "#f45d6d", "Ridge, Lasso, Elastic Net — preventing overfitting"),
        ("📏", "Evaluation Metrics", "Core", "#a78bfa", "MSE, RMSE, R², Adjusted R² — measuring model quality"),
    ]

    for icon, title, week, color, desc in modules:
        st.markdown(f"""<div class="concept-card" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span>
        <span class="iq-tag" style="background:{color}22;color:{color}">{week}</span>
        <b style="margin-left:6px">{title}</b>
        <br><span style="color:#8892b0">{desc}</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# SIMPLE REGRESSION
# ═══════════════════════════════════════
elif module == "📐 Simple Regression":
    st.markdown("# 📐 Simple Linear Regression")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>The simplest ML model:</b> draw the best straight line through data.
    <br>ŷ = β₀ + β₁x — that's it. One feature, one slope, one intercept.
    <br><br>🎯 The goal: find β₀ and β₁ that minimize the sum of squared errors.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: By-hand calculation
    def show_by_hand():
        x = df['employees'].values.astype(float)
        y = df['daily_sales'].values.astype(float)
        x_mean, y_mean = x.mean(), y.mean()
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        var_x = np.mean((x - x_mean) ** 2)
        b1 = cov_xy / var_x
        b0 = y_mean - b1 * x_mean
        results = pd.DataFrame({
            'Statistic': ['x̄ (mean employees)', 'ȳ (mean sales)', 'Cov(x,y)', 'Var(x)', 'β₁ = Cov/Var', 'β₀ = ȳ − β₁x̄'],
            'Value': [f'{x_mean:.2f}', f'${y_mean:.2f}', f'{cov_xy:.2f}', f'{var_x:.2f}', f'{b1:.2f}', f'{b0:.2f}']
        })
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.markdown(f"""<div class="insight-box">💡 <b>Model:</b> ŷ = {b0:.2f} + {b1:.2f} × Employees
        <br>Each additional employee → +${b1:.2f} daily sales</div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 How do we find the best line?</b>
        <br><br>Two formulas from statistics:
        <br>• <b>β₁ = Cov(x, y) / Var(x)</b> — the slope
        <br>• <b>β₀ = ȳ − β₁x̄</b> — the intercept
        <br><br>Covariance measures how x and y move together. Variance measures how spread out x is. Their ratio gives the slope.
        </div>
        <div class="math-box">
        <b>📐 Step by Step:</b>
        <br><br><b>1.</b> Compute means: x̄, ȳ
        <br><b>2.</b> Cov(x,y) = (1/n) × Σ(xᵢ − x̄)(yᵢ − ȳ)
        <br><b>3.</b> Var(x) = (1/n) × Σ(xᵢ − x̄)²
        <br><b>4.</b> β₁ = Cov / Var
        <br><b>5.</b> β₀ = ȳ − β₁ × x̄
        </div>""",
        code_str='''x, y = df['employees'], df['daily_sales']
x_mean, y_mean = x.mean(), y.mean()
cov_xy = np.mean((x - x_mean) * (y - y_mean))
var_x = np.mean((x - x_mean) ** 2)

beta_1 = cov_xy / var_x
beta_0 = y_mean - beta_1 * x_mean''',
        output_func=show_by_hand,
        concept_title="📐 Computing β by Hand",
        output_title="Step-by-Step Calculation"
    )

    # Row 2: Best-fit line plot
    def show_fit_line():
        x = df['employees'].values.astype(float)
        y = df['daily_sales'].values.astype(float)
        b1 = np.mean((x - x.mean()) * (y - y.mean())) / np.mean((x - x.mean()) ** 2)
        b0 = y.mean() - b1 * x.mean()
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b0 + b1 * x_line
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Stores',
                                 marker=dict(color='#5eaeff', opacity=0.5)))
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name=f'ŷ = {b0:.1f} + {b1:.1f}x',
                                 line=dict(color='#f45d6d', width=3)))
        fig.update_layout(**DL, title='Best-Fit Line: Sales vs Employees', height=400,
                          xaxis_title='Employees', yaxis_title='Daily Sales ($)')
        st.plotly_chart(fig, use_container_width=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🎯 What makes it the "best" line?</b>
        <br><br>It minimizes the <b>sum of squared residuals</b> — the vertical distances from each point to the line.
        <br><br>Residual = actual − predicted = yᵢ − ŷᵢ
        <br><br>We square them so positive and negative errors don't cancel out, and so large errors are penalized more.
        </div>
        <div class="warn-box">⚠️ <b>Outliers matter a lot!</b> Because we square errors, one extreme point can pull the entire line toward it. Always check for outliers before fitting.</div>""",
        code_str='''from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")''',
        output_func=show_fit_line,
        concept_title="📉 The Best-Fit Line",
        output_title="Scatter + Regression Line"
    )

    iq([
        {"q": "Explain linear regression to a non-technical person.", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "Linear regression draws the <b>best straight line</b> through data to make predictions. Example: 'For every extra employee, daily sales go up by about $25.'",
         "t": "Use a concrete example everyone understands."},
        {"q": "What does the slope (β₁) represent?", "d": "Easy", "c": ["Meta", "Microsoft"],
         "a": "β₁ is the <b>change in y for a one-unit increase in x</b>, holding everything else constant. It's the rate of change — how much the prediction moves when the feature moves by 1.",
         "t": "Always mention 'holding other variables constant' for multiple regression."},
        {"q": "Can linear regression be used for classification?", "d": "Medium", "c": ["Amazon"],
         "a": "Technically yes, but it's a bad idea. Linear regression can predict values outside [0,1], which makes no sense for probabilities. That's why we use <b>logistic regression</b> for classification — it wraps the linear output in a sigmoid to get valid probabilities.",
         "t": "This is a great segue into logistic regression."},
    ])


# ═══════════════════════════════════════
# MULTIPLE REGRESSION
# ═══════════════════════════════════════
elif module == "📊 Multiple Regression":
    st.markdown("# 📊 Multiple Linear Regression")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    Real-world predictions use <b>many features</b>. Multiple regression extends the simple model:
    <br>ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
    <br><br>Each coefficient tells you that feature's <b>unique contribution</b>, holding all others constant.
    </div>""", unsafe_allow_html=True)
    st.divider()

    def show_multi_model():
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': [f'{c:.2f}' for c in model.coef_],
            'Meaning': [f'Each +1 → ${c:.2f} sales' for c in model.coef_]
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.markdown(f"""<div class="insight-box">💡 <b>R² = {r2:.4f}</b> | <b>RMSE = ${rmse:.2f}</b>
        <br>The model explains {r2*100:.1f}% of sales variance using all 3 features.</div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why did the Employees coefficient change?</b>
        <br><br>Simple model: β₁ ≈ 25 (Employees captures ALL variation)
        <br>Multiple model: β₁ ≈ 25 (now it's the UNIQUE effect)
        <br><br>In simple regression, Employees was a proxy for everything. Multiple regression <b>isolates each feature's contribution</b>.
        </div>
        <div class="math-box">
        <b>📐 The Normal Equation:</b>
        <br><br>β = (XᵀX)⁻¹Xᵀy
        <br><br>This gives the exact optimal weights in one computation.
        <br>No iteration needed — just matrix math.
        </div>
        <div class="warn-box">⚠️ <b>Multicollinearity:</b> If features are highly correlated, coefficients become unstable. Check VIF (Variance Inflation Factor) — should be &lt; 5.</div>""",
        code_str='''from sklearn.linear_model import LinearRegression

features = ['employees', 'rating', 'ad_spend']
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['daily_sales'], test_size=0.3, random_state=42)

model = LinearRegression().fit(X_train, y_train)
print(f"Intercept: {model.intercept_:.2f}")
for f, c in zip(features, model.coef_):
    print(f"  {f}: {c:.2f}")''',
        output_func=show_multi_model,
        concept_title="📊 All Features Together",
        output_title="Multiple Regression Coefficients"
    )

    # Row 2: Actual vs Predicted
    def show_actual_vs_pred():
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions',
                                 marker=dict(color='#5eaeff', opacity=0.6)))
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', name='Perfect', line=dict(color='#f45d6d', dash='dash')))
        fig.update_layout(**DL, title=f'Actual vs Predicted (R²={r2_score(y_test, y_pred):.3f})',
                          height=400, xaxis_title='Actual Sales ($)', yaxis_title='Predicted Sales ($)')
        st.plotly_chart(fig, use_container_width=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>📊 Reading the Actual vs Predicted plot:</b>
        <br><br>• Points ON the red line = perfect predictions
        <br>• Points ABOVE = overpredicted
        <br>• Points BELOW = underpredicted
        <br>• Tighter cluster around the line = better model
        <br><br>This is the single most useful diagnostic plot for regression.
        </div>""",
        code_str='''y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('Actual'); plt.ylabel('Predicted')''',
        output_func=show_actual_vs_pred,
        concept_title="📊 Actual vs Predicted",
        output_title="How Good Are Our Predictions?"
    )

    iq([
        {"q": "What's the difference between simple and multiple linear regression?", "d": "Easy", "c": ["Google"],
         "a": "Simple uses <b>one feature</b> (ŷ = β₀ + β₁x). Multiple uses <b>several</b> (ŷ = β₀ + β₁x₁ + ... + βₚxₚ). Multiple regression isolates each feature's unique contribution by controlling for the others.",
         "t": "Mention 'holding other variables constant' — interviewers love that phrase."},
        {"q": "How do you solve for the coefficients?", "d": "Medium", "c": ["Amazon", "Meta"],
         "a": "Two ways: <b>(1) Normal equation</b> β = (XᵀX)⁻¹Xᵀy — direct, O(p³), good for small data. <b>(2) Gradient descent</b> — iterative, scales to large data. Both find the same optimal β.",
         "t": "Know when to use each: normal equation for small data, gradient descent for large."},
    ])


# ═══════════════════════════════════════
# NORMAL EQ vs GRADIENT DESCENT
# ═══════════════════════════════════════
elif module == "🧮 Normal Eq vs Gradient Descent":
    st.markdown("# 🧮 Normal Equation vs Gradient Descent")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    Linear regression is unique: it has <b>both</b> a closed-form solution (normal equation) and an iterative one (gradient descent).
    Both find the same optimal weights — the choice depends on your data size.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Normal Equation
    def show_normal_eq():
        from sklearn.model_selection import train_test_split
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_mat = np.column_stack([np.ones(len(X_train)), X_train.values])
        beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_train.values
        result = pd.DataFrame({
            'Parameter': ['β₀ (intercept)', 'β₁ (employees)', 'β₂ (rating)', 'β₃ (ad_spend)'],
            'Value': [f'{b:.4f}' for b in beta]
        })
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.markdown("""<div class="insight-box">💡 One matrix computation → exact answer. No iterations, no learning rate.</div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>📐 Normal Equation: β = (XᵀX)⁻¹Xᵀy</b>
        <br><br>Derived by setting the gradient of MSE to zero and solving algebraically.
        <br><br><b>Pros:</b>
        <br>• Exact solution in one step
        <br>• No hyperparameters to tune
        <br><br><b>Cons:</b>
        <br>• O(p³) — slow when features > 1000
        <br>• Fails if XᵀX is singular (multicollinearity)
        </div>
        <div class="math-box">
        <b>📐 Derivation:</b>
        <br>Loss = (y − Xβ)ᵀ(y − Xβ)
        <br>∂Loss/∂β = −2Xᵀ(y − Xβ) = 0
        <br>XᵀXβ = Xᵀy
        <br><b>β = (XᵀX)⁻¹Xᵀy</b>
        </div>""",
        code_str='''# Normal Equation from scratch
X_mat = np.column_stack([np.ones(len(X_train)), X_train.values])
beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_train.values''',
        output_func=show_normal_eq,
        concept_title="📐 Normal Equation",
        output_title="Direct Solution"
    )

    # Row 2: Gradient Descent
    def show_gradient_descent():
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_gd = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        beta = np.zeros(X_gd.shape[1])
        lr, n_iters = 0.01, 500
        losses = []
        for _ in range(n_iters):
            residuals = y_train.values - X_gd @ beta
            losses.append(np.mean(residuals ** 2))
            gradient = -(2 / len(y_train)) * (X_gd.T @ residuals)
            beta = beta - lr * gradient
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, mode='lines', line=dict(color='#5eaeff')))
        fig.update_layout(**DL, title='Loss Convergence', height=300,
                          xaxis_title='Iteration', yaxis_title='MSE')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div class="insight-box">💡 Started at MSE={losses[0]:.0f}, converged to MSE={losses[-1]:.0f} after {n_iters} iterations.</div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>⛰️ Gradient Descent: Walk Downhill</b>
        <br><br>Start with random weights. At each step:
        <br>1. Compute predictions
        <br>2. Compute gradient: ∂MSE/∂βⱼ = −(2/n)Σ(yᵢ − ŷᵢ)xᵢⱼ
        <br>3. Update: βⱼ = βⱼ − η × gradient
        <br>4. Repeat until converged
        <br><br><b>Pros:</b> Scales to millions of rows
        <br><b>Cons:</b> Needs learning rate, many iterations
        </div>
        <div class="warn-box">⚠️ <b>Feature scaling is critical!</b> Without it, gradient descent zig-zags and converges slowly. Always standardize features first.</div>""",
        code_str='''beta = np.zeros(n_features)
lr = 0.01

for i in range(1000):
    y_hat = X @ beta
    gradient = -(2/n) * X.T @ (y - y_hat)
    beta = beta - lr * gradient''',
        output_func=show_gradient_descent,
        concept_title="⛰️ Gradient Descent",
        output_title="Convergence Plot"
    )

    iq([
        {"q": "When would you use gradient descent over the normal equation?", "d": "Medium", "c": ["Google", "Amazon"],
         "a": "When the dataset is <b>large</b> (many features or many rows). The normal equation is O(p³) — with 10,000 features, that's 10¹² operations. Gradient descent scales linearly with data size.",
         "t": "Mention the O(p³) complexity — it shows you understand the tradeoff."},
    ])


# ═══════════════════════════════════════
# ASSUMPTIONS & DIAGNOSTICS
# ═══════════════════════════════════════
elif module == "✅ Assumptions & Diagnostics":
    st.markdown("# ✅ Assumptions & Diagnostics")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    Linear regression makes <b>4 key assumptions</b>. Violating them doesn't crash the model — it means the results may be <b>unreliable</b>.
    Always check these after fitting.
    </div>""", unsafe_allow_html=True)
    st.divider()

    def show_diagnostics():
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals = y_test.values - y_pred

        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                      marker=dict(color='#5eaeff', opacity=0.5)))
            fig1.add_hline(y=0, line_dash='dash', line_color='#f45d6d')
            fig1.update_layout(**DL, title='Residuals vs Predicted', height=300,
                               xaxis_title='Predicted', yaxis_title='Residual')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=residuals, nbinsx=20,
                                        marker_color='#22d3a7', opacity=0.7))
            fig2.update_layout(**DL, title='Residual Distribution', height=300,
                               xaxis_title='Residual', yaxis_title='Count')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""<div class="insight-box">💡 Residual mean: {np.mean(residuals):.2f} (should be ~0) ✅
        <br>Residual std: {np.std(residuals):.2f}</div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>The 4 Assumptions:</b>
        <br><br><b>1. Linearity</b> — relationship is linear
        <br>&nbsp;&nbsp;&nbsp;Check: residuals vs predicted → random scatter
        <br><br><b>2. Independence</b> — errors are independent
        <br>&nbsp;&nbsp;&nbsp;Check: Durbin-Watson test → value near 2
        <br><br><b>3. Homoscedasticity</b> — constant error variance
        <br>&nbsp;&nbsp;&nbsp;Check: residuals vs predicted → uniform band
        <br><br><b>4. Normality of Errors</b> — errors are normal
        <br>&nbsp;&nbsp;&nbsp;Check: histogram/Q-Q plot → bell shape
        </div>
        <div class="math-box">
        <b>📐 Bonus: No Multicollinearity</b>
        <br>Features shouldn't be highly correlated.
        <br>Check: VIF &lt; 5 for each feature.
        <br>Fix: Remove one, or use Ridge regression.
        </div>""",
        code_str='''residuals = y_test - model.predict(X_test)

# Check 1: Residuals vs Predicted
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')

# Check 4: Normality
plt.hist(residuals, bins=20)

# Bonus: VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor''',
        output_func=show_diagnostics,
        concept_title="🔍 Diagnostic Plots",
        output_title="Residual Analysis"
    )

    iq([
        {"q": "What assumptions does linear regression make?", "d": "Medium", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>1)</b> Linear relationship between features and target. <b>2)</b> Independent errors. <b>3)</b> Homoscedasticity (constant error variance). <b>4)</b> Normally distributed errors. <b>5)</b> No multicollinearity among features.",
         "t": "Memorize all 4+1. Then mention how to CHECK each one — that shows depth."},
    ])


# ═══════════════════════════════════════
# REGULARIZATION
# ═══════════════════════════════════════
elif module == "🛡️ Regularization":
    st.markdown("# 🛡️ Regularization: Ridge, Lasso, Elastic Net")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    Regularization adds a <b>penalty</b> to the loss function to prevent overfitting.
    It shrinks coefficients — keeping the model simple and generalizable.
    </div>""", unsafe_allow_html=True)
    st.divider()

    def show_regularization():
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_squared_error
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        results = []
        for name, model in [('OLS', LinearRegression()), ('Ridge α=1', Ridge(alpha=1)),
                             ('Ridge α=10', Ridge(alpha=10)), ('Lasso α=1', Lasso(alpha=1)),
                             ('Lasso α=10', Lasso(alpha=10)), ('Elastic Net', ElasticNet(alpha=1, l1_ratio=0.5))]:
            model.fit(X_tr, y_train)
            pred = model.predict(X_te)
            results.append({
                'Model': name, 'R²': f'{r2_score(y_test, pred):.4f}',
                'RMSE': f'${np.sqrt(mean_squared_error(y_test, pred)):.2f}',
                'β₁': f'{model.coef_[0]:.2f}', 'β₂': f'{model.coef_[1]:.2f}', 'β₃': f'{model.coef_[2]:.2f}'
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>Three flavors of regularization:</b>
        <br><br><b>Ridge (L2):</b> Loss = MSE + λΣβⱼ²
        <br>• Shrinks all coefficients, never zeros them
        <br>• Good when all features matter
        <br><br><b>Lasso (L1):</b> Loss = MSE + λΣ|βⱼ|
        <br>• Can zero out coefficients = feature selection!
        <br>• Good when some features are irrelevant
        <br><br><b>Elastic Net (L1+L2):</b> Best of both worlds
        </div>
        <div class="math-box">
        <b>📐 Choosing λ:</b>
        <br>λ = 0 → no penalty (standard OLS)
        <br>λ = small → slight shrinkage
        <br>λ = large → heavy shrinkage (underfitting)
        <br><br>Use <b>cross-validation</b> to find the best λ.
        </div>""",
        code_str='''from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=1.0).fit(X_train, y_train)
enet = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train, y_train)

# Lasso can zero out coefficients!
print(f"Lasso coefs: {lasso.coef_}")''',
        output_func=show_regularization,
        concept_title="🛡️ Ridge vs Lasso vs Elastic Net",
        output_title="Coefficient Comparison"
    )

    # Row 2: Regularization path
    def show_reg_path():
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        alphas = np.logspace(-2, 3, 50)

        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure()
            for i, feat in enumerate(features):
                coefs = [Ridge(alpha=a).fit(X_tr, y_train).coef_[i] for a in alphas]
                fig1.add_trace(go.Scatter(x=np.log10(alphas), y=coefs, name=feat, mode='lines'))
            fig1.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
            fig1.update_layout(**DL, title='Ridge: Shrinks but never zeros', height=300,
                               xaxis_title='log₁₀(α)', yaxis_title='Coefficient')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            for i, feat in enumerate(features):
                coefs = [Lasso(alpha=a, max_iter=10000).fit(X_tr, y_train).coef_[i] for a in alphas]
                fig2.add_trace(go.Scatter(x=np.log10(alphas), y=coefs, name=feat, mode='lines'))
            fig2.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
            fig2.update_layout(**DL, title='Lasso: Can zero out features', height=300,
                               xaxis_title='log₁₀(α)', yaxis_title='Coefficient')
            st.plotly_chart(fig2, use_container_width=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>📈 Regularization Paths</b>
        <br><br>As α increases (more penalty), coefficients shrink toward zero.
        <br><br><b>Ridge:</b> All coefficients approach zero but never reach it.
        <br><b>Lasso:</b> Coefficients hit exactly zero — features get eliminated.
        <br><br>This is why Lasso is used for <b>feature selection</b>.
        </div>""",
        code_str='''alphas = np.logspace(-2, 3, 50)
ridge_coefs = []
for a in alphas:
    ridge = Ridge(alpha=a).fit(X_train, y_train)
    ridge_coefs.append(ridge.coef_)''',
        output_func=show_reg_path,
        concept_title="📈 Regularization Paths",
        output_title="How Coefficients Shrink"
    )

    iq([
        {"q": "When would you use Ridge vs Lasso?", "d": "Medium", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>Ridge (L2)</b> when you want to keep all features but shrink coefficients — good when all features are somewhat useful or correlated. <b>Lasso (L1)</b> when you suspect some features are irrelevant — it zeros them out for automatic feature selection. <b>Elastic Net</b> combines both.",
         "t": "Mention the geometric intuition: L1 has diamond constraint (corners at axes = zeros), L2 has circular constraint (no corners = no zeros)."},
    ])


# ═══════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════
elif module == "📏 Evaluation Metrics":
    st.markdown("# 📏 Evaluation Metrics")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    How do you know if your model is good? These metrics quantify prediction quality.
    Each tells a different story — use them together.
    </div>""", unsafe_allow_html=True)
    st.divider()

    def show_metrics():
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        features = ['employees', 'rating', 'ad_spend']
        X = df[features]; y = df['daily_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        n, p = len(y_test), len(features)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
        mse = mean_squared_error(y_test, y_pred)
        metrics = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Adjusted R²'],
            'Formula': ['(1/n)Σ(yᵢ−ŷᵢ)²', '√MSE', '(1/n)Σ|yᵢ−ŷᵢ|', '1 − SS_res/SS_tot', '1 − [(1−R²)(n−1)/(n−p−1)]'],
            'Value': [f'{mse:.2f}', f'${np.sqrt(mse):.2f}', f'${mean_absolute_error(y_test, y_pred):.2f}',
                      f'{r2:.4f}', f'{adj_r2:.4f}'],
            'Meaning': [f'Average squared error', f'Avg error in dollars',
                        f'Avg absolute error', f'Explains {r2*100:.1f}% of variance',
                        f'R² penalized for extra features']
        })
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>The key metrics:</b>
        <br><br><b>MSE</b> = (1/n)Σ(yᵢ − ŷᵢ)² — penalizes large errors heavily
        <br><b>RMSE</b> = √MSE — same units as y (dollars)
        <br><b>MAE</b> = (1/n)Σ|yᵢ − ŷᵢ| — less sensitive to outliers
        <br><b>R²</b> = 1 − SS_res/SS_tot — proportion of variance explained
        <br><b>Adjusted R²</b> — penalizes for extra features
        </div>
        <div class="math-box">
        <b>📐 R² Decomposition:</b>
        <br>SS_tot = Σ(yᵢ − ȳ)² — total variance
        <br>SS_res = Σ(yᵢ − ŷᵢ)² — unexplained variance
        <br>SS_reg = SS_tot − SS_res — explained variance
        <br><br>R² = SS_reg / SS_tot = 1 − SS_res / SS_tot
        </div>
        <div class="warn-box">⚠️ <b>R² always increases</b> when you add features — even useless ones. Always use <b>Adjusted R²</b> to compare models with different numbers of features.</div>""",
        code_str='''from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Adjusted R²
n, p = len(y_test), len(features)
adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)''',
        output_func=show_metrics,
        concept_title="📏 All Metrics at a Glance",
        output_title="Model Evaluation"
    )

    iq([
        {"q": "What is R² and what are its limitations?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "R² measures the <b>proportion of variance explained</b> by the model (0 to 1). Limitation: it <b>always increases</b> with more features, even useless ones. Use <b>Adjusted R²</b> which penalizes for extra features.",
         "t": "Mention that R² can be negative if the model is worse than just predicting the mean."},
        {"q": "MSE vs MAE — when to use which?", "d": "Medium", "c": ["Meta"],
         "a": "<b>MSE</b> penalizes large errors more (squared). Use when big errors are especially bad (e.g., medical predictions). <b>MAE</b> treats all errors equally. Use when outliers are expected and you don't want them to dominate. MSE is differentiable everywhere; MAE has a kink at 0.",
         "t": "The differentiability point matters for gradient-based optimization."},
        {"q": "Linear regression vs logistic regression?", "d": "Easy", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>Linear</b> predicts a continuous number (sales, price). <b>Logistic</b> predicts a probability for classification (yes/no). Linear uses MSE loss; logistic uses log loss. Linear has a closed-form solution; logistic requires gradient descent.",
         "t": "Despite the name, logistic regression is a CLASSIFICATION algorithm."},
    ])
