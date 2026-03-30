# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="🤖 Phase 3: ML Concepts", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
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
.pros-cons-box {
    background: linear-gradient(135deg, #1a1d2e, #1f2235);
    border: 1px solid #2d3148; border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.85rem; color: #c8cfe0; line-height: 1.7;
}
.pros-cons-box b { color: #e2e8f0; }
.pros-cons-box .pros { color: #22d3a7; }
.pros-cons-box .cons { color: #f45d6d; }
.pros-cons-box .tips { color: #f5b731; }
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
    """Render split view: Concept on LEFT, Code + Output on RIGHT."""
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

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🤖 Phase 3: ML Concepts")
    st.caption("Concept | Code + Output")
    st.divider()
    module = st.radio("Module:", [
        "🏠 Overview",
        "📖 Bias & Variance",
        "🔮 Prediction & Forecasting",
        "🎓 Supervised Learning",
        "🌳 Decision Trees",
        "🌲 Ensemble Methods",
        "🧩 Unsupervised Learning",
    ], label_visibility="collapsed")


# ═══════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════
if module == "🏠 Overview":
    st.markdown("# 🤖 Phase 3: Machine Learning Concepts")
    st.caption("Concept on Left · Code + Output on Right")

    st.markdown("""<div class="concept-card">
    Phase 1 gave you statistics. Phase 2 gave you data skills. Now we answer the big question:
    <b>"Can a machine learn patterns from data and make predictions on its own?"</b>
    <br><br>This phase covers the core ML concepts through storytelling and interactive visuals.
    </div>""", unsafe_allow_html=True)

    modules = [
        ("📖", "Bias & Variance", "Week 1", "#f5b731", "The central tradeoff in all of ML — start here!"),
        ("🔮", "Prediction & Forecasting", "Week 1", "#f45d6d", "Can we guess the future? Overfitting vs underfitting."),
        ("🎓", "Supervised Learning", "Week 2", "#5eaeff", "Learning with a teacher — classification and regression."),
        ("🌳", "Decision Trees", "Week 2-3", "#22d3a7", "Flowchart of yes/no questions + Gini & Entropy math."),
        ("🌲", "Ensemble Methods", "Week 3-4", "#7c6aff", "Bagging, Boosting, AdaBoost, XGBoost — power of many models."),
        ("🧩", "Unsupervised Learning", "Week 4", "#a78bfa", "Finding hidden patterns without labels — clustering."),
    ]

    for icon, title, week, color, desc in modules:
        st.markdown(f"""<div class="concept-card" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span>
        <span class="iq-tag" style="background:{color}22;color:{color}">{week}</span>
        <b style="margin-left:6px">{title}</b>
        <br><span style="color:#8892b0">{desc}</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# PREDICTION & FORECASTING
# ═══════════════════════════════════════
elif module == "🔮 Prediction & Forecasting":
    st.markdown("# 🔮 Prediction & Forecasting")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Prediction?</b> It's using <b>patterns in data to guess unknown outcomes</b>. Will this customer churn? What's this house worth?
    Forecasting is similar but specifically about <b>future values over time</b> — what will sales be next month?
    <br><br>🎯 <b>Key challenge:</b> Overfitting (memorizing noise) vs underfitting (missing the pattern). Finding the sweet spot is the art of ML.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Prediction Demo
    def show_prediction_demo():
        np.random.seed(42)
        sizes = np.random.uniform(500, 3000, 50)
        prices = sizes * 150 + np.random.normal(0, 30000, 50)
        
        slope, intercept = np.polyfit(sizes, prices, 1)
        house_size = 1500
        predicted_price = slope * house_size + intercept
        
        mc = st.columns(3)
        mc[0].metric("House Size", f"{house_size:,} sq ft")
        mc[1].metric("Predicted Price", f"${predicted_price:,.0f}")
        mc[2].metric("Price/sq ft", f"${slope:.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=prices, mode='markers', marker=dict(color='#7c6aff', size=6, opacity=0.6), name='Past Sales'))
        x_line = np.linspace(400, 3600, 100)
        fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode='lines', line=dict(color='#22d3a7', width=2, dash='dash'), name='Trend'))
        fig.add_trace(go.Scatter(x=[house_size], y=[predicted_price], mode='markers', marker=dict(color='#f45d6d', size=15, symbol='star'), name='Prediction'))
        fig.update_layout(height=250, title="House Price Prediction", xaxis_title="Size (sq ft)", yaxis_title="Price ($)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What's the difference between Prediction and Forecasting?</b>
        <br><br>Both are about guessing unknown values, but they're subtly different:
        <br><br><b>Prediction:</b> Guessing an <b>outcome that exists but is unknown</b>.
        <br>• "Will this customer churn?" (they will or won't — we just don't know yet)
        <br>• "What's this house worth?" (it has a value — we're estimating it)
        <br>• "Is this email spam?" (it is or isn't — we're classifying it)
        <br><br><b>Forecasting:</b> Guessing a <b>future value that doesn't exist yet</b>.
        <br>• "What will sales be next quarter?" (hasn't happened yet)
        <br>• "How many customers will we have in 2025?" (future state)
        <br>• "What will the stock price be tomorrow?" (time-based)
        <br><br><b>Why the distinction matters:</b>
        <br>• Prediction uses cross-sectional data (snapshot in time)
        <br>• Forecasting uses time series data (sequence over time)
        <br>• Different techniques, different validation strategies
        </div>
        <div class="math-box">
        <b>📐 Prediction Example — House Price:</b>
        <br><br><b>Given:</b> Historical data shows:
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Price ≈ $150 × size + $50,000
        <br><br><b>New house:</b> 1,500 sq ft
        <br><br><b>Prediction:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Price = $150 × 1,500 + $50,000
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Price = $225,000 + $50,000 = <b>$275,000</b>
        <br><br>🧠 The house exists, has a true value — we're estimating it from patterns in past sales.
        </div>
        <div class="insight-box">💡 <b>Key insight:</b> Prediction = cross-sectional (comparing things at one point in time). Forecasting = temporal (projecting into the future).</div>""",
        code_str='''import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
sizes = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
prices = np.array([150000, 225000, 300000, 375000, 450000])

# Train model
model = LinearRegression()
model.fit(sizes, prices)

# Predict for new house
new_house = [[1500]]
predicted_price = model.predict(new_house)[0]
print(f"Predicted: ${predicted_price:,.0f}")''',
        output_func=show_prediction_demo,
        concept_title="🎯 Prediction vs Forecasting",
        output_title="House Price Predictor"
    )

    # Row 2: Overfitting vs Underfitting
    def show_fitting_demo():
        np.random.seed(42)
        x_fit = np.linspace(0, 10, 20)
        y_true = 2 * x_fit + 5 + np.random.normal(0, 3, 20)
        
        x_smooth = np.linspace(0, 10, 200)
        
        # Show all three fits
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_fit, y=y_true, mode='markers', marker=dict(color='#22d3a7', size=8), name='Data'))
        
        # Underfit (horizontal line)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.full_like(x_smooth, y_true.mean()), mode='lines', 
                                line=dict(color='#f5b731', width=2, dash='dot'), name='Underfit'))
        
        # Good fit (linear)
        z = np.polyfit(x_fit, y_true, 1)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z, x_smooth), mode='lines', 
                                line=dict(color='#7c6aff', width=3), name='Good Fit'))
        
        # Overfit (high degree polynomial)
        z_over = np.polyfit(x_fit, y_true, 12)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z_over, x_smooth).clip(-5, 35), mode='lines', 
                                line=dict(color='#f45d6d', width=2, dash='dash'), name='Overfit'))
        
        fig.update_layout(height=280, title="Fitting Comparison", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("""<div class="insight-box">
        🟡 <b>Underfit:</b> Too simple — misses the trend<br>
        🟣 <b>Good fit:</b> Captures pattern without noise<br>
        🔴 <b>Overfit:</b> Memorizes noise — fails on new data
        </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Overfitting and Underfitting?</b>
        <br><br>These are the two ways a model can fail:
        <br><br><b>⚠️ Overfitting (Too Complex):</b>
        <br>The model memorizes the training data, including the noise. It performs amazingly on training data but terribly on new data.
        <br><br><b>Analogy:</b> A student who memorizes every practice problem word-for-word. They ace the practice test but fail the real exam because they never learned the underlying concepts.
        <br><br><b>Signs of overfitting:</b>
        <br>• Training accuracy: 98%
        <br>• Test accuracy: 65%
        <br>• Big gap = overfitting!
        <br><br><b>📉 Underfitting (Too Simple):</b>
        <br>The model is too dumb to capture the real pattern. It performs poorly on both training and test data.
        <br><br><b>Analogy:</b> A student who barely studied. They fail both the practice test and the real exam.
        <br><br><b>Signs of underfitting:</b>
        <br>• Training accuracy: 60%
        <br>• Test accuracy: 58%
        <br>• Both low = underfitting!
        </div>
        <div class="math-box">
        <b>📐 Detection Cheat Sheet:</b>
        <br><br><b>Overfitting:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Train: 98% → Test: 65%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Gap = 33% → <b>Overfitting!</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Fix: Simpler model, more data, regularization
        <br><br><b>Underfitting:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Train: 60% → Test: 58%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Both low → <b>Underfitting!</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Fix: More features, complex model
        <br><br><b>Good Fit:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Train: 85% → Test: 82%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Both high, small gap → <b>Sweet spot!</b>
        </div>
        <div class="warn-box">⚠️ <b>The cruel truth:</b> A model that's "too good" on training data is probably overfitting. Always check test performance!</div>""",
        code_str='''from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Underfit: too simple
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
print(f"Train: {model_simple.score(X_train, y_train):.2f}")
print(f"Test: {model_simple.score(X_test, y_test):.2f}")

# Overfit: too complex (degree 15 polynomial)
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X_train)
# This will memorize training data but fail on test''',
        output_func=show_fitting_demo,
        concept_title="⚖️ Overfitting vs Underfitting",
        output_title="See the Difference"
    )

    iq([
        {"q": "What is overfitting? How do you detect and prevent it?", "d": "Medium", "c": ["Google", "Meta"],
         "a": "<b>Overfitting:</b> Model performs great on training but poorly on test — it memorized noise. <b>Detection:</b> Big gap between train and test accuracy. <b>Prevention:</b> Train/test split, cross-validation, regularization, simpler model, more data.",
         "t": "Use the student analogy — memorizing vs understanding."},
    ])


# ═══════════════════════════════════════
# SUPERVISED LEARNING
# ═══════════════════════════════════════
elif module == "🎓 Supervised Learning":
    st.markdown("# 🎓 Supervised Learning")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Supervised Learning?</b> It's <b>learning with labeled examples</b> — you show the model inputs AND correct answers, and it learns the pattern.
    Like a student with a teacher: "Here's the question, here's the answer. Now learn to solve similar problems."
    <br><br>🎯 <b>Two types:</b> Classification (predict categories: Yes/No) and Regression (predict numbers: $350,000).
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Classification vs Regression
    def show_class_vs_reg():
        comparison = pd.DataFrame({
            "Aspect": ["Output", "Question", "Example", "Metrics"],
            "Classification": ["Category (Yes/No, A/B/C)", "Which group?", "Will customer churn?", "Accuracy, Precision, Recall"],
            "Regression": ["Number (continuous)", "How much?", "What's the house price?", "MSE, RMSE, R²"],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Supervised Learning?</b>
        <br><br>Supervised learning is <b>learning with labeled examples</b>. You show the model inputs AND the correct answers, and it learns the pattern connecting them.
        <br><br><b>Analogy:</b> Like a student with a teacher. "Here's a math problem, here's the answer. Now learn to solve similar problems."
        <br><br><b>The two types:</b>
        <br><br><b>📊 Classification:</b> Predicting a <b>category</b>
        <br>• Output: Yes/No, A/B/C, Cat/Dog
        <br>• Question: "Which group does this belong to?"
        <br>• Examples: Spam detection, disease diagnosis, churn prediction
        <br><br><b>📈 Regression:</b> Predicting a <b>number</b>
        <br>• Output: Continuous value ($350,000, 72.5°F)
        <br>• Question: "How much? How many?"
        <br>• Examples: House prices, temperature, sales forecasting
        <br><br><b>How to tell them apart:</b>
        <br>• Can you count the possible outputs? → Classification
        <br>• Are outputs on a continuous scale? → Regression
        </div>
        <div class="math-box">
        <b>📐 Classification vs Regression — Examples:</b>
        <br><br><b>Classification:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Input: Customer data
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Output: Churn = <b>Yes</b> or <b>No</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;(2 possible outputs)
        <br><br><b>Regression:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Input: House features
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Output: Price = <b>$347,500</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;(infinite possible outputs)
        <br><br><b>Tricky case:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;"Predict customer rating (1-5 stars)"
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Could be classification (5 categories)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;OR regression (treat as continuous)
        </div>""",
        code_str='''from sklearn.linear_model import LogisticRegression, LinearRegression

# Classification: Predict category
clf = LogisticRegression()
clf.fit(X_train, y_train_labels)  # y = 0 or 1
predictions = clf.predict(X_test)  # Returns: [0, 1, 1, 0...]

# Regression: Predict number
reg = LinearRegression()
reg.fit(X_train, y_train_values)  # y = continuous
predictions = reg.predict(X_test)  # Returns: [150000, 225000...]''',
        output_func=show_class_vs_reg,
        concept_title="📊 Classification vs Regression",
        output_title="Side-by-Side"
    )

    # Row 2: Logistic Regression
    def show_logistic():
        z_vals = np.linspace(-8, 8, 200)
        sigmoid = 1 / (1 + np.exp(-z_vals))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z_vals, y=sigmoid, mode='lines', line=dict(color='#22d3a7', width=3), name='Sigmoid'))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#f5b731", annotation_text="Threshold: 0.5")
        fig.add_vrect(x0=-8, x1=0, fillcolor="rgba(244,93,109,0.06)", line_width=0)
        fig.add_vrect(x0=0, x1=8, fillcolor="rgba(34,211,167,0.06)", line_width=0)
        fig.add_annotation(x=-4, y=0.8, text="Predict: NO", font=dict(color="#f45d6d", size=12), showarrow=False)
        fig.add_annotation(x=4, y=0.8, text="Predict: YES", font=dict(color="#22d3a7", size=12), showarrow=False)
        fig.update_layout(height=250, title="Sigmoid Function", xaxis_title="Linear score (z)", yaxis_title="Probability", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Logistic Regression?</b>
        <br><br>Despite the name, logistic regression is for <b>classification</b>, not regression! It predicts the <b>probability</b> of belonging to a category.
        <br><br><b>How it works:</b>
        <br>1. Calculate a linear score: z = β₀ + β₁x₁ + β₂x₂ + ...
        <br>2. Squash it to [0,1] using the sigmoid function
        <br>3. If probability > 0.5, predict "Yes"; otherwise "No"
        <br><br><b>The sigmoid function:</b>
        <br>• Takes any number (-∞ to +∞)
        <br>• Outputs a probability (0 to 1)
        <br>• S-shaped curve — extreme inputs → extreme probabilities
        <br><br><b>Why not just use linear regression?</b>
        <br>Linear regression can predict values like -0.3 or 1.5 — but probabilities must be between 0 and 1! Sigmoid fixes this.
        <br><br><b>Interpreting coefficients:</b>
        <br>• Positive β → increases probability
        <br>• Negative β → decreases probability
        <br>• Larger |β| → stronger effect
        </div>
        <div class="pros-cons-box">
        <b>⚖️ Pros & Cons</b>
        <br><br><span class="pros">✅ Pros:</span>
        <br>• Outputs probabilities (not just classes)
        <br>• Coefficients are interpretable
        <br>• Fast training, works well with small data
        <br>• Less prone to overfitting than complex models
        <br><br><span class="cons">❌ Cons:</span>
        <br>• Assumes linear decision boundary
        <br>• Can't capture complex non-linear patterns
        <br>• Sensitive to outliers
        <br>• Requires feature scaling for regularization
        </div>
        <div class="warn-box">
        <b>⚠️ Avoiding Overfitting & Underfitting</b>
        <br><br><b>Overfitting signs:</b> High train accuracy, low test accuracy
        <br><b>Underfitting signs:</b> Both accuracies low (model too simple)
        <br><br><b>Key Hyperparameters:</b>
        <br>• <code>C</code>: Inverse regularization strength (smaller C = more regularization)
        <br>• <code>penalty</code>: 'l1' (Lasso) or 'l2' (Ridge) regularization
        <br><br><b>Best Practice:</b> Start with C=1.0, use cross-validation. If overfitting, decrease C. If underfitting, increase C or add polynomial features.
        </div>
        <div class="math-box">
        <b>📐 Logistic Regression — Step by Step:</b>
        <br><br><b>Given:</b> β₀ = -2, β₁ = 0.05 (tenure), β₂ = 0.03 (charges)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Customer: tenure=12 months, charges=$80
        <br><br><b>Step 1:</b> Calculate linear score
        <br>&nbsp;&nbsp;&nbsp;&nbsp;z = -2 + 0.05×12 + 0.03×80
        <br>&nbsp;&nbsp;&nbsp;&nbsp;z = -2 + 0.6 + 2.4 = <b>1.0</b>
        <br><br><b>Step 2:</b> Apply sigmoid
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P = 1 / (1 + e⁻¹)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P = 1 / (1 + 0.37) = 1/1.37 = <b>0.73 (73%)</b>
        <br><br><b>Step 3:</b> Classify (threshold = 0.5)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;0.73 > 0.5 → Predict: <b>CHURN</b>
        <br><br>🧠 This customer has a 73% chance of churning!
        </div>
        <div class="insight-box">💡 <b>Business use:</b> Rank customers by churn probability. Focus retention efforts on the top 20% highest risk!</div>""",
        code_str='''from sklearn.linear_model import LogisticRegression
import numpy as np

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
print(f"P(churn): {probabilities[0][1]:.2%}")

# Sigmoid function manually
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = -0.2
prob = sigmoid(z)
print(f"P = {prob:.2f}")  # 0.45''',
        output_func=show_logistic,
        concept_title="📊 Logistic Regression",
        output_title="The Sigmoid Curve"
    )

    # Row 3: Decision Trees Quick Overview (detailed version in dedicated module)
    def show_tree_quick():
        st.markdown("#### 🌳 Decision Tree — Quick Overview")
        
        # Simple tree visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0.5], y=[0.85], mode='markers+text',
            marker=dict(size=55, color='#7c6aff', symbol='square'),
            text=["tenure < 24?"], textposition="middle center", textfont=dict(size=10, color='white'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.25], y=[0.45], mode='markers+text',
            marker=dict(size=45, color='#f5b731', symbol='square'),
            text=["charges > 65?"], textposition="middle center", textfont=dict(size=9, color='white'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.75], y=[0.45], mode='markers+text',
            marker=dict(size=45, color='#22d3a7', symbol='square'),
            text=["→ STAY"], textposition="middle center", textfont=dict(size=9, color='white'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.1], y=[0.15], mode='markers+text',
            marker=dict(size=40, color='#22d3a7', symbol='square'),
            text=["→ STAY"], textposition="middle center", textfont=dict(size=8, color='white'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.4], y=[0.15], mode='markers+text',
            marker=dict(size=40, color='#f45d6d', symbol='square'),
            text=["→ CHURN"], textposition="middle center", textfont=dict(size=8, color='white'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.5, 0.25], y=[0.8, 0.5], mode='lines', line=dict(color='#8892b0', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.5, 0.75], y=[0.8, 0.5], mode='lines', line=dict(color='#8892b0', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.25, 0.1], y=[0.4, 0.2], mode='lines', line=dict(color='#8892b0', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[0.25, 0.4], y=[0.4, 0.2], mode='lines', line=dict(color='#8892b0', width=2), showlegend=False))
        fig.update_layout(height=280, xaxis=dict(visible=False, range=[0,1]), yaxis=dict(visible=False, range=[0,1]),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=20,l=20,r=20),
                         title="Churn Prediction Tree")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("""<div class="insight-box">
        🎯 <b>Tree learned:</b> IF tenure < 24 AND charges > 65 → CHURN
        <br><br>📚 <b>For detailed Gini/Entropy math and step-by-step tree building, see the "🌳 Decision Trees" module!</b>
        </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is a Decision Tree?</b>
        <br><br>A decision tree is a <b>flowchart of yes/no questions</b> that leads to a prediction. It's one of the most interpretable ML models.
        <br><br><b>How it works:</b>
        <br>1. Start at the root with a question
        <br>2. Follow YES or NO branch based on answer
        <br>3. Repeat until you reach a leaf (prediction)
        <br><br><b>Key concept — Gini Impurity:</b>
        <br>• Measures how "mixed" a node is
        <br>• Gini = 0 → pure (all same class)
        <br>• Gini = 0.5 → maximum impurity (50-50)
        <br>• Trees split to minimize Gini
        <br><br><b>Pros:</b> Interpretable, no scaling needed
        <br><b>Cons:</b> Prone to overfitting
        </div>
        <div class="warn-box">
        📚 <b>Want the full math?</b> The "🌳 Decision Trees" module has step-by-step Gini & Entropy calculations, interactive tree building, and Information Gain examples!
        </div>""",
        code_str='''from sklearn.tree import DecisionTreeClassifier

# Create and train tree
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X_train, y_train)

# Make prediction
new_customer = [[8, 75]]  # tenure=8, charges=75
prediction = tree.predict(new_customer)
print(f"Prediction: {'Churn' if prediction[0] else 'Stay'}")''',
        output_func=show_tree_quick,
        concept_title="🌳 Decision Trees",
        output_title="Quick Overview"
    )

    # Row 4: Random Forest - Step by Step Visual
    def show_forest():
        st.markdown("#### 🌲 How Random Forest Works")
        
        # Step selector for Random Forest
        rf_step = st.radio("Select View:", ["View 1: Single Tree Problem", "View 2: Multiple Trees", "View 3: Voting Process", "View 4: Final Ensemble"], horizontal=True)
        
        np.random.seed(42)
        
        if rf_step == "View 1: Single Tree Problem":
            # Show why single tree is problematic
            st.markdown("**Problem: A single tree is unstable and overfits**")
            
            # Simulate different trees from slightly different data
            trees_data = [
                {"split1": "tenure<20", "split2": "charges>70", "pred": "Churn"},
                {"split1": "tenure<25", "split2": "charges>60", "pred": "Churn"},
                {"split1": "charges>65", "split2": "tenure<22", "pred": "Churn"},
            ]
            
            col1, col2, col3 = st.columns(3)
            
            # Simulate different trees from slightly different data
            trees_data = [
                {"split1": "tenure<20", "split2": "charges>70", "pred": "Churn"},
                {"split1": "tenure<25", "split2": "charges>60", "pred": "Churn"},
                {"split1": "charges>65", "split2": "tenure<22", "pred": "Churn"},
            ]
            
            col1, col2, col3 = st.columns(3)
            for i, (col, tree) in enumerate(zip([col1, col2, col3], trees_data)):
                with col:
                    st.markdown(f"**Tree from Sample {i+1}**")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[0.5], y=[0.85], mode='markers+text',
                        marker=dict(size=40, color='#7c6aff', symbol='square'),
                        text=[tree["split1"]], textposition="middle center", textfont=dict(size=9, color='white'), showlegend=False))
                    fig.add_trace(go.Scatter(x=[0.3], y=[0.45], mode='markers+text',
                        marker=dict(size=35, color='#f5b731', symbol='square'),
                        text=[tree["split2"]], textposition="middle center", textfont=dict(size=8, color='white'), showlegend=False))
                    fig.add_trace(go.Scatter(x=[0.7], y=[0.45], mode='markers+text',
                        marker=dict(size=35, color='#22d3a7', symbol='square'),
                        text=["Stay"], textposition="middle center", textfont=dict(size=8, color='white'), showlegend=False))
                    fig.add_trace(go.Scatter(x=[0.5, 0.3], y=[0.8, 0.5], mode='lines', line=dict(color='#8892b0', width=1.5), showlegend=False))
                    fig.add_trace(go.Scatter(x=[0.5, 0.7], y=[0.8, 0.5], mode='lines', line=dict(color='#8892b0', width=1.5), showlegend=False))
                    fig.update_layout(height=200, xaxis=dict(visible=False, range=[0,1]), yaxis=dict(visible=False, range=[0,1]),
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            st.markdown("""<div class="warn-box">
            ⚠️ <b>Problem:</b> Each tree learned different rules from slightly different data!
            <br>• Tree 1: tenure<20 AND charges>70
            <br>• Tree 2: tenure<25 AND charges>60  
            <br>• Tree 3: charges>65 AND tenure<22
            <br><br>Which one is right? <b>None of them alone is reliable!</b>
            </div>""", unsafe_allow_html=True)
            
        elif rf_step == "View 2: Multiple Trees":
            # Show bootstrap sampling and feature randomness
            st.markdown("**Solution: Train many trees on random subsets**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎲 Bootstrap Sampling (Bagging)**")
                st.markdown("""
                Each tree sees a different random sample of the data:
                """)
                sample_data = pd.DataFrame({
                    "Tree": ["Tree 1", "Tree 2", "Tree 3", "Tree 4", "Tree 5"],
                    "Samples Used": ["Rows: 1,3,3,5,7,7,9", "Rows: 2,2,4,6,8,8,10", "Rows: 1,1,4,5,6,9,10", "Rows: 2,3,5,7,7,8,9", "Rows: 1,4,4,6,8,9,10"],
                    "% Unique": ["71%", "71%", "86%", "86%", "86%"]
                })
                st.dataframe(sample_data, use_container_width=True, hide_index=True, height=200)
                
            with col2:
                st.markdown("**🎯 Random Feature Selection**")
                st.markdown("""
                At each split, only consider a random subset of features:
                """)
                feature_data = pd.DataFrame({
                    "Split": ["Split 1", "Split 2", "Split 3", "Split 4"],
                    "Features Available": ["tenure, charges", "charges, tickets", "tenure, tickets", "tenure, charges"],
                    "Best Split": ["tenure<24", "charges>70", "tickets>3", "charges>65"]
                })
                st.dataframe(feature_data, use_container_width=True, hide_index=True, height=200)
            
            st.markdown("""<div class="insight-box">
            💡 <b>Key Insight:</b> Each tree is trained on:
            <br>• ~63% of the data (bootstrap sample)
            <br>• √n features at each split (typically)
            <br><br>This makes trees <b>different from each other</b> — they overfit in different ways!
            </div>""", unsafe_allow_html=True)
            
        elif rf_step == "View 3: Voting Process":
            # Show how trees vote
            st.markdown("**Each tree votes, majority wins!**")
            
            # Simulate 5 trees voting
            trees = [
                {"id": 1, "prediction": "Churn", "confidence": 0.72, "color": "#f45d6d"},
                {"id": 2, "prediction": "Stay", "confidence": 0.65, "color": "#22d3a7"},
                {"id": 3, "prediction": "Churn", "confidence": 0.81, "color": "#f45d6d"},
                {"id": 4, "prediction": "Churn", "confidence": 0.58, "color": "#f45d6d"},
                {"id": 5, "prediction": "Stay", "confidence": 0.77, "color": "#22d3a7"},
            ]
            
            # Visual voting
            fig = go.Figure()
            for i, tree in enumerate(trees):
                fig.add_trace(go.Scatter(
                    x=[i*0.2 + 0.1], y=[0.7], mode='markers+text',
                    marker=dict(size=50, color=tree["color"], symbol='square'),
                    text=[f"🌳{tree['id']}<br>{tree['prediction']}<br>{tree['confidence']:.0%}"],
                    textposition="middle center", textfont=dict(size=9, color='white'), showlegend=False
                ))
                # Arrow down
                fig.add_annotation(x=i*0.2 + 0.1, y=0.45, text="⬇️", font=dict(size=20), showarrow=False)
            
            # Vote count
            fig.add_trace(go.Scatter(
                x=[0.5], y=[0.2], mode='markers+text',
                marker=dict(size=80, color='#7c6aff', symbol='square'),
                text=["VOTES<br>Churn: 3<br>Stay: 2<br>→ CHURN"],
                textposition="middle center", textfont=dict(size=11, color='white'), showlegend=False
            ))
            
            fig.update_layout(height=350, xaxis=dict(visible=False, range=[0,1]), yaxis=dict(visible=False, range=[0,1]),
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            st.markdown("""<div class="math-box">
            <b>📐 Voting Calculation:</b>
            <br><br><b>Classification (Majority Vote):</b>
            <br>&nbsp;&nbsp;&nbsp;&nbsp;Tree 1: Churn ✓ | Tree 2: Stay | Tree 3: Churn ✓
            <br>&nbsp;&nbsp;&nbsp;&nbsp;Tree 4: Churn ✓ | Tree 5: Stay
            <br>&nbsp;&nbsp;&nbsp;&nbsp;Churn wins 3-2 → <b>Final: CHURN</b>
            <br><br><b>Probability (Average):</b>
            <br>&nbsp;&nbsp;&nbsp;&nbsp;P(Churn) = (0.72 + 0.35 + 0.81 + 0.58 + 0.23) / 5 = <b>0.54</b>
            </div>""", unsafe_allow_html=True)
            
        else:  # Final Ensemble
            # Show error reduction with more trees
            n_trees_range = list(range(1, 51))
            true_val = 75
            np.random.seed(42)
            all_preds = np.random.normal(true_val, 15, 100)
            ensemble_errors = [abs(all_preds[:k].mean() - true_val) for k in n_trees_range]
            single_errors = [abs(p - true_val) for p in all_preds[:50]]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=n_trees_range, y=single_errors, mode='markers',
                    marker=dict(color='#f45d6d', size=6, opacity=0.5), name='Single Tree Error'))
                fig.add_trace(go.Scatter(x=n_trees_range, y=ensemble_errors, mode='lines',
                    line=dict(color='#22d3a7', width=3), name='Ensemble Error'))
                fig.add_hline(y=0, line_dash="dash", line_color="#f5b731")
                fig.update_layout(height=280, title="Error Reduction with More Trees", 
                                 xaxis_title="Number of Trees", yaxis_title="Prediction Error", **DL)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            with col2:
                st.markdown("**📊 Results**")
                st.metric("Single Tree Avg Error", f"{np.mean(single_errors):.1f}")
                st.metric("50-Tree Ensemble Error", f"{ensemble_errors[-1]:.1f}")
                st.metric("Error Reduction", f"{(1 - ensemble_errors[-1]/np.mean(single_errors))*100:.0f}%")
            
            st.markdown("""<div class="insight-box">
            🎯 <b>The Magic of Ensembles:</b>
            <br>• Individual trees have high variance (scattered red dots)
            <br>• But their errors are <b>random</b> — some too high, some too low
            <br>• When averaged, random errors <b>cancel out</b>!
            <br>• Only the <b>true signal</b> remains (green line converges to 0 error)
            </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is a Random Forest?</b>
        <br><br>A Random Forest is <b>many decision trees voting together</b>. It's based on a powerful idea: a crowd of "okay" predictors beats one "great" predictor.
        <br><br><b>The Algorithm:</b>
        <br>1. <b>Bootstrap:</b> Create N random samples of data (with replacement)
        <br>2. <b>Train:</b> Build a decision tree on each sample
        <br>3. <b>Randomize:</b> At each split, only consider √features
        <br>4. <b>Aggregate:</b> Combine predictions (vote or average)
        <br><br><b>Why it works:</b>
        <br>• Each tree overfits differently (different data, features)
        <br>• Random errors cancel when averaged
        <br>• True signal remains!
        </div>
        <div class="pros-cons-box">
        <b>⚖️ Pros & Cons</b>
        <br><br><span class="pros">✅ Pros:</span>
        <br>• Much less overfitting than single tree
        <br>• Handles high-dimensional data well
        <br>• Provides feature importance rankings
        <br>• Works out-of-the-box with minimal tuning
        <br>• Robust to outliers and noise
        <br><br><span class="cons">❌ Cons:</span>
        <br>• Less interpretable than single tree
        <br>• Slower training (many trees)
        <br>• Large model size in memory
        <br>• Can still overfit with very deep trees
        </div>
        <div class="warn-box">
        <b>⚠️ Avoiding Overfitting & Underfitting</b>
        <br><br><b>Overfitting signs:</b> Train accuracy much higher than test
        <br><b>Underfitting signs:</b> Both accuracies low, trees too shallow
        <br><br><b>Key Hyperparameters:</b>
        <br>• <code>n_estimators</code>: More trees rarely hurts (100-500)
        <br>• <code>max_depth</code>: Limit depth to prevent overfit (5-15)
        <br>• <code>min_samples_leaf</code>: Increase to simplify (5-20)
        <br>• <code>max_features</code>: 'sqrt' for classification, 'log2' or 0.3 for regression
        <br><br><b>Best Practice:</b> Start with defaults, then tune max_depth and min_samples_leaf using cross-validation. More trees is almost always better (just slower).
        </div>
        <div class="math-box">
        <b>📐 Key Hyperparameters:</b>
        <br><br><b>n_estimators:</b> Number of trees (100-500 typical)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;More trees = more stable, but slower
        <br><br><b>max_depth:</b> How deep each tree grows
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Deeper = more complex, risk of overfit
        <br><br><b>max_features:</b> Features per split
        <br>&nbsp;&nbsp;&nbsp;&nbsp;√n for classification, n/3 for regression
        <br><br><b>min_samples_leaf:</b> Min samples in leaf
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Higher = simpler trees
        </div>""",
        code_str='''from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,    # 100 trees
    max_depth=5,         # limit tree depth
    max_features='sqrt', # √n features per split
    random_state=42
)
rf.fit(X_train, y_train)

# Predict with voting
predictions = rf.predict(X_test)

# Get probability (average across trees)
probabilities = rf.predict_proba(X_test)
print(f"P(Churn): {probabilities[0][1]:.2%}")

# Feature importance (which features matter?)
importance = rf.feature_importances_
for feat, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
    print(f"{feat}: {imp:.3f}")''',
        output_func=show_forest,
        concept_title="🌲 Random Forest",
        output_title="Ensemble Visualization"
    )

    iq([
        {"q": "Explain the difference between classification and regression.", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Classification:</b> Predicts a category (Yes/No, A/B/C). <b>Regression:</b> Predicts a continuous number. <b>Example:</b> 'Will customer churn?' = classification. 'How much will they spend?' = regression.",
         "t": "Give concrete examples for each."},
        {"q": "How does a Random Forest reduce overfitting compared to a single decision tree?", "d": "Medium", "c": ["Meta", "Google"],
         "a": "Each tree sees different data (bootstrap) and features (random subsets). They overfit in different ways. When averaged, the random errors cancel out, leaving only the true signal. It's the 'wisdom of crowds.'",
         "t": "Mention both sources of randomness: bootstrap sampling AND random feature subsets."},
    ])


# ═══════════════════════════════════════
# DECISION TREES (with Gini & Entropy Math)
# ═══════════════════════════════════════
elif module == "🌳 Decision Trees":
    st.markdown("# 🌳 Decision Trees — How Trees Make Decisions")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is a Decision Tree?</b> It's a <b>flowchart of yes/no questions</b> that leads to a prediction. 
    Trees are one of the most interpretable ML models — you can literally see why it made each decision.
    <br><br>🎯 <b>Key concepts:</b> How trees split data, Gini impurity, Entropy, Information Gain — the math behind every split.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Decision Tree Basics
    def show_tree_basics():
        st.markdown("#### 🌳 Step-by-Step Tree Building")
        
        # Fixed sample data for consistent visualization
        np.random.seed(42)
        
        # Create clear patterns: short tenure + high charges = churn
        tenure = np.array([5, 8, 10, 12, 15, 18, 20, 22, 6, 9, 11, 14, 16, 19, 21, 7, 13, 17, 4, 23,
                          30, 35, 40, 45, 50, 55, 60, 65, 32, 38, 42, 48, 52, 58, 62, 68, 33, 44, 56, 67,
                          8, 12, 15, 19, 22, 7, 11, 16, 20, 6, 14, 18, 9, 13, 17, 21, 5, 10, 23, 4,
                          28, 34, 39, 46, 51, 57, 63, 29, 36, 41, 47, 53, 59, 64, 31, 37, 43, 49, 54, 61,
                          3, 7, 11, 15, 19, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 26, 32, 38, 44, 48])
        
        charges = np.array([75, 80, 85, 70, 78, 82, 68, 72, 88, 76, 79, 83, 71, 77, 81, 74, 86, 69, 90, 73,
                           45, 50, 55, 40, 48, 52, 38, 42, 58, 46, 49, 53, 41, 47, 51, 44, 56, 39, 60, 43,
                           72, 78, 84, 70, 76, 82, 68, 74, 80, 86, 71, 77, 83, 69, 75, 81, 87, 73, 79, 85,
                           42, 48, 54, 40, 46, 52, 38, 44, 50, 56, 41, 47, 53, 39, 45, 51, 57, 43, 49, 55,
                           65, 70, 55, 60, 50, 45, 40, 35, 30, 48, 52, 58, 62, 44, 38, 42, 56, 34, 46, 36])
        
        churned = ((tenure < 24) & (charges > 65)).astype(int)
        
        step = st.radio("Select Step:", ["Step 1: All Data", "Step 2: First Split", "Step 3: Second Split", "Step 4: Final Tree"], horizontal=True, key="tree_step")
        
        if step == "Step 1: All Data":
            n_churn = churned.sum()
            n_stay = len(churned) - n_churn
            gini = 1 - (n_churn/len(churned))**2 - (n_stay/len(churned))**2
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(churned))
            col2.metric("Churned", n_churn, f"{n_churn/len(churned):.0%}")
            col3.metric("Gini Impurity", f"{gini:.3f}")
            
            fig = go.Figure()
            stay_mask = churned == 0
            churn_mask = churned == 1
            fig.add_trace(go.Scatter(x=tenure[stay_mask], y=charges[stay_mask], mode='markers',
                marker=dict(size=8, color='#22d3a7', opacity=0.7), name=f'Stay ({stay_mask.sum()})'))
            fig.add_trace(go.Scatter(x=tenure[churn_mask], y=charges[churn_mask], mode='markers',
                marker=dict(size=8, color='#f45d6d', opacity=0.7), name=f'Churn ({churn_mask.sum()})'))
            fig.update_layout(height=280, title=f"All {len(churned)} Customers — Gini: {gini:.3f}",
                             xaxis_title="Tenure (months)", yaxis_title="Monthly Charges ($)", **DL)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            st.markdown("""<div class="insight-box">
            🎯 <b>Goal:</b> Find the best question to split this mixed data into purer groups.
            <br>We'll try: "tenure < 24?" and "charges > 65?" to see which creates better separation.
            </div>""", unsafe_allow_html=True)
            
        elif step == "Step 2: First Split":
            left_mask = tenure < 24
            left_churn = churned[left_mask].sum()
            left_total = left_mask.sum()
            right_churn = churned[~left_mask].sum()
            right_total = (~left_mask).sum()
            
            gini_left = 1 - (left_churn/left_total)**2 - ((left_total-left_churn)/left_total)**2 if left_total > 0 else 0
            gini_right = 1 - (right_churn/right_total)**2 - ((right_total-right_churn)/right_total)**2 if right_total > 0 else 0
            
            # Create side-by-side: scatter plot + mini tree
            viz_col1, viz_col2 = st.columns([3, 2])
            
            with viz_col1:
                fig = go.Figure()
                left_stay = left_mask & (churned == 0)
                left_churn_mask = left_mask & (churned == 1)
                right_stay = (~left_mask) & (churned == 0)
                right_churn_mask = (~left_mask) & (churned == 1)
                
                fig.add_trace(go.Scatter(x=tenure[left_stay], y=charges[left_stay], mode='markers',
                    marker=dict(size=8, color='#22d3a7', opacity=0.7), name='Left-Stay'))
                fig.add_trace(go.Scatter(x=tenure[left_churn_mask], y=charges[left_churn_mask], mode='markers',
                    marker=dict(size=8, color='#f45d6d', opacity=0.7), name='Left-Churn'))
                fig.add_trace(go.Scatter(x=tenure[right_stay], y=charges[right_stay], mode='markers',
                    marker=dict(size=8, color='#22d3a7', opacity=0.7, symbol='diamond'), name='Right-Stay'))
                fig.add_trace(go.Scatter(x=tenure[right_churn_mask], y=charges[right_churn_mask], mode='markers',
                    marker=dict(size=8, color='#f45d6d', opacity=0.7, symbol='diamond'), name='Right-Churn'))
                
                fig.add_vline(x=24, line_dash="dash", line_color="#f5b731", line_width=3, 
                             annotation_text="Split: tenure=24", annotation_position="top")
                fig.add_vrect(x0=0, x1=24, fillcolor="rgba(244,93,109,0.1)", line_width=0)
                fig.add_vrect(x0=24, x1=75, fillcolor="rgba(34,211,167,0.1)", line_width=0)
                
                fig.update_layout(height=260, title="Data Split Visualization",
                                 xaxis_title="Tenure (months)", yaxis_title="Monthly Charges ($)", **DL)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            with viz_col2:
                # Mini tree diagram
                tree_fig = go.Figure()
                
                # Root node
                tree_fig.add_shape(type="rect", x0=0.3, y0=0.75, x1=0.7, y1=0.95,
                    fillcolor='#7c6aff', line=dict(color='#7c6aff', width=2))
                tree_fig.add_annotation(x=0.5, y=0.85, text="<b>tenure < 24?</b><br><span style='font-size:9px'>n=100</span>",
                    font=dict(size=10, color='white'), showarrow=False)
                
                # Left child
                tree_fig.add_shape(type="rect", x0=0.05, y0=0.25, x1=0.4, y1=0.45,
                    fillcolor='#f5b731', line=dict(color='#f5b731', width=2))
                tree_fig.add_annotation(x=0.225, y=0.35, text=f"<b>YES</b><br><span style='font-size:9px'>n={left_total} | G={gini_left:.2f}</span>",
                    font=dict(size=9, color='white'), showarrow=False)
                
                # Right child
                tree_fig.add_shape(type="rect", x0=0.6, y0=0.25, x1=0.95, y1=0.45,
                    fillcolor='#22d3a7', line=dict(color='#22d3a7', width=2))
                tree_fig.add_annotation(x=0.775, y=0.35, text=f"<b>NO</b><br><span style='font-size:9px'>n={right_total} | G={gini_right:.2f}</span>",
                    font=dict(size=9, color='white'), showarrow=False)
                
                # Edges
                tree_fig.add_trace(go.Scatter(x=[0.4, 0.225, 0.225], y=[0.75, 0.55, 0.45], mode='lines',
                    line=dict(color='#f5b731', width=3), showlegend=False))
                tree_fig.add_trace(go.Scatter(x=[0.6, 0.775, 0.775], y=[0.75, 0.55, 0.45], mode='lines',
                    line=dict(color='#22d3a7', width=3), showlegend=False))
                
                tree_fig.update_layout(height=260, xaxis=dict(visible=False, range=[0,1]),
                    yaxis=dict(visible=False, range=[0,1]), paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10), title="Tree Structure")
                st.plotly_chart(tree_fig, use_container_width=True, config={"displayModeBar": False})
            
            weighted_gini = (left_total/len(churned))*gini_left + (right_total/len(churned))*gini_right
            st.markdown(f"""<div class="math-box">
            <b>📐 Why tenure < 24?</b>
            <br>We tried many splits. This one reduced impurity the most!
            <br>• Weighted Gini = ({left_total}/{len(churned)})×{gini_left:.3f} + ({right_total}/{len(churned)})×{gini_right:.3f} = <b>{weighted_gini:.3f}</b>
            </div>""", unsafe_allow_html=True)
            
        elif step == "Step 3: Second Split":
            left_mask = tenure < 24
            left_high_charges = left_mask & (charges > 65)
            left_low_charges = left_mask & (charges <= 65)
            
            ll_churn = churned[left_low_charges].sum()
            ll_total = left_low_charges.sum() if left_low_charges.sum() > 0 else 1
            lh_churn = churned[left_high_charges].sum()
            lh_total = left_high_charges.sum() if left_high_charges.sum() > 0 else 1
            right_total = (~left_mask).sum()
            
            # Create side-by-side: scatter plot + mini tree
            viz_col1, viz_col2 = st.columns([3, 2])
            
            with viz_col1:
                fig = go.Figure()
                for mask, color, symbol in [
                    (left_low_charges & (churned == 0), '#22d3a7', 'circle'),
                    (left_low_charges & (churned == 1), '#f45d6d', 'circle'),
                    (left_high_charges & (churned == 0), '#22d3a7', 'square'),
                    (left_high_charges & (churned == 1), '#f45d6d', 'square'),
                    ((~left_mask) & (churned == 0), '#22d3a7', 'diamond'),
                    ((~left_mask) & (churned == 1), '#f45d6d', 'diamond'),
                ]:
                    if mask.sum() > 0:
                        fig.add_trace(go.Scatter(x=tenure[mask], y=charges[mask], mode='markers',
                            marker=dict(size=8, color=color, opacity=0.7, symbol=symbol), showlegend=False))
                
                fig.add_vline(x=24, line_dash="dash", line_color="#f5b731", line_width=2)
                fig.add_hline(y=65, line_dash="dash", line_color="#f5b731", line_width=2)
                fig.add_shape(type="rect", x0=0, x1=24, y0=65, y1=100, fillcolor="rgba(244,93,109,0.15)", line_width=0)
                fig.add_shape(type="rect", x0=0, x1=24, y0=0, y1=65, fillcolor="rgba(34,211,167,0.1)", line_width=0)
                fig.add_shape(type="rect", x0=24, x1=75, y0=0, y1=100, fillcolor="rgba(34,211,167,0.1)", line_width=0)
                fig.add_annotation(x=12, y=82, text="🔴 CHURN", font=dict(size=12, color='#f45d6d'), showarrow=False)
                fig.add_annotation(x=12, y=45, text="🟢 STAY", font=dict(size=12, color='#22d3a7'), showarrow=False)
                fig.add_annotation(x=50, y=50, text="🟢 STAY", font=dict(size=12, color='#22d3a7'), showarrow=False)
                
                fig.update_layout(height=280, title="Second Split: charges > 65?",
                                 xaxis_title="Tenure (months)", yaxis_title="Monthly Charges ($)", **DL)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            with viz_col2:
                # Full tree diagram
                tree_fig = go.Figure()
                
                # Root node
                tree_fig.add_shape(type="rect", x0=0.35, y0=0.82, x1=0.65, y1=0.98,
                    fillcolor='#7c6aff', line=dict(color='#7c6aff', width=2))
                tree_fig.add_annotation(x=0.5, y=0.9, text="<b>tenure<24?</b>",
                    font=dict(size=9, color='white'), showarrow=False)
                
                # Level 2 nodes
                tree_fig.add_shape(type="rect", x0=0.1, y0=0.52, x1=0.4, y1=0.68,
                    fillcolor='#f5b731', line=dict(color='#f5b731', width=2))
                tree_fig.add_annotation(x=0.25, y=0.6, text="<b>charges>65?</b>",
                    font=dict(size=8, color='white'), showarrow=False)
                
                tree_fig.add_shape(type="rect", x0=0.6, y0=0.52, x1=0.9, y1=0.68,
                    fillcolor='#22d3a7', line=dict(color='#22d3a7', width=2))
                tree_fig.add_annotation(x=0.75, y=0.6, text=f"<b>STAY</b><br><span style='font-size:8px'>n={right_total}</span>",
                    font=dict(size=8, color='white'), showarrow=False)
                
                # Level 3 leaves
                tree_fig.add_shape(type="rect", x0=0.02, y0=0.15, x1=0.22, y1=0.35,
                    fillcolor='#22d3a7', line=dict(color='#22d3a7', width=2))
                tree_fig.add_annotation(x=0.12, y=0.25, text=f"<b>STAY</b><br><span style='font-size:8px'>n={ll_total}</span>",
                    font=dict(size=8, color='white'), showarrow=False)
                
                tree_fig.add_shape(type="rect", x0=0.28, y0=0.15, x1=0.48, y1=0.35,
                    fillcolor='#f45d6d', line=dict(color='#f45d6d', width=2))
                tree_fig.add_annotation(x=0.38, y=0.25, text=f"<b>CHURN</b><br><span style='font-size:8px'>n={lh_total}</span>",
                    font=dict(size=8, color='white'), showarrow=False)
                
                # Edges
                tree_fig.add_trace(go.Scatter(x=[0.42, 0.25, 0.25], y=[0.82, 0.72, 0.68], mode='lines',
                    line=dict(color='#f5b731', width=2), showlegend=False))
                tree_fig.add_trace(go.Scatter(x=[0.58, 0.75, 0.75], y=[0.82, 0.72, 0.68], mode='lines',
                    line=dict(color='#22d3a7', width=2), showlegend=False))
                tree_fig.add_trace(go.Scatter(x=[0.18, 0.12, 0.12], y=[0.52, 0.42, 0.35], mode='lines',
                    line=dict(color='#22d3a7', width=2), showlegend=False))
                tree_fig.add_trace(go.Scatter(x=[0.32, 0.38, 0.38], y=[0.52, 0.42, 0.35], mode='lines',
                    line=dict(color='#f45d6d', width=2), showlegend=False))
                
                # Edge labels
                tree_fig.add_annotation(x=0.32, y=0.76, text="YES", font=dict(size=8, color='#f5b731'), showarrow=False)
                tree_fig.add_annotation(x=0.68, y=0.76, text="NO", font=dict(size=8, color='#22d3a7'), showarrow=False)
                tree_fig.add_annotation(x=0.1, y=0.44, text="NO", font=dict(size=7, color='#22d3a7'), showarrow=False)
                tree_fig.add_annotation(x=0.4, y=0.44, text="YES", font=dict(size=7, color='#f45d6d'), showarrow=False)
                
                tree_fig.update_layout(height=280, xaxis=dict(visible=False, range=[0,1]),
                    yaxis=dict(visible=False, range=[0,1]), paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10), title="Complete Tree")
                st.plotly_chart(tree_fig, use_container_width=True, config={"displayModeBar": False})
            
            st.markdown("""<div class="insight-box">
            🎯 <b>Pattern Found:</b> Short tenure + high charges = churners!
            <br>The tree learned: <code>IF tenure < 24 AND charges > 65 THEN Churn</code>
            </div>""", unsafe_allow_html=True)
            
        else:  # Final Tree - Professional Interactive Visualization
            # Interactive prediction path
            st.markdown("**🎯 Try a prediction — click a path:**")
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                test_tenure = st.slider("Customer Tenure (months)", 1, 72, 8, key="tree_tenure")
            with pred_col2:
                test_charges = st.slider("Monthly Charges ($)", 20, 100, 75, key="tree_charges")
            
            # Determine prediction path
            path_nodes = [0]  # Start at root
            if test_tenure < 24:
                path_nodes.append(1)  # Go left
                if test_charges > 65:
                    path_nodes.append(3)  # Churn leaf
                    prediction = "🔴 CHURN"
                    pred_color = "#f45d6d"
                else:
                    path_nodes.append(4)  # Stay leaf
                    prediction = "🟢 STAY"
                    pred_color = "#22d3a7"
            else:
                path_nodes.append(2)  # Go right - Stay leaf
                prediction = "🟢 STAY"
                pred_color = "#22d3a7"
            
            fig = go.Figure()
            
            # Node definitions: (x, y, width, height, color, label, samples, gini, is_leaf)
            nodes = [
                (0.5, 0.88, 0.22, 0.12, '#7c6aff', 'tenure < 24?', 100, 0.420, False),   # Root
                (0.25, 0.55, 0.20, 0.12, '#f5b731', 'charges > 65?', 60, 0.480, False),  # Left child
                (0.75, 0.55, 0.18, 0.12, '#22d3a7', 'STAY', 40, 0.000, True),            # Right child (leaf)
                (0.12, 0.22, 0.16, 0.11, '#f45d6d', 'CHURN', 40, 0.000, True),           # Left-left leaf
                (0.38, 0.22, 0.16, 0.11, '#22d3a7', 'STAY', 20, 0.000, True),            # Left-right leaf
            ]
            
            # Draw edges first (so they appear behind nodes)
            edges = [(0, 1, "YES"), (0, 2, "NO"), (1, 3, "YES"), (1, 4, "NO")]
            for parent_idx, child_idx, label in edges:
                px, py = nodes[parent_idx][0], nodes[parent_idx][1] - nodes[parent_idx][3]/2
                cx, cy = nodes[child_idx][0], nodes[child_idx][1] + nodes[child_idx][3]/2
                
                # Check if this edge is on the prediction path
                is_active = parent_idx in path_nodes and child_idx in path_nodes
                edge_color = pred_color if is_active else '#4a5568'
                edge_width = 4 if is_active else 2
                
                # Draw curved edge using bezier-like path
                mid_y = (py + cy) / 2
                fig.add_trace(go.Scatter(
                    x=[px, px, cx, cx], y=[py, mid_y, mid_y, cy],
                    mode='lines', line=dict(color=edge_color, width=edge_width, shape='spline'),
                    hoverinfo='skip', showlegend=False
                ))
                
                # Edge label
                label_x = (px + cx) / 2 + (0.03 if cx > px else -0.03)
                label_y = mid_y + 0.02
                fig.add_annotation(x=label_x, y=label_y, text=f"<b>{label}</b>",
                    font=dict(size=11, color=edge_color), showarrow=False, bgcolor='rgba(26,32,44,0.8)')
            
            # Draw nodes as rounded rectangles using shapes
            for idx, (x, y, w, h, color, label, samples, gini, is_leaf) in enumerate(nodes):
                is_active = idx in path_nodes
                border_color = pred_color if is_active else color
                border_width = 4 if is_active else 2
                fill_opacity = 1.0 if is_active else 0.85
                
                # Add rounded rectangle shape
                fig.add_shape(type="rect", x0=x-w/2, y0=y-h/2, x1=x+w/2, y1=y+h/2,
                    fillcolor=color, line=dict(color=border_color, width=border_width),
                    opacity=fill_opacity, layer='above')
                
                # Node content
                if is_leaf:
                    node_text = f"<b>{label}</b><br><span style='font-size:10px'>n={samples}</span>"
                else:
                    node_text = f"<b>{label}</b><br><span style='font-size:9px'>n={samples} | Gini={gini:.2f}</span>"
                
                # Add invisible scatter for hover
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(size=50, color='rgba(0,0,0,0)'),
                    hovertemplate=f"<b>{'Leaf: ' if is_leaf else ''}{label}</b><br>Samples: {samples}<br>Gini: {gini:.3f}<extra></extra>",
                    showlegend=False
                ))
                
                # Add text annotation
                fig.add_annotation(x=x, y=y, text=node_text, font=dict(size=11, color='white'),
                    showarrow=False, align='center')
            
            # Add prediction result banner
            fig.add_annotation(x=0.5, y=0.02, 
                text=f"<b>Prediction: {prediction}</b>",
                font=dict(size=14, color=pred_color), showarrow=False,
                bgcolor='rgba(26,32,44,0.9)', bordercolor=pred_color, borderwidth=2, borderpad=6)
            
            fig.update_layout(
                height=420,
                xaxis=dict(visible=False, range=[-0.05, 1.05], fixedrange=True),
                yaxis=dict(visible=False, range=[-0.05, 1.05], fixedrange=True, scaleanchor='x'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=30, l=10, r=10),
                hoverlabel=dict(bgcolor='#1a202c', font_size=12, font_color='white')
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            # Decision path explanation
            st.markdown(f"""<div class="math-box">
            <b>📐 Decision Path for Customer (tenure={test_tenure}, charges=${test_charges}):</b>
            <br><br>1️⃣ ROOT: "tenure < 24?" → {test_tenure} < 24 → <b>{"YES ✓" if test_tenure < 24 else "NO ✗"}</b>
            {"<br>2️⃣ Next: 'charges > 65?' → " + str(test_charges) + " > 65 → <b>" + ("YES ✓" if test_charges > 65 else "NO ✗") + "</b>" if test_tenure < 24 else ""}
            <br>{"3️⃣" if test_tenure < 24 else "2️⃣"} Leaf: <b style="color:{pred_color}">{prediction}</b>
            </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is a Decision Tree?</b>
        <br><br>A decision tree is a <b>flowchart of yes/no questions</b> that leads to a prediction. It's one of the most interpretable ML models.
        <br><br><b>How it builds the tree:</b>
        <br>1. <b>Start:</b> All data at the root node
        <br>2. <b>Find best split:</b> Try every feature & threshold
        <br>3. <b>Split:</b> Divide data into two branches
        <br>4. <b>Repeat:</b> For each branch, find next best split
        <br>5. <b>Stop:</b> When nodes are pure or max depth reached
        <br><br><b>Measuring "purity" — Gini Impurity:</b>
        <br>• Gini = 1 - Σ(pᵢ²) for each class
        <br>• Gini = 0 → perfectly pure (all one class)
        <br>• Gini = 0.5 → maximum impurity (50-50)
        </div>
        <div class="pros-cons-box">
        <b>⚖️ Pros & Cons</b>
        <br><br><span class="pros">✅ Pros:</span>
        <br>• Highly interpretable — can explain every decision
        <br>• No feature scaling needed
        <br>• Handles both numerical & categorical data
        <br>• Captures non-linear relationships
        <br>• Fast training and prediction
        <br><br><span class="cons">❌ Cons:</span>
        <br>• Very prone to overfitting (memorizes noise)
        <br>• Unstable — small data changes → completely different tree
        <br>• Biased toward features with more levels
        <br>• Can't extrapolate beyond training data range
        </div>
        <div class="warn-box">
        <b>⚠️ Avoiding Overfitting & Underfitting</b>
        <br><br><b>Signs of Overfitting:</b> Train accuracy >> Test accuracy, very deep tree
        <br><b>Signs of Underfitting:</b> Both train & test accuracy low, tree too shallow
        <br><br><b>Key Hyperparameters to Tune:</b>
        <br>• <code>max_depth</code>: Limit tree depth (start with 3-5)
        <br>• <code>min_samples_split</code>: Min samples to split a node (try 5-20)
        <br>• <code>min_samples_leaf</code>: Min samples in leaf (try 5-10)
        <br>• <code>max_features</code>: Limit features per split
        <br><br><b>Best Practice:</b> Use cross-validation to find optimal depth. Start shallow, increase until validation score stops improving.
        </div>
        <div class="insight-box">💡 <b>Use the step selector</b> on the right to see how the tree is built!</div>""",
        code_str='''from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create and train tree
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X_train, y_train)

# Make prediction for new customer
new_customer = [[8, 75]]  # tenure=8, charges=75
prediction = tree.predict(new_customer)
print(f"Prediction: {'Churn' if prediction[0] else 'Stay'}")''',
        output_func=show_tree_basics,
        concept_title="🌳 Decision Tree Basics",
        output_title="Interactive Tree Builder"
    )

    # Row 2: Gini Impurity - Full Math
    def show_gini_math():
        st.markdown("#### 📊 Step-by-Step Gini Calculation")
        
        # Interactive example
        st.markdown("**Example: Customer Churn Dataset**")
        col1, col2 = st.columns(2)
        with col1:
            churn = st.slider("Churned customers", 0, 100, 30, key="gini_churn")
        with col2:
            stay = 100 - churn
            st.metric("Stayed customers", stay)
        
        total = churn + stay
        p_churn = churn / total if total > 0 else 0
        p_stay = stay / total if total > 0 else 0
        gini = 1 - p_churn**2 - p_stay**2
        
        st.markdown(f"""<div class="math-box">
        <b>📐 Gini Impurity Formula:</b>
        <br><br>Gini = 1 - Σ(pᵢ²) = 1 - (p₁² + p₂² + ... + pₖ²)
        <br><br><b>Step-by-step calculation:</b>
        <br><br>1️⃣ <b>Count classes:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Churned: {churn} customers
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Stayed: {stay} customers
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Total: {total} customers
        <br><br>2️⃣ <b>Calculate proportions:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• p(Churn) = {churn}/{total} = <b>{p_churn:.4f}</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• p(Stay) = {stay}/{total} = <b>{p_stay:.4f}</b>
        <br><br>3️⃣ <b>Square each proportion:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• p(Churn)² = {p_churn:.4f}² = <b>{p_churn**2:.4f}</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• p(Stay)² = {p_stay:.4f}² = <b>{p_stay**2:.4f}</b>
        <br><br>4️⃣ <b>Sum of squares:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Σ(pᵢ²) = {p_churn**2:.4f} + {p_stay**2:.4f} = <b>{p_churn**2 + p_stay**2:.4f}</b>
        <br><br>5️⃣ <b>Final Gini:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Gini = 1 - {p_churn**2 + p_stay**2:.4f} = <b>{gini:.4f}</b>
        </div>""", unsafe_allow_html=True)
        
        # Visual interpretation
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Churned', 'Stayed'], y=[churn, stay], 
                            marker_color=['#f45d6d', '#22d3a7']))
        fig.update_layout(height=200, title=f"Distribution — Gini: {gini:.3f}", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown(f"""<div class="insight-box">
        💡 <b>Interpretation:</b>
        <br>• Gini = 0 → <b>Pure node</b> (all same class)
        <br>• Gini = 0.5 → <b>Maximum impurity</b> (50-50 split for binary)
        <br>• Current Gini = {gini:.3f} → {"Very pure!" if gini < 0.2 else "Quite mixed" if gini < 0.4 else "Very impure"}
        </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Gini Impurity?</b>
        <br><br>Gini measures <b>how often a randomly chosen element would be incorrectly labeled</b> if it was randomly labeled according to the distribution in the node.
        <br><br><b>Intuition:</b> Imagine picking a random customer and guessing their class based on the node's distribution. Gini = probability of guessing wrong.
        <br><br><b>Formula:</b> Gini = 1 - Σ(pᵢ²)
        <br>Where pᵢ = proportion of class i in the node
        <br><br><b>Properties:</b>
        <br>• Range: [0, 0.5] for binary classification
        <br>• Gini = 0 → perfectly pure (all one class)
        <br>• Gini = 0.5 → maximum impurity (50-50)
        <br><br><b>Used by:</b> CART (Classification and Regression Trees), scikit-learn's DecisionTreeClassifier (default)
        </div>
        <div class="math-box">
        <b>📐 Why square the probabilities?</b>
        <br><br>Squaring penalizes imbalance. If p=0.9:
        <br>• p² = 0.81 (high contribution to purity)
        <br>If p=0.5:
        <br>• p² = 0.25 (low contribution)
        <br><br>So nodes with dominant classes get lower Gini (more pure).
        </div>""",
        code_str='''import numpy as np

def gini_impurity(y):
    """Calculate Gini impurity for a node."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# Example: 30 churned, 70 stayed
y = [1]*30 + [0]*70  # 1=churn, 0=stay

gini = gini_impurity(y)
print(f"Gini = 1 - (0.3² + 0.7²)")
print(f"Gini = 1 - (0.09 + 0.49)")
print(f"Gini = 1 - 0.58 = {gini:.2f}")''',
        output_func=show_gini_math,
        concept_title="📊 Gini Impurity",
        output_title="Interactive Calculator"
    )

    # Row 3: Entropy - Full Math
    def show_entropy_math():
        st.markdown("#### 📊 Step-by-Step Entropy Calculation")
        
        col1, col2 = st.columns(2)
        with col1:
            churn = st.slider("Churned customers", 1, 99, 30, key="entropy_churn")
        with col2:
            stay = 100 - churn
            st.metric("Stayed customers", stay)
        
        total = churn + stay
        p_churn = churn / total
        p_stay = stay / total
        
        # Entropy calculation (handle log(0) by using small epsilon)
        entropy = 0
        if p_churn > 0:
            entropy -= p_churn * np.log2(p_churn)
        if p_stay > 0:
            entropy -= p_stay * np.log2(p_stay)
        
        st.markdown(f"""<div class="math-box">
        <b>📐 Entropy Formula:</b>
        <br><br>Entropy = -Σ(pᵢ × log₂(pᵢ))
        <br><br><b>Step-by-step calculation:</b>
        <br><br>1️⃣ <b>Calculate proportions:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• p(Churn) = {churn}/{total} = <b>{p_churn:.4f}</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• p(Stay) = {stay}/{total} = <b>{p_stay:.4f}</b>
        <br><br>2️⃣ <b>Calculate log₂ of each:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• log₂({p_churn:.4f}) = <b>{np.log2(p_churn):.4f}</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• log₂({p_stay:.4f}) = <b>{np.log2(p_stay):.4f}</b>
        <br><br>3️⃣ <b>Multiply p × log₂(p):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• {p_churn:.4f} × {np.log2(p_churn):.4f} = <b>{p_churn * np.log2(p_churn):.4f}</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• {p_stay:.4f} × {np.log2(p_stay):.4f} = <b>{p_stay * np.log2(p_stay):.4f}</b>
        <br><br>4️⃣ <b>Sum and negate:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Entropy = -({p_churn * np.log2(p_churn):.4f} + {p_stay * np.log2(p_stay):.4f})
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Entropy = -({p_churn * np.log2(p_churn) + p_stay * np.log2(p_stay):.4f}) = <b>{entropy:.4f}</b>
        </div>""", unsafe_allow_html=True)
        
        # Compare Gini vs Entropy
        gini = 1 - p_churn**2 - p_stay**2
        st.markdown(f"""<div class="insight-box">
        📊 <b>Comparison at this point:</b>
        <br>• Gini = {gini:.4f}
        <br>• Entropy = {entropy:.4f}
        <br>• Both agree: {"Pure!" if entropy < 0.5 else "Mixed" if entropy < 0.9 else "Very impure"}
        </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Entropy?</b>
        <br><br>Entropy measures <b>uncertainty or disorder</b> in a node. It comes from information theory — how many bits needed to encode the class?
        <br><br><b>Intuition:</b> If a node is 50-50, you need 1 full bit to encode the class. If it's 100-0, you need 0 bits (you already know!).
        <br><br><b>Formula:</b> Entropy = -Σ(pᵢ × log₂(pᵢ))
        <br><br><b>Properties:</b>
        <br>• Range: [0, 1] for binary classification
        <br>• Entropy = 0 → perfectly pure
        <br>• Entropy = 1 → maximum uncertainty (50-50)
        <br><br><b>Used by:</b> ID3, C4.5 algorithms, Information Gain criterion
        </div>
        <div class="math-box">
        <b>📐 Why logarithm?</b>
        <br><br>Log measures "bits of information":
        <br>• log₂(1) = 0 → certain event, no info needed
        <br>• log₂(0.5) = -1 → uncertain, need 1 bit
        <br>• log₂(0.25) = -2 → very uncertain, need 2 bits
        <br><br>Multiplying by p weights by frequency.
        </div>""",
        code_str='''import numpy as np

def entropy(y):
    """Calculate entropy for a node."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Filter out zero probabilities (log(0) undefined)
    probabilities = probabilities[probabilities > 0]
    ent = -np.sum(probabilities * np.log2(probabilities))
    return ent

# Example: 30 churned, 70 stayed
y = [1]*30 + [0]*70

ent = entropy(y)
print(f"Entropy = -(0.3×log₂(0.3) + 0.7×log₂(0.7))")
print(f"Entropy = -(0.3×(-1.737) + 0.7×(-0.515))")
print(f"Entropy = -(-0.521 + -0.360)")
print(f"Entropy = {ent:.3f}")''',
        output_func=show_entropy_math,
        concept_title="📊 Entropy",
        output_title="Interactive Calculator"
    )

    # Row 4: Information Gain - How splits are chosen
    def show_info_gain():
        st.markdown("#### 🎯 How Trees Choose the Best Split")
        
        # Sample data
        st.markdown("**Dataset: 100 customers**")
        
        # Parent node
        parent_churn, parent_stay = 40, 60
        parent_gini = 1 - (0.4)**2 - (0.6)**2
        
        # Split A: tenure < 24
        left_a_churn, left_a_stay = 30, 20  # 50 customers
        right_a_churn, right_a_stay = 10, 40  # 50 customers
        gini_left_a = 1 - (30/50)**2 - (20/50)**2
        gini_right_a = 1 - (10/50)**2 - (40/50)**2
        weighted_gini_a = 0.5 * gini_left_a + 0.5 * gini_right_a
        
        # Split B: charges > 65
        left_b_churn, left_b_stay = 35, 15  # 50 customers
        right_b_churn, right_b_stay = 5, 45  # 50 customers
        gini_left_b = 1 - (35/50)**2 - (15/50)**2
        gini_right_b = 1 - (5/50)**2 - (45/50)**2
        weighted_gini_b = 0.5 * gini_left_b + 0.5 * gini_right_b
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="math-box">
            <b>Split A: tenure < 24?</b>
            <br><br>Left (tenure < 24): 50 customers
            <br>&nbsp;&nbsp;• Churn: 30, Stay: 20
            <br>&nbsp;&nbsp;• Gini = 1 - (0.6)² - (0.4)² = <b>{gini_left_a:.3f}</b>
            <br><br>Right (tenure ≥ 24): 50 customers
            <br>&nbsp;&nbsp;• Churn: 10, Stay: 40
            <br>&nbsp;&nbsp;• Gini = 1 - (0.2)² - (0.8)² = <b>{gini_right_a:.3f}</b>
            <br><br>Weighted Gini = (50/100)×{gini_left_a:.3f} + (50/100)×{gini_right_a:.3f}
            <br>= <b>{weighted_gini_a:.3f}</b>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""<div class="math-box">
            <b>Split B: charges > 65?</b>
            <br><br>Left (charges > 65): 50 customers
            <br>&nbsp;&nbsp;• Churn: 35, Stay: 15
            <br>&nbsp;&nbsp;• Gini = 1 - (0.7)² - (0.3)² = <b>{gini_left_b:.3f}</b>
            <br><br>Right (charges ≤ 65): 50 customers
            <br>&nbsp;&nbsp;• Churn: 5, Stay: 45
            <br>&nbsp;&nbsp;• Gini = 1 - (0.1)² - (0.9)² = <b>{gini_right_b:.3f}</b>
            <br><br>Weighted Gini = (50/100)×{gini_left_b:.3f} + (50/100)×{gini_right_b:.3f}
            <br>= <b>{weighted_gini_b:.3f}</b>
            </div>""", unsafe_allow_html=True)
        
        gain_a = parent_gini - weighted_gini_a
        gain_b = parent_gini - weighted_gini_b
        winner = "Split B (charges > 65)" if gain_b > gain_a else "Split A (tenure < 24)"
        
        st.markdown(f"""<div class="insight-box">
        🏆 <b>Winner: {winner}</b>
        <br><br>Parent Gini: {parent_gini:.3f}
        <br>• Split A reduces Gini by: {parent_gini:.3f} - {weighted_gini_a:.3f} = <b>{gain_a:.3f}</b>
        <br>• Split B reduces Gini by: {parent_gini:.3f} - {weighted_gini_b:.3f} = <b>{gain_b:.3f}</b>
        <br><br>Split B creates purer nodes → chosen!
        </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 How does a tree choose the best split?</b>
        <br><br>The tree tries <b>every possible split</b> on every feature and picks the one that <b>reduces impurity the most</b>.
        <br><br><b>Information Gain = Parent Impurity - Weighted Child Impurity</b>
        <br><br><b>Algorithm:</b>
        <br>1. Calculate parent node's Gini/Entropy
        <br>2. For each feature:
        <br>&nbsp;&nbsp;&nbsp;• Try every unique value as threshold
        <br>&nbsp;&nbsp;&nbsp;• Split data into left/right
        <br>&nbsp;&nbsp;&nbsp;• Calculate weighted average of child impurities
        <br>3. Pick split with lowest weighted impurity
        <br><br><b>Weighted average:</b>
        <br>Weighted Gini = (n_left/n_total) × Gini_left + (n_right/n_total) × Gini_right
        </div>
        <div class="warn-box">
        ⚠️ <b>Computational cost:</b> For n samples and m features, the tree evaluates O(n × m) possible splits at each node!
        </div>""",
        code_str='''def information_gain(parent, left, right):
    """Calculate information gain from a split."""
    parent_gini = gini_impurity(parent)
    
    n = len(parent)
    n_left, n_right = len(left), len(right)
    
    # Weighted average of child impurities
    weighted_child = (n_left/n) * gini_impurity(left) + \\
                     (n_right/n) * gini_impurity(right)
    
    gain = parent_gini - weighted_child
    return gain

# Example: Which split is better?
parent = [1]*40 + [0]*60  # 40 churn, 60 stay

# Split A: tenure < 24
left_a = [1]*30 + [0]*20   # 30 churn, 20 stay
right_a = [1]*10 + [0]*40  # 10 churn, 40 stay

# Split B: charges > 65  
left_b = [1]*35 + [0]*15   # 35 churn, 15 stay
right_b = [1]*5 + [0]*45   # 5 churn, 45 stay

print(f"Gain A: {information_gain(parent, left_a, right_a):.3f}")
print(f"Gain B: {information_gain(parent, left_b, right_b):.3f}")''',
        output_func=show_info_gain,
        concept_title="🎯 Information Gain",
        output_title="Comparing Splits"
    )

    # Row 5: Gini vs Entropy comparison
    def show_comparison():
        st.markdown("#### 📊 Gini vs Entropy Curves")
        
        p = np.linspace(0.001, 0.999, 100)
        gini = 2 * p * (1 - p)  # Simplified for binary: 1 - p² - (1-p)²
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p, y=gini, mode='lines', name='Gini (scaled)', 
                                line=dict(color='#f45d6d', width=3)))
        fig.add_trace(go.Scatter(x=p, y=entropy, mode='lines', name='Entropy',
                                line=dict(color='#22d3a7', width=3)))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#f5b731", 
                     annotation_text="Max impurity", annotation_position="top")
        fig.update_layout(height=300, title="Impurity vs Class Proportion (Binary)",
                         xaxis_title="P(class=1)", yaxis_title="Impurity", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        comparison = pd.DataFrame({
            "Aspect": ["Formula", "Range (binary)", "Max at", "Computation", "Used by"],
            "Gini": ["1 - Σpᵢ²", "[0, 0.5]", "p = 0.5", "Faster (no log)", "CART, sklearn default"],
            "Entropy": ["-Σpᵢ log₂(pᵢ)", "[0, 1]", "p = 0.5", "Slower (log)", "ID3, C4.5, C5.0"]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Gini vs Entropy — Which to use?</b>
        <br><br><b>Similarities:</b>
        <br>• Both measure impurity
        <br>• Both are 0 for pure nodes
        <br>• Both max at 50-50 split
        <br>• Usually give same tree structure
        <br><br><b>Differences:</b>
        <br>• Gini is faster (no logarithm)
        <br>• Entropy has information-theoretic meaning
        <br>• Gini tends to isolate most frequent class
        <br>• Entropy tends to produce more balanced trees
        <br><br><b>In practice:</b> Use Gini (sklearn default). Switch to Entropy only if you need slightly different tree structure.
        </div>
        <div class="insight-box">
        💡 <b>Interview tip:</b> Know both formulas and be able to calculate by hand. Interviewers love asking "calculate Gini for [30, 70] split."
        </div>""",
        code_str='''from sklearn.tree import DecisionTreeClassifier

# Using Gini (default)
tree_gini = DecisionTreeClassifier(criterion='gini')

# Using Entropy
tree_entropy = DecisionTreeClassifier(criterion='entropy')

# Both usually produce similar trees
tree_gini.fit(X_train, y_train)
tree_entropy.fit(X_train, y_train)

print(f"Gini tree depth: {tree_gini.get_depth()}")
print(f"Entropy tree depth: {tree_entropy.get_depth()}")''',
        output_func=show_comparison,
        concept_title="⚖️ Gini vs Entropy",
        output_title="Visual Comparison"
    )

    iq([
        {"q": "Calculate Gini impurity for a node with 25 positive and 75 negative samples.", "d": "Easy", "c": ["Amazon", "Google"],
         "a": "Gini = 1 - (p₁² + p₂²) = 1 - (0.25² + 0.75²) = 1 - (0.0625 + 0.5625) = 1 - 0.625 = <b>0.375</b>",
         "t": "Show the formula, plug in numbers, compute step by step."},
        {"q": "Why might you choose Entropy over Gini impurity?", "d": "Medium", "c": ["Meta", "Microsoft"],
         "a": "Entropy has information-theoretic interpretation (bits of information). It tends to create more balanced trees and is more sensitive to class probability changes. However, Gini is computationally faster and usually gives similar results.",
         "t": "Mention both the theoretical reason (information theory) and practical consideration (computation speed)."},
        {"q": "What is Information Gain and how is it used in decision trees?", "d": "Medium", "c": ["Google", "Amazon"],
         "a": "Information Gain = Parent Impurity - Weighted Average of Child Impurities. It measures how much a split reduces uncertainty. The tree picks the split with highest information gain (lowest weighted child impurity).",
         "t": "Give the formula and explain that higher gain = better split."},
    ])


# ═══════════════════════════════════════
# ENSEMBLE METHODS
# ═══════════════════════════════════════
elif module == "🌲 Ensemble Methods":
    st.markdown("# 🌲 Ensemble Methods — Bagging, Boosting, XGBoost")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>Why Ensembles?</b> A single model can be wrong. But if we combine many models, their errors can <b>cancel out</b>.
    <br><br>🎯 <b>Two main strategies:</b>
    <br>• <b>Bagging:</b> Train models in parallel on different data samples → reduce variance
    <br>• <b>Boosting:</b> Train models sequentially, each fixing previous errors → reduce bias
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Bagging
    def show_bagging():
        st.markdown("#### 🎒 Bagging Step-by-Step")
        
        step = st.radio("Step:", ["1. Bootstrap Sampling", "2. Train Models", "3. Aggregate"], horizontal=True, key="bag_step")
        
        if step == "1. Bootstrap Sampling":
            st.markdown("**Original data: [A, B, C, D, E, F, G, H, I, J]**")
            np.random.seed(42)
            original = list("ABCDEFGHIJ")
            
            samples = []
            for i in range(3):
                sample = np.random.choice(original, size=10, replace=True)
                samples.append(sample)
            
            col1, col2, col3 = st.columns(3)
            for i, (col, sample) in enumerate(zip([col1, col2, col3], samples)):
                with col:
                    unique = len(set(sample))
                    st.markdown(f"**Sample {i+1}:**")
                    st.code(", ".join(sample))
                    st.caption(f"Unique: {unique}/10 ({unique*10}%)")
            
            st.markdown("""<div class="math-box">
            <b>📐 Bootstrap Sampling Math:</b>
            <br><br>• Sample WITH replacement: same item can appear multiple times
            <br>• Each sample has n items (same as original)
            <br>• P(item NOT picked in one draw) = (n-1)/n
            <br>• P(item NOT in sample) = ((n-1)/n)ⁿ ≈ 1/e ≈ 0.368
            <br>• So each sample contains ~63.2% unique items
            <br><br>This means each model sees different data → learns different patterns!
            </div>""", unsafe_allow_html=True)
            
        elif step == "2. Train Models":
            st.markdown("**Each model trains on its bootstrap sample:**")
            
            col1, col2, col3 = st.columns(3)
            trees = [
                {"split": "tenure < 20", "pred": "Churn if YES"},
                {"split": "charges > 70", "pred": "Churn if YES"},
                {"split": "tenure < 25", "pred": "Churn if YES"},
            ]
            for i, (col, tree) in enumerate(zip([col1, col2, col3], trees)):
                with col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[0.5], y=[0.7], mode='markers+text',
                        marker=dict(size=50, color='#7c6aff', symbol='square'),
                        text=[tree["split"]], textposition="middle center", 
                        textfont=dict(size=9, color='white'), showlegend=False))
                    fig.add_trace(go.Scatter(x=[0.25, 0.75], y=[0.3, 0.3], mode='markers+text',
                        marker=dict(size=35, color=['#f45d6d', '#22d3a7'], symbol='square'),
                        text=["Churn", "Stay"], textposition="middle center",
                        textfont=dict(size=8, color='white'), showlegend=False))
                    fig.update_layout(height=180, xaxis=dict(visible=False, range=[0,1]), 
                                     yaxis=dict(visible=False, range=[0,1]),
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     margin=dict(t=10,b=10,l=10,r=10), title=f"Tree {i+1}")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            st.markdown("""<div class="insight-box">
            💡 <b>Key insight:</b> Each tree learned slightly different rules because they saw different data!
            <br>Tree 1: tenure < 20 | Tree 2: charges > 70 | Tree 3: tenure < 25
            </div>""", unsafe_allow_html=True)
            
        else:  # Aggregate
            st.markdown("**For a new customer: tenure=18, charges=75**")
            
            predictions = [
                {"tree": 1, "rule": "tenure < 20?", "answer": "YES", "pred": "Churn", "color": "#f45d6d"},
                {"tree": 2, "rule": "charges > 70?", "answer": "YES", "pred": "Churn", "color": "#f45d6d"},
                {"tree": 3, "rule": "tenure < 25?", "answer": "YES", "pred": "Churn", "color": "#f45d6d"},
            ]
            
            cols = st.columns(4)
            for i, pred in enumerate(predictions):
                with cols[i]:
                    st.markdown(f"**Tree {pred['tree']}**")
                    st.markdown(f"{pred['rule']} → {pred['answer']}")
                    st.markdown(f"<span style='color:{pred['color']}'><b>{pred['pred']}</b></span>", unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown("**🗳️ VOTE**")
                st.markdown("Churn: 3")
                st.markdown("Stay: 0")
                st.markdown("<span style='color:#f45d6d'><b>→ CHURN</b></span>", unsafe_allow_html=True)
            
            st.markdown("""<div class="math-box">
            <b>📐 Aggregation Methods:</b>
            <br><br><b>Classification (Majority Vote):</b>
            <br>&nbsp;&nbsp;ŷ = mode(y₁, y₂, ..., yₙ)
            <br>&nbsp;&nbsp;Example: [Churn, Churn, Churn] → Churn (3-0)
            <br><br><b>Regression (Average):</b>
            <br>&nbsp;&nbsp;ŷ = (1/n) × Σyᵢ
            <br>&nbsp;&nbsp;Example: [100, 95, 105] → 100
            <br><br><b>Probability (Average probabilities):</b>
            <br>&nbsp;&nbsp;P(Churn) = (0.8 + 0.7 + 0.9) / 3 = 0.8
            </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Bagging?</b>
        <br><br><b>B</b>ootstrap <b>Agg</b>regat<b>ing</b> — train multiple models on random samples, then combine predictions.
        <br><br><b>Algorithm:</b>
        <br>1. Create B bootstrap samples (sample with replacement)
        <br>2. Train one model on each sample
        <br>3. Aggregate predictions (vote or average)
        <br><br><b>Why it works:</b>
        <br>• Each model overfits differently (different data)
        <br>• Random errors cancel when averaged
        <br>• Reduces <b>variance</b> without increasing bias
        </div>
        <div class="pros-cons-box">
        <b>⚖️ Pros & Cons</b>
        <br><br><span class="pros">✅ Pros:</span>
        <br>• Significantly reduces variance (overfitting)
        <br>• Models train in parallel (fast)
        <br>• Works with any base model
        <br>• Robust to noisy data
        <br><br><span class="cons">❌ Cons:</span>
        <br>• Doesn't reduce bias (underfitting)
        <br>• Less interpretable than single model
        <br>• More memory for storing models
        </div>
        <div class="warn-box">
        <b>⚠️ Avoiding Overfitting & Underfitting</b>
        <br><br><b>Overfitting:</b> Rare with bagging, but can happen with very deep base models
        <br><b>Underfitting:</b> If base models are too simple, bagging won't help
        <br><br><b>Key Hyperparameters:</b>
        <br>• <code>n_estimators</code>: More is usually better (50-500)
        <br>• <code>max_samples</code>: Fraction of data per sample (0.5-1.0)
        <br>• Base model complexity: Control via base estimator params
        <br><br><b>Best Practice:</b> Use moderately complex base models. Bagging reduces variance, so slightly overfit base models are OK.
        </div>
        <div class="insight-box">
        💡 <b>Random Forest = Bagging + Random Feature Selection</b>
        <br>RF adds extra randomness by only considering √m features at each split.
        </div>""",
        code_str='''from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging with decision trees
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,      # 100 trees
    max_samples=1.0,       # 100% of data per sample
    bootstrap=True,        # Sample with replacement
    random_state=42
)
bagging.fit(X_train, y_train)

# Each tree's prediction
for i, tree in enumerate(bagging.estimators_[:3]):
    pred = tree.predict([[18, 75]])  # tenure=18, charges=75
    print(f"Tree {i+1}: {pred[0]}")

# Final prediction (majority vote)
final = bagging.predict([[18, 75]])
print(f"Ensemble: {final[0]}")''',
        output_func=show_bagging,
        concept_title="🎒 Bagging",
        output_title="Step-by-Step"
    )

    # Row 2: Boosting Concept
    def show_boosting():
        st.markdown("#### 🚀 Boosting Step-by-Step")
        
        step = st.radio("Step:", ["1. Initial Model", "2. Focus on Errors", "3. Weighted Combination"], horizontal=True, key="boost_step")
        
        if step == "1. Initial Model":
            st.markdown("**Round 1: Train on original data**")
            
            # Sample data
            data = pd.DataFrame({
                "ID": [1, 2, 3, 4, 5, 6, 7, 8],
                "Tenure": [5, 10, 30, 40, 8, 15, 35, 45],
                "Actual": ["Churn", "Churn", "Stay", "Stay", "Churn", "Stay", "Stay", "Stay"],
                "Model 1": ["Churn", "Stay", "Stay", "Stay", "Churn", "Churn", "Stay", "Stay"],
                "Correct?": ["✓", "✗", "✓", "✓", "✓", "✗", "✓", "✓"]
            })
            st.dataframe(data, use_container_width=True, hide_index=True)
            
            st.markdown("""<div class="warn-box">
            ⚠️ <b>Model 1 made 2 errors:</b>
            <br>• ID 2: Predicted Stay, was Churn
            <br>• ID 6: Predicted Churn, was Stay
            <br><br>Next model will focus MORE on these!
            </div>""", unsafe_allow_html=True)
            
        elif step == "2. Focus on Errors":
            st.markdown("**Round 2: Increase weight on misclassified samples**")
            
            data = pd.DataFrame({
                "ID": [1, 2, 3, 4, 5, 6, 7, 8],
                "Original Weight": ["1/8", "1/8", "1/8", "1/8", "1/8", "1/8", "1/8", "1/8"],
                "Was Wrong?": ["No", "YES", "No", "No", "No", "YES", "No", "No"],
                "New Weight": ["0.08", "<b>0.25</b>", "0.08", "0.08", "0.08", "<b>0.25</b>", "0.08", "0.08"],
            })
            st.markdown(data.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            st.markdown("""<div class="math-box">
            <b>📐 Weight Update (AdaBoost):</b>
            <br><br>1. Calculate error rate: ε = (wrong predictions) / (total)
            <br>&nbsp;&nbsp;&nbsp;ε = 2/8 = 0.25
            <br><br>2. Calculate model weight: α = 0.5 × ln((1-ε)/ε)
            <br>&nbsp;&nbsp;&nbsp;α = 0.5 × ln(0.75/0.25) = 0.5 × ln(3) = <b>0.55</b>
            <br><br>3. Update sample weights:
            <br>&nbsp;&nbsp;&nbsp;• Correct: w × e^(-α) = 0.125 × e^(-0.55) = <b>0.08</b>
            <br>&nbsp;&nbsp;&nbsp;• Wrong: w × e^(+α) = 0.125 × e^(0.55) = <b>0.25</b>
            <br><br>Wrong samples now have 3× more weight!
            </div>""", unsafe_allow_html=True)
            
        else:  # Weighted Combination
            st.markdown("**Final: Combine all models with their weights**")
            
            models = [
                {"model": "Model 1", "weight": 0.55, "pred": "Churn", "contrib": 0.55},
                {"model": "Model 2", "weight": 0.72, "pred": "Churn", "contrib": 0.72},
                {"model": "Model 3", "weight": 0.48, "pred": "Stay", "contrib": -0.48},
            ]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                df = pd.DataFrame(models)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            with col2:
                total = sum(m["contrib"] for m in models)
                st.metric("Sum of weights", f"{total:.2f}")
                st.metric("Final", "CHURN" if total > 0 else "STAY")
            
            st.markdown("""<div class="math-box">
            <b>📐 Final Prediction:</b>
            <br><br>H(x) = sign(Σ αₘ × hₘ(x))
            <br><br>= sign(0.55×(+1) + 0.72×(+1) + 0.48×(-1))
            <br>= sign(0.55 + 0.72 - 0.48)
            <br>= sign(0.79)
            <br>= <b>+1 (Churn)</b>
            <br><br>Better models (higher α) have more influence!
            </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Boosting?</b>
        <br><br>Train models <b>sequentially</b>, each one focusing on the errors of previous models.
        <br><br><b>Key idea:</b> Instead of random samples (bagging), we <b>weight samples</b> — misclassified samples get higher weight.
        <br><br><b>Algorithm:</b>
        <br>1. Train model on weighted data
        <br>2. Calculate model's error rate
        <br>3. Increase weights of misclassified samples
        <br>4. Train next model on reweighted data
        <br>5. Combine all models (weighted vote)
        </div>
        <div class="pros-cons-box">
        <b>⚖️ Pros & Cons</b>
        <br><br><span class="pros">✅ Pros:</span>
        <br>• Reduces bias (fixes underfitting)
        <br>• Often achieves best accuracy
        <br>• Works well with weak learners
        <br>• Handles imbalanced data well
        <br><br><span class="cons">❌ Cons:</span>
        <br>• Prone to overfitting with noisy data
        <br>• Sequential training (slower)
        <br>• Sensitive to outliers
        <br>• More hyperparameters to tune
        </div>
        <div class="warn-box">
        <b>⚠️ Avoiding Overfitting & Underfitting</b>
        <br><br><b>Overfitting signs:</b> Train accuracy keeps improving, test plateaus/drops
        <br><b>Underfitting signs:</b> Both accuracies low, need more rounds
        <br><br><b>Key Hyperparameters:</b>
        <br>• <code>n_estimators</code>: More rounds = more complex (use early stopping!)
        <br>• <code>learning_rate</code>: Lower = more regularization (0.01-0.3)
        <br>• <code>max_depth</code>: Shallower trees = less overfit (3-6)
        <br><br><b>Best Practice:</b> Use early stopping! Monitor validation error and stop when it stops improving. Lower learning rate + more trees often works best.
        </div>
        <div class="insight-box">
        💡 <b>Boosting vs Bagging:</b>
        <br>• Bagging: Parallel, reduces variance
        <br>• Boosting: Sequential, reduces bias
        </div>""",
        code_str='''from sklearn.ensemble import AdaBoostClassifier

# AdaBoost with decision stumps
ada = AdaBoostClassifier(
    n_estimators=50,       # 50 rounds
    learning_rate=1.0,     # How much to trust each model
    random_state=42
)
ada.fit(X_train, y_train)

# See model weights (alpha values)
print("Model weights (first 5):")
for i, weight in enumerate(ada.estimator_weights_[:5]):
    print(f"  Model {i+1}: α = {weight:.3f}")

# Staged predictions (see improvement over rounds)
for i, pred in enumerate(ada.staged_predict(X_test)):
    if i % 10 == 9:  # Every 10 rounds
        acc = (pred == y_test).mean()
        print(f"After {i+1} rounds: {acc:.1%}")''',
        output_func=show_boosting,
        concept_title="🚀 Boosting",
        output_title="Step-by-Step"
    )

    # Row 3: AdaBoost Math
    def show_adaboost():
        st.markdown("#### 📐 AdaBoost Algorithm")
        
        st.markdown("""<div class="math-box">
        <b>AdaBoost (Adaptive Boosting) — Full Algorithm:</b>
        <br><br><b>Initialize:</b> wᵢ = 1/n for all samples
        <br><br><b>For m = 1 to M (number of models):</b>
        <br><br>&nbsp;&nbsp;1️⃣ <b>Train weak learner</b> hₘ on weighted data
        <br><br>&nbsp;&nbsp;2️⃣ <b>Calculate weighted error:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;εₘ = Σ wᵢ × I(yᵢ ≠ hₘ(xᵢ)) / Σ wᵢ
        <br><br>&nbsp;&nbsp;3️⃣ <b>Calculate model weight:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;αₘ = 0.5 × ln((1 - εₘ) / εₘ)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• If ε = 0.1: α = 0.5 × ln(9) = 1.10 (high weight)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• If ε = 0.4: α = 0.5 × ln(1.5) = 0.20 (low weight)
        <br><br>&nbsp;&nbsp;4️⃣ <b>Update sample weights:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;wᵢ = wᵢ × exp(-αₘ × yᵢ × hₘ(xᵢ))
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Correct: wᵢ × e^(-α) → weight decreases
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Wrong: wᵢ × e^(+α) → weight increases
        <br><br>&nbsp;&nbsp;5️⃣ <b>Normalize weights:</b> wᵢ = wᵢ / Σwᵢ
        <br><br><b>Final prediction:</b>
        <br>H(x) = sign(Σ αₘ × hₘ(x))
        </div>""", unsafe_allow_html=True)
        
        # Visualize weight evolution
        st.markdown("**Sample Weight Evolution:**")
        rounds = [1, 2, 3, 4, 5]
        correct_weights = [0.125, 0.08, 0.05, 0.03, 0.02]
        wrong_weights = [0.125, 0.25, 0.40, 0.55, 0.65]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rounds, y=correct_weights, mode='lines+markers',
            name='Correctly classified', line=dict(color='#22d3a7', width=3)))
        fig.add_trace(go.Scatter(x=rounds, y=wrong_weights, mode='lines+markers',
            name='Misclassified', line=dict(color='#f45d6d', width=3)))
        fig.update_layout(height=250, title="Weight Changes Over Rounds",
                         xaxis_title="Round", yaxis_title="Sample Weight", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 AdaBoost Deep Dive</b>
        <br><br><b>Key insight:</b> The model weight α depends on error rate:
        <br>• Low error → high α (trust this model)
        <br>• High error → low α (don't trust much)
        <br>• Error = 50% → α = 0 (random guessing, ignore)
        <br>• Error > 50% → α < 0 (flip predictions!)
        <br><br><b>Why "weak learners"?</b>
        <br>AdaBoost works best with simple models (stumps = 1-split trees). Complex models overfit before boosting helps.
        <br><br><b>Exponential loss:</b>
        <br>AdaBoost minimizes: L = Σ exp(-yᵢ × H(xᵢ))
        <br>This is why weights update exponentially.
        </div>
        <div class="insight-box">
        💡 <b>Fun fact:</b> AdaBoost was one of the first ML algorithms proven to have theoretical guarantees. It's guaranteed to reduce training error to 0 given enough rounds (if weak learners are better than random).
        </div>""",
        code_str='''from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# AdaBoost with stumps (depth=1)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME',  # Original AdaBoost
    random_state=42
)
ada.fit(X_train, y_train)

# Examine individual stumps
for i, (stump, alpha) in enumerate(zip(
    ada.estimators_[:3], 
    ada.estimator_weights_[:3]
)):
    feature = stump.tree_.feature[0]
    threshold = stump.tree_.threshold[0]
    print(f"Stump {i+1}: feature[{feature}] <= {threshold:.2f}")
    print(f"         weight α = {alpha:.3f}")''',
        output_func=show_adaboost,
        concept_title="📐 AdaBoost Math",
        output_title="Algorithm Details"
    )

    # Row 4: Gradient Boosting & XGBoost
    def show_xgboost():
        st.markdown("#### 🚀 Gradient Boosting & XGBoost")
        
        view = st.radio("View:", ["Gradient Boosting Idea", "XGBoost Improvements", "Math Details"], horizontal=True, key="xgb_view")
        
        if view == "Gradient Boosting Idea":
            st.markdown("""<div class="math-box">
            <b>📐 Gradient Boosting — Key Idea:</b>
            <br><br>Instead of reweighting samples, fit each new model to the <b>residuals</b> (errors) of the previous model.
            <br><br><b>Algorithm:</b>
            <br>1. Start with initial prediction: F₀(x) = mean(y)
            <br>2. For m = 1 to M:
            <br>&nbsp;&nbsp;&nbsp;a. Calculate residuals: rᵢ = yᵢ - Fₘ₋₁(xᵢ)
            <br>&nbsp;&nbsp;&nbsp;b. Fit tree hₘ to residuals
            <br>&nbsp;&nbsp;&nbsp;c. Update: Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
            <br><br><b>Example (Regression):</b>
            <br>• Actual: y = 100
            <br>• Model 1 predicts: 80 → residual = 20
            <br>• Model 2 predicts residual: 15 → new pred = 80 + 15 = 95
            <br>• Model 3 predicts residual: 4 → new pred = 95 + 4 = 99
            <br>• Getting closer each round!
            </div>""", unsafe_allow_html=True)
            
            # Visual
            rounds = [0, 1, 2, 3, 4, 5]
            predictions = [50, 75, 88, 94, 97, 99]
            actual = [100] * 6
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rounds, y=actual, mode='lines', name='Actual',
                line=dict(color='#22d3a7', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=rounds, y=predictions, mode='lines+markers', name='Prediction',
                line=dict(color='#7c6aff', width=3), marker=dict(size=10)))
            fig.update_layout(height=250, title="Prediction Converges to Actual",
                             xaxis_title="Boosting Round", yaxis_title="Value", **DL)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
        elif view == "XGBoost Improvements":
            st.markdown("**XGBoost = eXtreme Gradient Boosting**")
            
            improvements = pd.DataFrame({
                "Feature": ["Regularization", "Second-order gradients", "Sparsity handling", "Parallel processing", "Tree pruning"],
                "Gradient Boosting": ["None", "First-order only", "Basic", "Sequential", "Pre-pruning"],
                "XGBoost": ["L1 + L2 on weights", "Uses Hessian (2nd derivative)", "Built-in sparse aware", "Parallel tree building", "Post-pruning (max_depth)"],
                "Benefit": ["Prevents overfitting", "Faster convergence", "Handles missing values", "10x faster", "Better trees"]
            })
            st.dataframe(improvements, use_container_width=True, hide_index=True)
            
            st.markdown("""<div class="insight-box">
            💡 <b>Why XGBoost dominates Kaggle:</b>
            <br>• Regularization prevents overfitting
            <br>• Handles missing values automatically
            <br>• Fast parallel training
            <br>• Built-in cross-validation
            <br>• Feature importance for free
            </div>""", unsafe_allow_html=True)
            
        else:  # Math Details
            st.markdown("""<div class="math-box">
            <b>📐 XGBoost Objective Function:</b>
            <br><br>Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
            <br><br>Where:
            <br>• L = Loss function (e.g., MSE, log loss)
            <br>• Ω = Regularization term
            <br><br><b>Regularization:</b>
            <br>Ω(f) = γT + ½λΣwⱼ²
            <br>• γ = penalty for number of leaves (T)
            <br>• λ = L2 penalty on leaf weights (w)
            <br><br><b>Split Gain (how XGBoost chooses splits):</b>
            <br><br>Gain = ½ × [G²ₗ/(Hₗ+λ) + G²ᵣ/(Hᵣ+λ) - (Gₗ+Gᵣ)²/(Hₗ+Hᵣ+λ)] - γ
            <br><br>Where:
            <br>• G = sum of gradients (first derivative)
            <br>• H = sum of hessians (second derivative)
            <br>• λ = regularization parameter
            <br>• γ = minimum gain to make split
            </div>""", unsafe_allow_html=True)
            
            st.markdown("""<div class="warn-box">
            ⚠️ <b>Key hyperparameters:</b>
            <br>• <b>n_estimators:</b> Number of trees (100-1000)
            <br>• <b>max_depth:</b> Tree depth (3-10, usually 6)
            <br>• <b>learning_rate:</b> Step size (0.01-0.3)
            <br>• <b>subsample:</b> Row sampling (0.8-1.0)
            <br>• <b>colsample_bytree:</b> Column sampling (0.8-1.0)
            </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Gradient Boosting vs XGBoost</b>
        <br><br><b>Gradient Boosting:</b>
        <br>• Fits trees to residuals (errors)
        <br>• Uses gradient descent in function space
        <br>• Each tree corrects previous errors
        <br><br><b>XGBoost (eXtreme Gradient Boosting):</b>
        <br>• Optimized implementation of gradient boosting
        <br>• Adds regularization (L1 + L2)
        <br>• Uses second-order gradients (Hessian)
        <br>• Handles missing values automatically
        </div>
        <div class="pros-cons-box">
        <b>⚖️ Pros & Cons</b>
        <br><br><span class="pros">✅ Pros:</span>
        <br>• State-of-the-art for tabular data
        <br>• Built-in regularization prevents overfitting
        <br>• Handles missing values automatically
        <br>• Feature importance built-in
        <br>• Very fast (parallel tree building)
        <br><br><span class="cons">❌ Cons:</span>
        <br>• Many hyperparameters to tune
        <br>• Can still overfit without proper tuning
        <br>• Less interpretable than single tree
        <br>• Sensitive to outliers
        </div>
        <div class="warn-box">
        <b>⚠️ Avoiding Overfitting & Underfitting</b>
        <br><br><b>Overfitting signs:</b> Train AUC >> Test AUC, validation loss increases
        <br><b>Underfitting signs:</b> Both metrics low, need more trees/depth
        <br><br><b>Key Hyperparameters:</b>
        <br>• <code>learning_rate</code>: Lower = more regularization (0.01-0.3)
        <br>• <code>max_depth</code>: Shallower = less overfit (3-8)
        <br>• <code>n_estimators</code>: Use early stopping!
        <br>• <code>subsample</code>: Row sampling (0.6-0.9)
        <br>• <code>colsample_bytree</code>: Column sampling (0.6-0.9)
        <br>• <code>reg_alpha/reg_lambda</code>: L1/L2 regularization
        <br><br><b>Best Practice:</b> Always use early stopping with validation set. Start with learning_rate=0.1, max_depth=6, then tune. Lower learning rate + more trees usually wins.
        </div>
        <div class="math-box">
        <b>📐 Learning Rate (η):</b>
        <br><br>Fₘ(x) = Fₘ₋₁(x) + <b>η</b> × hₘ(x)
        <br><br>• η = 1.0: Full step (fast but may overshoot)
        <br>• η = 0.1: Small step (slow but stable)
        <br>• Lower η needs more trees
        <br>• Typical: η = 0.1, n_estimators = 1000
        </div>""",
        code_str='''import xgboost as xgb

# XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_
for feat, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
    print(f"{feat}: {imp:.3f}")

# Predict with probability
proba = model.predict_proba(X_test)
print(f"P(Churn): {proba[0][1]:.2%}")''',
        output_func=show_xgboost,
        concept_title="🚀 XGBoost",
        output_title="Deep Dive"
    )

    # Row 5: Comparison Chart
    def show_ensemble_comparison():
        st.markdown("#### 📊 Ensemble Methods Comparison")
        
        comparison = pd.DataFrame({
            "Method": ["Bagging", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost"],
            "Training": ["Parallel", "Parallel", "Sequential", "Sequential", "Sequential"],
            "Reduces": ["Variance", "Variance", "Bias", "Bias", "Bias"],
            "Base Model": ["Any", "Decision Tree", "Weak learner", "Decision Tree", "Decision Tree"],
            "Key Idea": ["Bootstrap samples", "Bootstrap + random features", "Reweight samples", "Fit residuals", "Fit residuals + regularization"],
            "Overfitting Risk": ["Low", "Low", "Medium", "High", "Medium (regularized)"]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        # When to use what
        st.markdown("""<div class="insight-box">
        🎯 <b>When to use what:</b>
        <br><br>• <b>Random Forest:</b> Good default, interpretable, robust
        <br>• <b>XGBoost:</b> Best performance on tabular data, Kaggle winner
        <br>• <b>LightGBM:</b> Faster than XGBoost, good for large data
        <br>• <b>CatBoost:</b> Best for categorical features
        <br>• <b>AdaBoost:</b> Simple, good for understanding boosting
        </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Summary: Bagging vs Boosting</b>
        <br><br><b>Bagging (Random Forest):</b>
        <br>• Train in parallel
        <br>• Each model independent
        <br>• Reduces variance (overfitting)
        <br>• Robust to noise
        <br><br><b>Boosting (XGBoost):</b>
        <br>• Train sequentially
        <br>• Each model fixes previous errors
        <br>• Reduces bias (underfitting)
        <br>• Can overfit if not regularized
        <br><br><b>Rule of thumb:</b>
        <br>• High variance (overfitting)? → Bagging
        <br>• High bias (underfitting)? → Boosting
        <br>• Don't know? → Try XGBoost first
        </div>
        <div class="math-box">
        <b>📐 Bias-Variance in Ensembles:</b>
        <br><br><b>Bagging:</b>
        <br>• Var(avg) = Var(single)/n + correlation term
        <br>• More trees → lower variance
        <br>• Bias unchanged
        <br><br><b>Boosting:</b>
        <br>• Each round reduces training error
        <br>• Bias decreases with more rounds
        <br>• Variance may increase (overfit risk)
        </div>""",
        code_str='''# Quick comparison
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier
)
import xgboost as xgb

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"{name}:")
    print(f"  Train: {train_acc:.1%}, Test: {test_acc:.1%}")''',
        output_func=show_ensemble_comparison,
        concept_title="⚖️ Comparison",
        output_title="Which to Use?"
    )

    iq([
        {"q": "Explain the difference between Bagging and Boosting.", "d": "Medium", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>Bagging:</b> Trains models in parallel on bootstrap samples, combines by voting/averaging. Reduces variance. <b>Boosting:</b> Trains models sequentially, each focusing on previous errors. Reduces bias. Bagging = independent models, Boosting = dependent models.",
         "t": "Mention parallel vs sequential, variance vs bias reduction."},
        {"q": "How does XGBoost differ from regular Gradient Boosting?", "d": "Medium", "c": ["Amazon", "Microsoft"],
         "a": "XGBoost adds: (1) L1/L2 regularization on leaf weights, (2) Second-order gradients (Hessian) for better optimization, (3) Built-in handling of missing values, (4) Parallel tree construction, (5) Tree pruning. These make it faster and less prone to overfitting.",
         "t": "Focus on regularization and second-order gradients as key differences."},
        {"q": "What is the role of learning rate in boosting algorithms?", "d": "Easy", "c": ["Google", "Meta"],
         "a": "Learning rate (η) controls how much each tree contributes: F(x) = F(x) + η×h(x). Lower η = smaller steps = more trees needed but more stable. Higher η = faster training but may overshoot. Typical: η=0.1 with 100-1000 trees.",
         "t": "Explain the tradeoff: lower learning rate needs more trees but is more stable."},
        {"q": "Why does AdaBoost use 'weak learners' like decision stumps?", "d": "Hard", "c": ["Google", "Meta"],
         "a": "Weak learners (slightly better than random) are ideal because: (1) They have high bias but low variance, (2) Boosting reduces bias, so starting with high bias is fine, (3) Complex models would overfit before boosting helps, (4) Stumps are fast to train. The ensemble becomes strong through combination.",
         "t": "Explain that boosting reduces bias, so we want models with low variance to start."},
    ])


# ═══════════════════════════════════════
# UNSUPERVISED LEARNING
# ═══════════════════════════════════════
elif module == "🧩 Unsupervised Learning":
    st.markdown("# 🧩 Unsupervised Learning")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Unsupervised Learning?</b> It's <b>finding hidden patterns without labels</b> — the algorithm discovers structure on its own.
    No "correct answers" to learn from. Like exploring a new city without a map — you find natural groupings yourself.
    <br><br>🎯 <b>Key tasks:</b> Clustering (group similar items), dimensionality reduction (compress features), anomaly detection (find outliers).
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Supervised vs Unsupervised
    def show_unsupervised_intro():
        comparison = pd.DataFrame({
            "Aspect": ["Labels", "Goal", "Example", "Algorithms"],
            "Supervised": ["Has labels (Y)", "Predict Y from X", "Predict churn", "Logistic Reg, Trees, RF"],
            "Unsupervised": ["No labels", "Find hidden patterns", "Segment customers", "K-Means, PCA, DBSCAN"],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Unsupervised Learning?</b>
        <br><br>Unsupervised learning is <b>finding patterns without labels</b>. No "correct answers" to learn from — the algorithm discovers structure on its own.
        <br><br><b>Analogy:</b> Imagine exploring a new city without a map or guide. You naturally find that restaurants cluster in one area, shops in another, and residential areas elsewhere. You discovered the structure yourself!
        <br><br><b>Supervised vs Unsupervised:</b>
        <br>• <b>Supervised:</b> "Here are customers and whether they churned. Learn to predict churn."
        <br>• <b>Unsupervised:</b> "Here are customers. Find natural groupings."
        <br><br><b>Common tasks:</b>
        <br>• <b>Clustering:</b> Group similar items (customer segments)
        <br>• <b>Dimensionality Reduction:</b> Compress features (PCA)
        <br>• <b>Anomaly Detection:</b> Find outliers (fraud detection)
        <br>• <b>Association:</b> Find co-occurring items (market basket)
        <br><br><b>When to use it:</b>
        <br>• You don't have labels (no "correct answer" to learn from)
        <br>• You want to discover hidden structure
        <br>• You want to reduce complexity before supervised learning
        </div>
        <div class="math-box">
        <b>📐 Supervised vs Unsupervised — Example:</b>
        <br><br><b>Supervised (Churn Prediction):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Input: [tenure=12, charges=$80]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Label: churned = <b>Yes</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Learn: what features predict churn?
        <br><br><b>Unsupervised (Customer Segmentation):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Input: [tenure=12, charges=$80]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Label: <b>None!</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Discover: are there natural customer groups?
        <br><br>🧠 Unsupervised might find: "High-value loyalists", "Price-sensitive newbies", "At-risk churners" — without being told these groups exist!
        </div>
        <div class="insight-box">💡 <b>Business value:</b> Unsupervised learning discovers segments you didn't know existed. Then you can create targeted strategies for each!</div>""",
        code_str='''# Supervised: Has labels
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)  # y_train = labels

# Unsupervised: No labels
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)  # No y! Just X
clusters = kmeans.labels_  # Discovered groups''',
        output_func=show_unsupervised_intro,
        concept_title="🧩 Supervised vs Unsupervised",
        output_title="Key Differences"
    )

    # Row 2: K-Means Clustering
    def show_kmeans():
        np.random.seed(42)
        c1 = np.random.normal([2, 2], 0.8, (50, 2))
        c2 = np.random.normal([7, 7], 0.8, (50, 2))
        c3 = np.random.normal([2, 8], 0.8, (50, 2))
        data = np.vstack([c1, c2, c3])
        
        from sklearn.cluster import KMeans
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        colors = ['#7c6aff', '#22d3a7', '#f5b731']
        
        fig = go.Figure()
        for i in range(k):
            mask = labels == i
            fig.add_trace(go.Scatter(x=data[mask, 0], y=data[mask, 1], mode='markers',
                marker=dict(color=colors[i], size=8, opacity=0.6), name=f'Cluster {i+1}'))
        fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers',
            marker=dict(color='white', size=15, symbol='x', line=dict(width=2, color='black')), name='Centroids'))
        fig.update_layout(height=280, title=f"K-Means with K={k}", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is K-Means Clustering?</b>
        <br><br>K-Means groups data into K clusters based on <b>similarity</b>. Points in the same cluster are close to each other; points in different clusters are far apart.
        <br><br><b>The algorithm (simple but powerful):</b>
        <br>1. Pick K random points as initial "centroids"
        <br>2. Assign each data point to the nearest centroid
        <br>3. Move each centroid to the center of its assigned points
        <br>4. Repeat steps 2-3 until centroids stop moving
        </div>
        <div class="pros-cons-box">
        <b>⚖️ Pros & Cons</b>
        <br><br><span class="pros">✅ Pros:</span>
        <br>• Simple and fast (scales to large datasets)
        <br>• Easy to interpret results
        <br>• Works well with spherical clusters
        <br>• Guaranteed to converge
        <br><br><span class="cons">❌ Cons:</span>
        <br>• Must specify K in advance
        <br>• Assumes spherical clusters (bad for elongated shapes)
        <br>• Sensitive to initial centroids
        <br>• Sensitive to outliers
        <br>• Requires feature scaling
        </div>
        <div class="warn-box">
        <b>⚠️ Avoiding Poor Clustering</b>
        <br><br><b>Signs of bad clustering:</b> Uneven cluster sizes, low silhouette score
        <br><br><b>Key Hyperparameters:</b>
        <br>• <code>n_clusters</code>: Use elbow method or silhouette score
        <br>• <code>n_init</code>: Run multiple times with different seeds (10-20)
        <br>• <code>init</code>: Use 'k-means++' for smarter initialization
        <br><br><b>Best Practices:</b>
        <br>• Always scale features first (StandardScaler)
        <br>• Try multiple K values and compare
        <br>• Use silhouette score to validate
        <br>• Consider DBSCAN for non-spherical clusters
        </div>
        <div class="math-box">
        <b>📐 K-Means Distance — Step by Step:</b>
        <br><br><b>Given:</b> Point P = (5, 8)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Centroid A = (2, 4), Centroid B = (7, 9)
        <br><br><b>Distance to A (Euclidean):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;d = √[(5-2)² + (8-4)²]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;d = √[9 + 16] = √25 = <b>5.0</b>
        <br><br><b>Distance to B:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;d = √[(5-7)² + (8-9)²]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;d = √[4 + 1] = √5 = <b>2.24</b>
        <br><br><b>Assignment:</b> P → Cluster B (closer!)
        </div>""",
        code_str='''from sklearn.cluster import KMeans
import numpy as np

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit to data (no labels needed!)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_
print(f"Cluster assignments: {labels[:10]}")

# Get centroids
centroids = kmeans.cluster_centers_
print(f"Centroids:\\n{centroids}")

# Predict cluster for new data
new_point = [[5, 5]]
cluster = kmeans.predict(new_point)
print(f"New point belongs to cluster: {cluster[0]}")''',
        output_func=show_kmeans,
        concept_title="🎯 K-Means Clustering",
        output_title="Interactive Demo"
    )

    # Row 3: Elbow Method
    def show_elbow():
        np.random.seed(42)
        c1 = np.random.normal([2, 2], 0.8, (50, 2))
        c2 = np.random.normal([7, 7], 0.8, (50, 2))
        c3 = np.random.normal([2, 8], 0.8, (50, 2))
        data = np.vstack([c1, c2, c3])
        
        from sklearn.cluster import KMeans
        inertias = []
        K_range = range(1, 8)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
            inertias.append(km.inertia_)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
            line=dict(color='#22d3a7', width=2), marker=dict(size=10)))
        fig.add_vline(x=3, line_dash="dash", line_color="#f5b731", annotation_text="Elbow at K=3")
        fig.update_layout(height=250, title="Elbow Method: Finding Optimal K", xaxis_title="K (clusters)", yaxis_title="Inertia", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 How do you choose K?</b>
        <br><br>The Elbow Method helps you find the optimal number of clusters. It's based on a simple idea: more clusters = lower inertia, but at some point, adding more clusters doesn't help much.
        <br><br><b>What is Inertia?</b>
        <br>Inertia measures how "tight" the clusters are — the sum of squared distances from each point to its centroid. Lower = tighter clusters.
        <br><br><b>The Elbow Method:</b>
        <br>1. Run K-Means for K = 1, 2, 3, 4, 5...
        <br>2. Plot inertia vs K
        <br>3. Look for the "elbow" — where the curve bends
        <br>4. That's your optimal K!
        <br><br><b>Why does it work?</b>
        <br>• K=1: All points in one cluster → high inertia
        <br>• K=2: Two clusters → inertia drops a lot
        <br>• K=3: Three clusters → inertia drops more
        <br>• K=4+: Diminishing returns — not much improvement
        <br><br>The "elbow" is where you get the best bang for your buck — good cluster quality without over-segmenting.
        </div>
        <div class="math-box">
        <b>📐 Inertia Calculation — Step by Step:</b>
        <br><br><b>Cluster 1:</b> Points at (1,1), (2,2), (1,2)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Centroid = ((1+2+1)/3, (1+2+2)/3) = (1.33, 1.67)
        <br><br>&nbsp;&nbsp;&nbsp;&nbsp;Distance² from (1,1):
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1-1.33)² + (1-1.67)² = 0.11 + 0.45 = 0.56
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Distance² from (2,2): 0.56
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Distance² from (1,2): 0.22
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Cluster inertia = 0.56 + 0.56 + 0.22 = <b>1.34</b>
        <br><br><b>Total Inertia:</b> Sum across all clusters
        <br><br>🧠 Lower inertia = tighter clusters = better!
        </div>
        <div class="insight-box">💡 <b>Pro tip:</b> The elbow isn't always obvious. Also try the silhouette score, which measures how well-separated clusters are.</div>""",
        code_str='''from sklearn.cluster import KMeans

# Calculate inertia for different K values
inertias = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
import matplotlib.pyplot as plt
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K (number of clusters)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Pick K at the "elbow" (where curve bends)''',
        output_func=show_elbow,
        concept_title="📐 Choosing K",
        output_title="The Elbow Method"
    )

    iq([
        {"q": "Explain K-Means clustering. How does it work?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "K-Means groups data into K clusters. <b>Steps:</b> (1) Pick K random centroids. (2) Assign each point to nearest centroid. (3) Move centroids to cluster means. (4) Repeat until stable. <b>Key:</b> You must choose K beforehand.",
         "t": "Mention the elbow method for choosing K."},
        {"q": "What's the difference between supervised and unsupervised learning?", "d": "Easy", "c": ["Google", "Meta"],
         "a": "<b>Supervised:</b> Has labels — learns to predict Y from X. <b>Unsupervised:</b> No labels — finds hidden patterns. <b>Example:</b> Predicting churn (supervised) vs segmenting customers (unsupervised).",
         "t": "Give concrete examples for each."},
    ])


# ═══════════════════════════════════════
# BIAS & VARIANCE
# ═══════════════════════════════════════
elif module == "📖 Bias & Variance":
    st.markdown("# 📖 Bias & Variance")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is the Bias-Variance Tradeoff?</b> It's the <b>central tension in all of ML</b>. Every model makes errors, and those errors come from two sources:
    <br><br>• <b>Bias:</b> Error from oversimplifying — your model is too dumb to capture the real pattern
    <br>• <b>Variance:</b> Error from overcomplicating — your model memorizes noise instead of learning the signal
    <br><br>🎯 <b>The cruel truth:</b> Reducing one often increases the other. Your job is to find the sweet spot.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: The Tradeoff
    def show_bias_variance_intro():
        # Simulate bias-variance curves
        x = np.linspace(1, 10, 100)
        bias = 5 / x
        variance = 0.1 * x ** 1.5
        total = bias + variance
        
        mc = st.columns(3)
        mc[0].metric("Low Complexity", "High Bias", "Underfitting")
        mc[1].metric("Optimal", "Balanced", "Best generalization")
        mc[2].metric("High Complexity", "High Variance", "Overfitting")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=bias, mode='lines', line=dict(color='#7c6aff', width=2), name='Bias²'))
        fig.add_trace(go.Scatter(x=x, y=variance, mode='lines', line=dict(color='#f45d6d', width=2), name='Variance'))
        fig.add_trace(go.Scatter(x=x, y=total, mode='lines', line=dict(color='#22d3a7', width=3), name='Total Error'))
        optimal_x = x[np.argmin(total)]
        fig.add_vline(x=optimal_x, line_dash="dash", line_color="#f5b731", annotation_text=f"Optimal: {optimal_x:.1f}")
        fig.update_layout(height=280, title="Bias-Variance Tradeoff", xaxis_title="Model Complexity", yaxis_title="Error", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Bias? (Underfitting)</b>
        <br><br><b>Bias</b> is error from <b>wrong assumptions</b>. Your model is too simple to capture the real pattern.
        <br><br><b>Analogy:</b> Imagine predicting house prices using ONLY square footage. You ignore bedrooms, location, age... Your model will consistently miss because it's too simple.
        <br><br><b>Symptoms:</b>
        <br>• Training accuracy is low
        <br>• Test accuracy is also low
        <br>• Model predictions are consistently wrong in the same direction
        <br><br><b>🤔 What is Variance? (Overfitting)</b>
        <br><br><b>Variance</b> is error from <b>sensitivity to training data</b>. Your model memorized the noise.
        <br><br><b>Analogy:</b> A student who memorizes every practice problem but can't solve new ones. They learned the answers, not the concepts.
        <br><br><b>Symptoms:</b>
        <br>• Training accuracy is very high (95%+)
        <br>• Test accuracy is much lower (70%)
        <br>• Model predictions are wildly different on different datasets
        </div>
        <div class="math-box">
        <b>📐 Bias-Variance — Dartboard Analogy:</b>
        <br><br><b>High Bias, Low Variance:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;All darts land in the same spot... but far from bullseye
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Predictions: [85, 84, 86, 85, 85] (True: 100)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Consistent but <b>wrong</b>
        <br><br><b>Low Bias, High Variance:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Darts scattered all over... but centered on bullseye
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Predictions: [70, 130, 95, 110, 85] (True: 100)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Average is right but <b>unreliable</b>
        <br><br><b>Goal:</b> Low bias AND low variance = accurate AND consistent
        </div>
        <div class="warn-box">⚠️ <b>Total Error = Bias² + Variance + Irreducible Noise</b>. You can't eliminate noise, but you can balance bias and variance.</div>""",
        code_str='''from sklearn.model_selection import cross_val_score

# High bias (underfitting): simple model
from sklearn.linear_model import LinearRegression
simple = LinearRegression()
scores_simple = cross_val_score(simple, X, y, cv=5)
print(f"Simple model: {scores_simple.mean():.3f} (+/- {scores_simple.std():.3f})")

# High variance (overfitting): complex model
from sklearn.tree import DecisionTreeRegressor
complex_model = DecisionTreeRegressor(max_depth=None)
scores_complex = cross_val_score(complex_model, X, y, cv=5)
print(f"Complex model: {scores_complex.mean():.3f} (+/- {scores_complex.std():.3f})")

# Balanced: regularized model
from sklearn.ensemble import RandomForestRegressor
balanced = RandomForestRegressor(max_depth=5, n_estimators=100)
scores_balanced = cross_val_score(balanced, X, y, cv=5)
print(f"Balanced: {scores_balanced.mean():.3f} (+/- {scores_balanced.std():.3f})")''',
        output_func=show_bias_variance_intro,
        concept_title="⚖️ The Tradeoff",
        output_title="Bias vs Variance Curves"
    )

    # Row 2: Model Complexity
    def show_complexity():
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2 * x + 5 + np.random.normal(0, 3, 20)
        x_smooth = np.linspace(0, 10, 200)
        
        # Three models
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#22d3a7', size=8), name='Data'))
        
        # Simple (underfit)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.full_like(x_smooth, y.mean()), mode='lines', 
                                line=dict(color='#f5b731', width=2), name='Simple (High Bias)'))
        
        # Good fit
        z = np.polyfit(x, y, 1)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z, x_smooth), mode='lines', 
                                line=dict(color='#7c6aff', width=3), name='Balanced'))
        
        # Complex (overfit)
        z_over = np.polyfit(x, y, 12)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z_over, x_smooth).clip(-5, 35), mode='lines', 
                                line=dict(color='#f45d6d', width=2), name='Complex (High Variance)'))
        
        fig.update_layout(height=280, title="Model Complexity Comparison", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why does complexity matter?</b>
        <br><br>Look at the chart on the right. The data has a clear upward trend (y ≈ 2x + 5).
        <br><br><b>🟡 Yellow line (High Bias):</b> A horizontal line at the mean. It's SO simple it misses the obvious upward trend. This is underfitting.
        <br><br><b>🟣 Purple line (Balanced):</b> A straight line that captures the trend. It doesn't hit every point, but it generalizes well.
        <br><br><b>🔴 Red line (High Variance):</b> A wiggly polynomial that passes through every training point. Looks great on training data! But those wiggles are just noise — on new data, it will fail badly.
        <br><br><b>The key insight:</b> The wiggly model has ZERO training error but will have HIGH test error. It memorized the noise.
        </div>
        <div class="math-box">
        <b>📐 How to Diagnose:</b>
        <br><br><b>High Bias (Underfitting):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Train accuracy: 65%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Test accuracy: 63%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Both low → model too simple
        <br>&nbsp;&nbsp;&nbsp;&nbsp;<b>Fix:</b> Add features, use complex model
        <br><br><b>High Variance (Overfitting):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Train accuracy: 98%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Test accuracy: 72%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Big gap → model memorized training data
        <br>&nbsp;&nbsp;&nbsp;&nbsp;<b>Fix:</b> More data, regularization, simpler model
        <br><br><b>Good Fit:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Train accuracy: 85%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Test accuracy: 83%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Both high, small gap → sweet spot!
        </div>""",
        code_str='''# Diagnose bias vs variance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train: {train_score:.3f}")
print(f"Test: {test_score:.3f}")

# Interpretation:
# Both low → High bias (underfit)
# Train high, test low → High variance (overfit)
# Both high and similar → Good fit!''',
        output_func=show_complexity,
        concept_title="📊 Model Complexity",
        output_title="Visual Comparison"
    )

    # Row 3: Regularization
    def show_regularization():
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2 * x + 5 + np.random.normal(0, 3, 20)
        x_smooth = np.linspace(0, 10, 200)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#22d3a7', size=8), name='Data'))
        
        # No regularization (overfit)
        z_over = np.polyfit(x, y, 8)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z_over, x_smooth).clip(-5, 35), mode='lines', 
                                line=dict(color='#f45d6d', width=2, dash='dash'), name='No Regularization'))
        
        # With regularization (good fit)
        z = np.polyfit(x, y, 1)
        fig.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z, x_smooth), mode='lines', 
                                line=dict(color='#7c6aff', width=3), name='With Regularization'))
        
        fig.update_layout(height=280, title="Effect of Regularization", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Regularization?</b>
        <br><br>Regularization is a technique to <b>prevent overfitting by penalizing complexity</b>. It adds a "cost" for having large coefficients.
        <br><br><b>The intuition:</b> Large coefficients mean the model is relying heavily on specific features — often a sign of overfitting. Regularization says: "You can use big coefficients, but you'll pay a price."
        <br><br><b>L1 (Lasso) — "The Feature Selector":</b>
        <br>• Adds |coefficients| to the loss function
        <br>• Drives some coefficients to EXACTLY zero
        <br>• Result: Automatic feature selection!
        <br>• Use when: You suspect many features are irrelevant
        <br><br><b>L2 (Ridge) — "The Shrinker":</b>
        <br>• Adds coefficients² to the loss function
        <br>• Shrinks ALL coefficients toward zero (but never exactly zero)
        <br>• Result: All features kept, but with smaller weights
        <br>• Use when: You believe all features matter somewhat
        <br><br><b>λ (lambda):</b> Controls regularization strength
        <br>• λ = 0 → No regularization (might overfit)
        <br>• λ = ∞ → All coefficients → 0 (will underfit)
        </div>
        <div class="math-box">
        <b>📐 Regularization — Numerical Example:</b>
        <br><br><b>Without regularization:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Coefficients: [2.5, -1.8, 15.2, -12.4]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Large values → overfitting
        <br><br><b>L2 Ridge (λ=1.0):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Loss = MSE + 1.0 × (2.5² + 1.8² + 15.2² + 12.4²)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Loss = MSE + 1.0 × 396.5
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Result: [1.2, -0.9, 3.1, -2.8] (shrunk)
        <br><br><b>L1 Lasso (λ=1.0):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Result: [1.1, 0, 2.8, 0] (some = 0!)
        <br><br>🧠 L1 does feature selection automatically
        </div>""",
        code_str='''from sklearn.linear_model import Ridge, Lasso

# L2 Regularization (Ridge)
ridge = Ridge(alpha=1.0)  # alpha = lambda
ridge.fit(X_train, y_train)
print(f"Ridge coefficients: {ridge.coef_}")

# L1 Regularization (Lasso) - feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Lasso coefficients: {lasso.coef_}")
# Note: Some coefficients become exactly 0!

# Elastic Net (L1 + L2)
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)''',
        output_func=show_regularization,
        concept_title="🎛️ Regularization",
        output_title="Control Complexity"
    )

    iq([
        {"q": "Explain the bias-variance tradeoff.", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "<b>Bias:</b> Error from wrong assumptions (underfitting). <b>Variance:</b> Error from sensitivity to training data (overfitting). <b>Tradeoff:</b> Simple models have high bias, low variance. Complex models have low bias, high variance. Goal: minimize total error.",
         "t": "Use the dartboard analogy — bias is accuracy, variance is precision."},
        {"q": "What is regularization and why is it used?", "d": "Medium", "c": ["Google", "Amazon"],
         "a": "Regularization penalizes model complexity to prevent overfitting. <b>L1 (Lasso):</b> Adds |coefficients| — drives some to 0 (feature selection). <b>L2 (Ridge):</b> Adds coefficients² — shrinks all toward 0. <b>λ</b> controls strength.",
         "t": "Mention that L1 is good for feature selection, L2 for when all features matter."},
    ])

    # Summary: Strategies to Avoid Overfitting/Underfitting
    st.markdown("---")
    st.markdown("### 🎯 Quick Reference: Fixing Overfitting & Underfitting")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="warn-box">
        <b>🔴 OVERFITTING (High Variance)</b>
        <br><br><b>Symptoms:</b>
        <br>• Train accuracy >> Test accuracy
        <br>• Model is too complex
        <br>• Memorized noise in training data
        <br><br><b>Solutions:</b>
        <br>• Get more training data
        <br>• Reduce model complexity (fewer features, shallower trees)
        <br>• Add regularization (L1/L2)
        <br>• Use cross-validation
        <br>• Early stopping (for boosting/neural nets)
        <br>• Dropout (for neural nets)
        <br>• Use ensemble methods (bagging)
        <br>• Increase min_samples_leaf / min_samples_split
        <br>• Decrease max_depth
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="insight-box">
        <b>🟢 UNDERFITTING (High Bias)</b>
        <br><br><b>Symptoms:</b>
        <br>• Both train & test accuracy are low
        <br>• Model is too simple
        <br>• Missing important patterns
        <br><br><b>Solutions:</b>
        <br>• Use a more complex model
        <br>• Add more features / feature engineering
        <br>• Reduce regularization strength
        <br>• Increase model capacity (deeper trees, more neurons)
        <br>• Train longer (more epochs/iterations)
        <br>• Use polynomial features
        <br>• Try boosting (reduces bias)
        <br>• Increase max_depth
        <br>• Decrease min_samples_leaf
        </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="math-box">
    <b>📐 Model-Specific Tuning Cheat Sheet:</b>
    <br><br><b>Decision Tree:</b> max_depth ↓, min_samples_leaf ↑ → less overfit
    <br><b>Random Forest:</b> n_estimators ↑ (rarely hurts), max_depth ↓ → less overfit
    <br><b>XGBoost:</b> learning_rate ↓, max_depth ↓, early_stopping → less overfit
    <br><b>Logistic Regression:</b> C ↓ (more regularization) → less overfit
    <br><b>K-Means:</b> Use elbow method + silhouette score to find optimal K
    <br><br><b>Golden Rule:</b> Always use cross-validation to tune hyperparameters!
    </div>""", unsafe_allow_html=True)
