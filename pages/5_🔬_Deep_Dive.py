# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="🔬 Deep Dive", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Force black background everywhere */
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
.block-container, [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #000000 !important;
}
.stApp, .stMarkdown, p, span, label { color: #e2e8f0 !important; }
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
.stApp { font-family: 'Inter', sans-serif; }

.dd-card {
    background: linear-gradient(135deg, #1a1d2e, #1f2235);
    border: 1px solid #2d3148; border-radius: 14px; padding: 1.2rem 1.4rem;
    margin: 0.5rem 0; line-height: 1.85; font-size: 0.91rem; color: #c8cfe0 !important;
}
.dd-card b { color: #e2e8f0 !important; }
.dd-step {
    background: #181a27; border: 1px solid #2d3148; border-radius: 14px;
    padding: 1.2rem 1.4rem; margin: 0.5rem 0; border-left: 4px solid;
}
.dd-math {
    background: #252840; border-left: 4px solid #f5b731;
    border-radius: 0 10px 10px 0; padding: 1rem 1.2rem;
    margin: 0.5rem 0; font-size: 0.88rem; color: #c8cfe0 !important; line-height: 1.9;
    font-family: 'Fira Code', monospace;
}
.dd-math b { color: #f5b731 !important; }
.dd-insight {
    background: linear-gradient(135deg, #1a2a1f, #1f3528);
    border: 1px solid #2a5a3a; border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.87rem; color: #c8d8c0 !important; line-height: 1.7;
}
.dd-insight b { color: #d0f0e0 !important; }
.dd-warn {
    background: linear-gradient(135deg, #2a1a1e, #351f25);
    border: 1px solid #5a2a3a; border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.87rem; color: #d8a8b8 !important; line-height: 1.7;
}
.dd-warn b { color: #f0c8d8 !important; }
.node-box {
    display: inline-block; padding: 8px 16px; border-radius: 10px;
    font-weight: 600; font-size: 0.85rem; margin: 3px;
}
</style>
""", unsafe_allow_html=True)

DL = dict(
    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    xaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'), title_font=dict(color='#8892b0')),
    yaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'), title_font=dict(color='#8892b0')),
    font=dict(color='#e2e8f0'), legend=dict(font=dict(color='#8892b0')),
    margin=dict(t=40, b=40, l=40, r=40),
)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🔬 Deep Dive")
    st.caption("Visual step-by-step explainers with math")
    st.divider()
    topic = st.radio("Topic:", [
        "📈 Logistic Regression",
        "🌳 Decision Trees",
        "🌳 Random Forest",
        "🚀 Gradient Boosting & XGBoost",
        "🧩 K-Means Clustering",
    ], label_visibility="collapsed")


# ═══════════════════════════════════════
# LOGISTIC REGRESSION DEEP DIVE
# ═══════════════════════════════════════
if topic == "📈 Logistic Regression":
    st.markdown("# 📈 Logistic Regression: Complete Visual Guide")
    st.caption("Step-by-step visualization with math at every stage")

    st.markdown("""<div class="dd-card">
    <b>Logistic Regression predicts the PROBABILITY of a yes/no outcome.</b>
    <br><br>Despite the name, it's a <b>classification</b> algorithm, not regression.
    <br>It answers: "What's the probability this store is successful?" → 0.82 → ✅ Yes (>0.5)
    <br><br>🔹 Uses the <b>Sigmoid function</b> to squash any number into [0, 1]
    <br>🔹 Learns <b>weights</b> for each feature via gradient descent
    <br>🔹 Decision boundary at <b>0.5</b> (customizable)
    </div>""", unsafe_allow_html=True)

    # ── STEP 1: THE PROBLEM ──
    st.divider()
    st.markdown("## Step 1: The Problem — Why Not Linear Regression?")

    x_vals = np.linspace(1, 5, 50)
    y_linear = -0.8 + 0.4 * x_vals
    y_sigmoid = 1 / (1 + np.exp(-(- 6 + 1.8 * x_vals)))
    y_actual = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    col1, col2 = st.columns(2)
    with col1:
        fig_bad = go.Figure()
        fig_bad.add_trace(go.Scatter(x=x_vals, y=y_actual, mode='markers', marker=dict(color='#22d3a7', size=8), name='Actual'))
        fig_bad.add_trace(go.Scatter(x=x_vals, y=y_linear, mode='lines', line=dict(color='#f45d6d', width=2, dash='dash'), name='Linear'))
        fig_bad.add_hline(y=0, line_dash="dot", line_color="#4a4e6a")
        fig_bad.add_hline(y=1, line_dash="dot", line_color="#4a4e6a")
        fig_bad.update_layout(height=250, title="❌ Linear Regression: Predicts outside [0,1]!", **DL)
        st.plotly_chart(fig_bad, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig_good = go.Figure()
        fig_good.add_trace(go.Scatter(x=x_vals, y=y_actual, mode='markers', marker=dict(color='#22d3a7', size=8), name='Actual'))
        fig_good.add_trace(go.Scatter(x=x_vals, y=y_sigmoid, mode='lines', line=dict(color='#7c6aff', width=3), name='Logistic'))
        fig_good.add_hline(y=0.5, line_dash="dash", line_color="#f5b731", annotation_text="Decision boundary: 0.5")
        fig_good.update_layout(height=250, title="✅ Logistic Regression: Always between [0,1]", **DL)
        st.plotly_chart(fig_good, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-insight">
    💡 Linear regression can predict -0.3 or 1.5 for a probability — that's nonsense!
    Logistic regression uses the <b>sigmoid function</b> to keep predictions between 0 and 1.
    </div>""", unsafe_allow_html=True)

    # ── STEP 2: SIGMOID ──
    st.divider()
    st.markdown("## Step 2: The Sigmoid Function")

    z = np.linspace(-8, 8, 200)
    sig = 1 / (1 + np.exp(-z))

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=z, y=sig, mode='lines', line=dict(color='#7c6aff', width=3), name='σ(z)'))
    fig_sig.add_hline(y=0.5, line_dash="dash", line_color="#f5b731")
    fig_sig.add_vline(x=0, line_dash="dot", line_color="#4a4e6a")
    fig_sig.add_annotation(x=4, y=0.9, text="z > 0 → P > 0.5 → ✅", showarrow=False, font=dict(color='#22d3a7'))
    fig_sig.add_annotation(x=-4, y=0.1, text="z < 0 → P < 0.5 → ❌", showarrow=False, font=dict(color='#f45d6d'))
    fig_sig.update_layout(height=280, title="Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ)", xaxis_title="z (linear combination)", yaxis_title="P(success)", **DL)
    st.plotly_chart(fig_sig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Sigmoid Function:</b>
    <br><br><b>σ(z) = 1 / (1 + e⁻ᶻ)</b>
    <br><br>Where z = β₀ + β₁x₁ + β₂x₂ + ... (linear combination of features)
    <br><br><b>Properties:</b>
    <br>&nbsp;&nbsp;• z = 0 → σ(0) = 1/(1+1) = <b>0.5</b> (50/50)
    <br>&nbsp;&nbsp;• z = +∞ → σ(∞) = 1/(1+0) = <b>1.0</b> (certain yes)
    <br>&nbsp;&nbsp;• z = -∞ → σ(-∞) = 1/(1+∞) = <b>0.0</b> (certain no)
    <br>&nbsp;&nbsp;• z = 2 → σ(2) = 1/(1+e⁻²) = 1/(1+0.135) = <b>0.88</b>
    <br>&nbsp;&nbsp;• z = -2 → σ(-2) = 1/(1+e²) = 1/(1+7.389) = <b>0.12</b>
    </div>""", unsafe_allow_html=True)

    # ── STEP 3: EXAMPLE ──
    st.divider()
    st.markdown("## Step 3: Prediction — Step by Step")

    st.markdown("""<div class="dd-step" style="border-left-color: #7c6aff">
    <b style="color:#7c6aff">🍕 Scenario:</b> Predict if a pizza store is successful.
    <br>Features: Rating (x₁), Delivery_Min (x₂)
    <br>Learned weights: β₀ = -5.0, β₁ = 1.5, β₂ = -0.05
    </div>""", unsafe_allow_html=True)

    st.markdown("#### New Store: Rating = 4.2, Delivery = 25 min")

    st.markdown("""<div class="dd-math">
    <b>📐 Step-by-Step Prediction:</b>
    <br><br><b>Step 1: Calculate z (linear combination)</b>
    <br>&nbsp;&nbsp;z = β₀ + β₁×Rating + β₂×Delivery
    <br>&nbsp;&nbsp;z = -5.0 + 1.5×(4.2) + (-0.05)×(25)
    <br>&nbsp;&nbsp;z = -5.0 + 6.3 - 1.25
    <br>&nbsp;&nbsp;z = <b>0.05</b>
    <br><br><b>Step 2: Apply sigmoid</b>
    <br>&nbsp;&nbsp;P = σ(z) = 1 / (1 + e⁻⁰·⁰⁵)
    <br>&nbsp;&nbsp;P = 1 / (1 + 0.951)
    <br>&nbsp;&nbsp;P = 1 / 1.951
    <br>&nbsp;&nbsp;P = <b>0.512 = 51.2%</b>
    <br><br><b>Step 3: Apply decision boundary</b>
    <br>&nbsp;&nbsp;P = 0.512 > 0.5 → <b>✅ Successful</b>
    <br><br>🧠 Barely above the threshold! This store is borderline.
    </div>""", unsafe_allow_html=True)

    mc = st.columns(4)
    mc[0].metric("z (linear)", "0.05")
    mc[1].metric("P(success)", "51.2%")
    mc[2].metric("Threshold", "50%")
    mc[3].metric("Prediction", "✅ Yes")

    # ── STEP 4: COST FUNCTION ──
    st.divider()
    st.markdown("## Step 4: How It Learns — Log Loss")

    st.markdown("""<div class="dd-step" style="border-left-color: #f5b731">
    <b style="color:#f5b731">🎯 Goal:</b> Find weights (β₀, β₁, β₂) that minimize prediction errors.
    <br>Uses <b>Log Loss</b> (Binary Cross-Entropy) — not MSE like linear regression.
    </div>""", unsafe_allow_html=True)

    p_range = np.linspace(0.01, 0.99, 100)
    loss_y1 = -np.log(p_range)
    loss_y0 = -np.log(1 - p_range)

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=p_range, y=loss_y1, mode='lines', line=dict(color='#22d3a7', width=2), name='Actual=1 (success)'))
    fig_loss.add_trace(go.Scatter(x=p_range, y=loss_y0, mode='lines', line=dict(color='#f45d6d', width=2), name='Actual=0 (failure)'))
    fig_loss.update_layout(height=280, title="Log Loss: Penalizes confident wrong predictions heavily",
                           xaxis_title="Predicted Probability", yaxis_title="Loss", **DL)
    st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Log Loss Formula:</b>
    <br><br><b>Loss = -[y × log(p) + (1-y) × log(1-p)]</b>
    <br><br>Where y = actual (0 or 1), p = predicted probability
    <br><br><b>Example 1:</b> Actual = 1 (success), Predicted = 0.9
    <br>&nbsp;&nbsp;Loss = -[1×log(0.9) + 0×log(0.1)]
    <br>&nbsp;&nbsp;Loss = -[-0.105] = <b>0.105</b> ← Low loss (good prediction!)
    <br><br><b>Example 2:</b> Actual = 1 (success), Predicted = 0.1
    <br>&nbsp;&nbsp;Loss = -[1×log(0.1) + 0×log(0.9)]
    <br>&nbsp;&nbsp;Loss = -[-2.303] = <b>2.303</b> ← High loss (terrible prediction!)
    <br><br>🧠 Predicting 0.1 when actual is 1 → 22× more penalty than predicting 0.9!
    </div>""", unsafe_allow_html=True)

    # ── STEP 5: COEFFICIENTS ──
    st.divider()
    st.markdown("## Step 5: Interpreting Coefficients")

    st.markdown("""<div class="dd-math">
    <b>📐 What the weights mean:</b>
    <br><br>Learned: β₀ = -5.0, β₁(Rating) = +1.5, β₂(Delivery) = -0.05
    <br><br><b>β₁ = +1.5 (Rating):</b>
    <br>&nbsp;&nbsp;+1 point in rating → z increases by 1.5
    <br>&nbsp;&nbsp;Odds multiply by e¹·⁵ = <b>4.48×</b>
    <br>&nbsp;&nbsp;"Each extra rating point makes success 4.5× more likely"
    <br><br><b>β₂ = -0.05 (Delivery):</b>
    <br>&nbsp;&nbsp;+1 minute delivery → z decreases by 0.05
    <br>&nbsp;&nbsp;Odds multiply by e⁻⁰·⁰⁵ = <b>0.95×</b>
    <br>&nbsp;&nbsp;"Each extra minute reduces success odds by 5%"
    <br><br><b>Odds Ratio = eᵝ</b>
    <br>&nbsp;&nbsp;eᵝ > 1 → feature increases probability
    <br>&nbsp;&nbsp;eᵝ < 1 → feature decreases probability
    <br>&nbsp;&nbsp;eᵝ = 1 → feature has no effect
    </div>""", unsafe_allow_html=True)

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(y=["Delivery_Min", "Rating"], x=[-0.05, 1.5], orientation='h',
                              marker_color=["#f45d6d", "#22d3a7"],
                              text=["-0.05 (hurts)", "+1.50 (helps)"], textposition="auto"))
    fig_coef.update_layout(height=160, title="Coefficients: Direction & Magnitude", xaxis_title="Weight (β)", **DL)
    st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-warn">
    ⚠️ <b>Interview tip:</b> "How do you interpret logistic regression coefficients?"
    → "A coefficient of 1.5 for Rating means each 1-point increase multiplies the odds of success by e¹·⁵ ≈ 4.5×"
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# DECISION TREES DEEP DIVE
# ═══════════════════════════════════════
elif topic == "🌳 Decision Trees":
    st.markdown("# 🌳 Decision Trees: Complete Visual Guide")
    st.caption("Step-by-step visualization with math at every stage")

    st.markdown("""<div class="dd-card">
    <b>A Decision Tree asks a series of yes/no questions to classify data.</b>
    <br><br>Think of it like a flowchart: "Is rating > 3.5?" → Yes → "Is delivery < 30?" → Yes → ✅ Successful
    <br><br>🔹 <b>Gini Impurity</b> or <b>Entropy</b> decides which question to ask first
    <br>🔹 Greedy algorithm: picks the best split at each step
    <br>🔹 Easy to interpret — you can literally read the rules
    </div>""", unsafe_allow_html=True)

    data = pd.DataFrame({
        "Store": [f"S{i}" for i in range(1, 9)],
        "Rating": [4.5, 3.2, 4.8, 2.9, 4.1, 3.5, 4.3, 3.0],
        "Delivery_Min": [20, 45, 18, 50, 25, 35, 22, 40],
        "Employees": [10, 5, 12, 4, 8, 6, 9, 5],
        "Successful": ["✅ Yes", "❌ No", "✅ Yes", "❌ No", "✅ Yes", "❌ No", "✅ Yes", "❌ No"],
    })

    # ── GINI vs ENTROPY ──
    st.divider()
    st.markdown("## Step 1: Gini Impurity vs Entropy")

    p_range = np.linspace(0.01, 0.99, 100)
    gini_vals = 2 * p_range * (1 - p_range)
    entropy_vals = -(p_range * np.log2(p_range) + (1-p_range) * np.log2(1-p_range))

    fig_ge = go.Figure()
    fig_ge.add_trace(go.Scatter(x=p_range, y=gini_vals, mode='lines', line=dict(color='#22d3a7', width=3), name='Gini'))
    fig_ge.add_trace(go.Scatter(x=p_range, y=entropy_vals, mode='lines', line=dict(color='#f5b731', width=3), name='Entropy'))
    fig_ge.update_layout(height=280, title="Both peak at P=0.5 (maximum impurity)", xaxis_title="P(class 1)", yaxis_title="Impurity", **DL)
    st.plotly_chart(fig_ge, use_container_width=True, config={"displayModeBar": False})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="dd-math">
        <b>📐 Gini Impurity:</b>
        <br><br>Gini = 1 - Σ(pᵢ²)
        <br><br>For binary: Gini = 1 - p² - (1-p)²
        <br>&nbsp;&nbsp;= 2p(1-p)
        <br><br>Example: 4✅, 4❌
        <br>&nbsp;&nbsp;Gini = 1 - (0.5² + 0.5²) = <b>0.500</b>
        <br><br>Range: 0 (pure) to 0.5 (max impurity)
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="dd-math">
        <b>📐 Entropy:</b>
        <br><br>H = -Σ(pᵢ × log₂(pᵢ))
        <br><br>For binary: H = -p×log₂(p) - (1-p)×log₂(1-p)
        <br><br>Example: 4✅, 4❌
        <br>&nbsp;&nbsp;H = -0.5×log₂(0.5) - 0.5×log₂(0.5)
        <br>&nbsp;&nbsp;H = -0.5×(-1) - 0.5×(-1) = <b>1.000</b>
        <br><br>Range: 0 (pure) to 1.0 (max impurity)
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="dd-insight">
    💡 Gini and Entropy give very similar results. Gini is faster to compute (no logarithm).
    Scikit-learn uses <b>Gini by default</b>.
    </div>""", unsafe_allow_html=True)

    # ── INFORMATION GAIN ──
    st.divider()
    st.markdown("## Step 2: Information Gain — Choosing the Best Split")

    st.markdown("""<div class="dd-math">
    <b>📐 Information Gain = Parent Impurity - Weighted Child Impurity</b>
    <br><br><b>Parent:</b> 4✅, 4❌ → Gini = 0.500
    <br><br><b>Split: Rating > 3.5</b>
    <br>&nbsp;&nbsp;Left (>3.5): 4✅, 1❌ → Gini = 1-(0.8²+0.2²) = 0.320
    <br>&nbsp;&nbsp;Right (≤3.5): 0✅, 3❌ → Gini = 1-(0²+1²) = 0.000
    <br>&nbsp;&nbsp;Weighted = (5/8)×0.320 + (3/8)×0.000 = 0.200
    <br>&nbsp;&nbsp;<b>Gain = 0.500 - 0.200 = 0.300</b>
    <br><br><b>Split: Delivery > 30</b>
    <br>&nbsp;&nbsp;Left (≤30): 4✅, 0❌ → Gini = 0.000
    <br>&nbsp;&nbsp;Right (>30): 0✅, 4❌ → Gini = 0.000
    <br>&nbsp;&nbsp;Weighted = 0.000
    <br>&nbsp;&nbsp;<b>Gain = 0.500 - 0.000 = 0.500</b> ← Perfect split!
    </div>""", unsafe_allow_html=True)

    fig_gain = go.Figure()
    fig_gain.add_trace(go.Bar(x=["Delivery > 30", "Rating > 3.5", "Employees > 7"],
                              y=[0.500, 0.300, 0.125],
                              marker_color=["#22d3a7", "#7c6aff", "#7c6aff"],
                              text=["0.500 ✅", "0.300", "0.125"], textposition="auto"))
    fig_gain.update_layout(height=220, title="Information Gain per Split", yaxis_title="Gain", **DL)
    st.plotly_chart(fig_gain, use_container_width=True, config={"displayModeBar": False})

    # ── TREE VISUAL ──
    st.divider()
    st.markdown("## Step 3: The Complete Tree")

    def draw_tree(nodes, edges, title):
        fig = go.Figure()
        for e in edges:
            x0, y0 = nodes[e[0]][:2]
            x1, y1 = nodes[e[1]][:2]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                     line=dict(color='#4a4e6a', width=2), showlegend=False))
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=e[2], showarrow=False,
                              font=dict(size=11, color='#8892b0'))
        for key, (x, y, label, color) in nodes.items():
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                                     marker=dict(size=50, color=color, line=dict(color='#2d3148', width=2)),
                                     text=[label], textposition='middle center',
                                     textfont=dict(size=10, color='white'), showlegend=False))
        fig.update_layout(height=350, title=title,
                         xaxis=dict(visible=False, range=[-0.1, 1.1]),
                         yaxis=dict(visible=False, range=[-0.1, 1.1]),
                         paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                         font=dict(color='#e2e8f0'), margin=dict(t=40, b=10, l=10, r=10))
        return fig

    nodes = {
        "A": (0.5, 0.92, "Delivery\n>30?", "#7c6aff"),
        "B": (0.25, 0.52, "✅ Yes\n4 stores", "#22d3a7"),
        "C": (0.75, 0.52, "Rating\n>3.2?", "#5eaeff"),
        "D": (0.6, 0.12, "❌ No\n2 stores", "#f45d6d"),
        "E": (0.9, 0.12, "❌ No\n2 stores", "#f45d6d"),
    }
    edges = [("A","B","≤30"), ("A","C",">30"), ("C","D","≤3.2"), ("C","E",">3.2")]
    st.plotly_chart(draw_tree(nodes, edges, "Decision Tree: Pizza Store Success"), use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-insight">
    💡 <b>Reading the tree:</b> Start at the top. Follow the branch that matches your data.
    <br>New store: Delivery=28, Rating=4.2 → Delivery ≤ 30? YES → <b>✅ Successful</b>
    </div>""", unsafe_allow_html=True)

    # ── OVERFITTING ──
    st.divider()
    st.markdown("## Step 4: The Overfitting Problem")

    depths = [1, 2, 3, 5, 10, 20]
    train_acc = [0.70, 0.80, 0.88, 0.95, 0.99, 1.00]
    test_acc = [0.68, 0.78, 0.85, 0.82, 0.75, 0.65]

    fig_of = go.Figure()
    fig_of.add_trace(go.Scatter(x=depths, y=train_acc, mode='lines+markers', name='Train', line=dict(color='#22d3a7', width=2)))
    fig_of.add_trace(go.Scatter(x=depths, y=test_acc, mode='lines+markers', name='Test', line=dict(color='#f45d6d', width=2)))
    fig_of.add_vline(x=3, line_dash="dash", line_color="#f5b731", annotation_text="Sweet spot")
    fig_of.update_layout(height=280, title="Deeper Tree = More Overfitting", xaxis_title="Max Depth", yaxis_title="Accuracy", **DL)
    st.plotly_chart(fig_of, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-warn">
    ⚠️ <b>Single decision trees overfit easily.</b> That's why we use Random Forest (many trees)
    or pruning (limiting depth). This is the #1 weakness of decision trees.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# RANDOM FOREST DEEP DIVE
# ═══════════════════════════════════════
elif topic == "🌳 Random Forest":
    st.markdown("# 🌳 Random Forest: Complete Visual Guide")
    st.caption("Step-by-step visualization with math at every stage")

    st.markdown("""<div class="dd-card">
    <b>Random Forest = Many Decision Trees voting together.</b>
    <br><br>One tree can be wrong. But if 100 trees vote, the majority is usually right.
    <br><br>It combines two ideas:
    <br>🔹 <b>Bagging:</b> Train each tree on a different random sample of data
    <br>🔹 <b>Random Features:</b> Each tree considers different features at each split
    <br><br>Result: Diverse trees that make different mistakes → errors cancel out!
    </div>""", unsafe_allow_html=True)

    # ── Sample Data ──
    st.divider()
    st.markdown("### 📊 Our Pizza Store Data")

    np.random.seed(42)
    data = pd.DataFrame({
        "Store": [f"S{i}" for i in range(1, 9)],
        "Rating": [4.5, 3.2, 4.8, 2.9, 4.1, 3.5, 4.3, 3.0],
        "Delivery_Min": [20, 45, 18, 50, 25, 35, 22, 40],
        "Employees": [10, 5, 12, 4, 8, 6, 9, 5],
        "Successful": ["✅ Yes", "❌ No", "✅ Yes", "❌ No", "✅ Yes", "❌ No", "✅ Yes", "❌ No"],
    })
    st.dataframe(data, use_container_width=True, hide_index=True)

    st.markdown("""<div class="dd-insight">
    💡 <b>Goal:</b> Predict if a new store will be successful based on Rating, Delivery Time, and Employees.
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # STEP 1: BOOTSTRAP SAMPLING
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Step 1: Bootstrap Sampling (Bagging)")

    st.markdown("""<div class="dd-step" style="border-left-color: #7c6aff">
    <b style="color:#7c6aff">🎲 What happens:</b> Each tree gets its own random copy of the data, sampled WITH replacement.
    <br>Some rows appear multiple times, some don't appear at all.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    np.random.seed(42)
    for i, col in enumerate([col1, col2, col3]):
        sample_idx = np.random.choice(8, size=8, replace=True)
        sample = data.iloc[sample_idx][["Store", "Rating", "Successful"]].reset_index(drop=True)
        with col:
            st.markdown(f"**🌳 Tree {i+1} Sample**")
            st.dataframe(sample, use_container_width=True, hide_index=True, height=200)
            unique = len(set(sample_idx))
            st.caption(f"Uses {unique}/8 unique stores ({unique/8*100:.0f}%)")

    st.markdown("""<div class="dd-math">
    <b>📐 Bootstrap Math:</b>
    <br><br>Each sample draws n=8 rows WITH replacement from 8 original rows.
    <br><br>P(specific row NOT picked in one draw) = 1 - 1/n = 7/8 = 0.875
    <br>P(specific row NOT in entire sample) = (7/8)⁸ = 0.344 = 34.4%
    <br>P(specific row IS in sample) = 1 - 0.344 = <b>63.2%</b>
    <br><br>~63% of rows appear in each sample. The other ~37% are "Out-of-Bag" (OOB) → free validation data!
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # STEP 2: RANDOM FEATURE SELECTION
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Step 2: Random Feature Selection")

    st.markdown("""<div class="dd-step" style="border-left-color: #22d3a7">
    <b style="color:#22d3a7">🎯 What happens:</b> At EACH split, each tree only considers a random SUBSET of features.
    <br>This forces trees to be different from each other!
    </div>""", unsafe_allow_html=True)

    features = ["Rating", "Delivery_Min", "Employees"]
    np.random.seed(42)

    cols = st.columns(3)
    for i, c in enumerate(cols):
        selected = np.random.choice(features, size=2, replace=False)
        with c:
            st.markdown(f"**🌳 Tree {i+1}, Split 1**")
            for f in features:
                if f in selected:
                    st.markdown(f"✅ {f}")
                else:
                    st.markdown(f"❌ ~~{f}~~")

    st.markdown("""<div class="dd-math">
    <b>📐 How many features per split?</b>
    <br><br><b>Classification:</b> max_features = √(total features) = √3 = 1.73 ≈ <b>2 features</b>
    <br><b>Regression:</b> max_features = total / 3 = 3/3 = <b>1 feature</b>
    <br><br>Why? If all trees use the same "best" feature first, they'd all be identical.
    <br>Random selection → diverse trees → better ensemble!
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # STEP 3: GINI IMPURITY
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Step 3: Finding the Best Split (Gini Impurity)")

    st.markdown("""<div class="dd-step" style="border-left-color: #f5b731">
    <b style="color:#f5b731">🔍 What happens:</b> At each node, the tree tries every possible split and picks the one that
    separates the classes best. "Best" = lowest Gini impurity after splitting.
    </div>""", unsafe_allow_html=True)

    # Visual: Parent node
    st.markdown("#### Parent Node: All 8 stores")

    fig_parent = go.Figure()
    fig_parent.add_trace(go.Bar(x=["✅ Successful", "❌ Not Successful"], y=[4, 4],
                                marker_color=["#22d3a7", "#f45d6d"], text=["4", "4"], textposition="auto"))
    fig_parent.update_layout(height=200, title="Parent: 4 ✅ + 4 ❌ = Gini 0.500 (maximum impurity)", **DL)
    st.plotly_chart(fig_parent, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Gini Impurity Formula: Gini = 1 - Σ(pᵢ²)</b>
    <br><br><b>Parent Node:</b> 4 Successful, 4 Not Successful (8 total)
    <br>&nbsp;&nbsp;P(✅) = 4/8 = 0.5
    <br>&nbsp;&nbsp;P(❌) = 4/8 = 0.5
    <br>&nbsp;&nbsp;Gini = 1 - (0.5² + 0.5²)
    <br>&nbsp;&nbsp;Gini = 1 - (0.25 + 0.25)
    <br>&nbsp;&nbsp;Gini = 1 - 0.50 = <b>0.500</b> ← Maximum impurity (50/50 split)
    <br><br>Gini = 0 means pure (all same class). Gini = 0.5 means maximum mix.
    </div>""", unsafe_allow_html=True)

    # Visual: Try split Rating > 3.5
    st.markdown("#### Try Split: Rating > 3.5")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_left = go.Figure()
        fig_left.add_trace(go.Bar(x=["✅", "❌"], y=[4, 1],
                                  marker_color=["#22d3a7", "#f45d6d"], text=["4", "1"], textposition="auto"))
        fig_left.update_layout(height=200, title="Left: Rating > 3.5 → Gini = 0.320", **DL)
        st.plotly_chart(fig_left, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        fig_right = go.Figure()
        fig_right.add_trace(go.Bar(x=["✅", "❌"], y=[0, 3],
                                   marker_color=["#22d3a7", "#f45d6d"], text=["0", "3"], textposition="auto"))
        fig_right.update_layout(height=200, title="Right: Rating ≤ 3.5 → Gini = 0.000 (pure!)", **DL)
        st.plotly_chart(fig_right, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Gini After Split: Rating > 3.5</b>
    <br><br><b>Left Child</b> (Rating > 3.5): S1✅, S3✅, S5✅, S7✅, S6❌ → 4 ✅, 1 ❌
    <br>&nbsp;&nbsp;P(✅) = 4/5 = 0.8, P(❌) = 1/5 = 0.2
    <br>&nbsp;&nbsp;Gini_left = 1 - (0.8² + 0.2²) = 1 - (0.64 + 0.04) = <b>0.320</b>
    <br><br><b>Right Child</b> (Rating ≤ 3.5): S2❌, S4❌, S8❌ → 0 ✅, 3 ❌
    <br>&nbsp;&nbsp;P(✅) = 0, P(❌) = 1.0
    <br>&nbsp;&nbsp;Gini_right = 1 - (0² + 1²) = <b>0.000</b> ← Pure node!
    <br><br><b>Weighted Gini:</b>
    <br>&nbsp;&nbsp;= (5/8) × 0.320 + (3/8) × 0.000
    <br>&nbsp;&nbsp;= 0.200 + 0.000 = <b>0.200</b>
    <br><br><b>Information Gain = Parent Gini - Weighted Child Gini</b>
    <br>&nbsp;&nbsp;= 0.500 - 0.200 = <b>0.300</b> ← Higher = better split!
    </div>""", unsafe_allow_html=True)

    # Compare splits
    st.markdown("#### Compare All Possible Splits")

    splits_df = pd.DataFrame({
        "Split": ["Rating > 3.5", "Delivery_Min > 30", "Employees > 7"],
        "Weighted Gini": [0.200, 0.278, 0.375],
        "Information Gain": [0.300, 0.222, 0.125],
        "Winner?": ["✅ Best", "", ""],
    })
    st.dataframe(splits_df, use_container_width=True, hide_index=True)

    fig_splits = go.Figure()
    fig_splits.add_trace(go.Bar(x=["Rating > 3.5", "Delivery > 30", "Employees > 7"],
                                y=[0.300, 0.222, 0.125],
                                marker_color=["#22d3a7", "#7c6aff", "#7c6aff"],
                                text=["0.300 ✅", "0.222", "0.125"], textposition="auto"))
    fig_splits.update_layout(height=220, title="Information Gain: Higher = Better Split", yaxis_title="Info Gain", **DL)
    st.plotly_chart(fig_splits, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-insight">
    💡 <b>Rating > 3.5</b> wins because it creates the purest child nodes (highest information gain).
    The tree picks this as its first split.
    </div>""", unsafe_allow_html=True)


    # ═══════════════════════════════════════
    # STEP 4: COMPLETE TREES
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Step 4: Complete Decision Trees")

    st.markdown("""<div class="dd-step" style="border-left-color: #e879a8">
    <b style="color:#e879a8">🌳 What happens:</b> Each tree keeps splitting until nodes are pure or a stopping condition is met.
    Each tree looks DIFFERENT because of different data samples and random features.
    </div>""", unsafe_allow_html=True)

    # Tree visualizations using Plotly scatter
    def draw_tree(nodes, edges, title):
        fig = go.Figure()
        # Draw edges
        for e in edges:
            x0, y0 = nodes[e[0]][:2]
            x1, y1 = nodes[e[1]][:2]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                     line=dict(color='#4a4e6a', width=2), showlegend=False))
            # Edge label
            mid_x, mid_y = (x0+x1)/2, (y0+y1)/2
            fig.add_annotation(x=mid_x, y=mid_y, text=e[2], showarrow=False,
                              font=dict(size=10, color='#8892b0'))
        # Draw nodes
        for key, (x, y, label, color) in nodes.items():
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                                     marker=dict(size=40, color=color, line=dict(color='#2d3148', width=2)),
                                     text=[label], textposition='middle center',
                                     textfont=dict(size=9, color='white'), showlegend=False))
        fig.update_layout(height=300, title=title,
                         xaxis=dict(visible=False, range=[-0.1, 1.1]),
                         yaxis=dict(visible=False, range=[-0.1, 1.1]),
                         paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                         font=dict(color='#e2e8f0'), margin=dict(t=40, b=10, l=10, r=10))
        return fig

    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        nodes1 = {
            "A": (0.5, 0.95, "Rating\n>3.5?", "#7c6aff"),
            "B": (0.25, 0.55, "Deliv\n<30?", "#5eaeff"),
            "C": (0.75, 0.55, "❌ No", "#f45d6d"),
            "D": (0.1, 0.15, "✅ Yes", "#22d3a7"),
            "E": (0.4, 0.15, "❌ No", "#f45d6d"),
        }
        edges1 = [("A","B","Yes"), ("A","C","No"), ("B","D","Yes"), ("B","E","No")]
        st.plotly_chart(draw_tree(nodes1, edges1, "🌳 Tree 1"), use_container_width=True, config={"displayModeBar": False})

    with col_t2:
        nodes2 = {
            "A": (0.5, 0.95, "Empl\n>7?", "#7c6aff"),
            "B": (0.25, 0.55, "✅ Yes", "#22d3a7"),
            "C": (0.75, 0.55, "Deliv\n<35?", "#5eaeff"),
            "D": (0.6, 0.15, "✅ Yes", "#22d3a7"),
            "E": (0.9, 0.15, "❌ No", "#f45d6d"),
        }
        edges2 = [("A","B","Yes"), ("A","C","No"), ("C","D","Yes"), ("C","E","No")]
        st.plotly_chart(draw_tree(nodes2, edges2, "🌳 Tree 2"), use_container_width=True, config={"displayModeBar": False})

    with col_t3:
        nodes3 = {
            "A": (0.5, 0.95, "Deliv\n<25?", "#7c6aff"),
            "B": (0.25, 0.55, "✅ Yes", "#22d3a7"),
            "C": (0.75, 0.55, "Rating\n>4.0?", "#5eaeff"),
            "D": (0.6, 0.15, "✅ Yes", "#22d3a7"),
            "E": (0.9, 0.15, "❌ No", "#f45d6d"),
        }
        edges3 = [("A","B","Yes"), ("A","C","No"), ("C","D","Yes"), ("C","E","No")]
        st.plotly_chart(draw_tree(nodes3, edges3, "🌳 Tree 3"), use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-insight">
    💡 Notice: Each tree uses <b>different features</b> at the root (Rating, Employees, Delivery).
    This diversity is the key to Random Forest's power!
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # STEP 5: MAJORITY VOTE
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Step 5: Prediction — Majority Vote")

    st.markdown("""<div class="dd-step" style="border-left-color: #f5b731">
    <b style="color:#f5b731">🗳️ What happens:</b> A new store comes in. Each tree makes its own prediction.
    The final answer = majority vote (classification) or average (regression).
    </div>""", unsafe_allow_html=True)

    st.markdown("#### New Store: Rating=4.2, Delivery=28min, Employees=7")

    vote_col1, vote_col2, vote_col3 = st.columns(3)

    with vote_col1:
        st.markdown("""<div class="dd-step" style="border-left-color: #22d3a7">
        <b>🌳 Tree 1</b>
        <br>Rating > 3.5? → <b>YES</b> (4.2 > 3.5)
        <br>Delivery < 30? → <b>YES</b> (28 < 30)
        <br><br>🗳️ Vote: <b style="color:#22d3a7">✅ Successful</b>
        </div>""", unsafe_allow_html=True)

    with vote_col2:
        st.markdown("""<div class="dd-step" style="border-left-color: #22d3a7">
        <b>🌳 Tree 2</b>
        <br>Employees > 7? → <b>NO</b> (7 = 7, not >)
        <br>Delivery < 35? → <b>YES</b> (28 < 35)
        <br><br>🗳️ Vote: <b style="color:#22d3a7">✅ Successful</b>
        </div>""", unsafe_allow_html=True)

    with vote_col3:
        st.markdown("""<div class="dd-step" style="border-left-color: #22d3a7">
        <b>🌳 Tree 3</b>
        <br>Delivery < 25? → <b>NO</b> (28 > 25)
        <br>Rating > 4.0? → <b>YES</b> (4.2 > 4.0)
        <br><br>🗳️ Vote: <b style="color:#22d3a7">✅ Successful</b>
        </div>""", unsafe_allow_html=True)

    # Vote result
    fig_vote = go.Figure()
    fig_vote.add_trace(go.Bar(x=["✅ Successful", "❌ Not Successful"], y=[3, 0],
                              marker_color=["#22d3a7", "#f45d6d"],
                              text=["3 votes", "0 votes"], textposition="auto"))
    fig_vote.update_layout(height=200, title="🗳️ Final Vote: 3-0 → ✅ SUCCESSFUL", **DL)
    st.plotly_chart(fig_vote, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Voting Math:</b>
    <br><br><b>Classification (Majority Vote):</b>
    <br>&nbsp;&nbsp;Tree 1: ✅ | Tree 2: ✅ | Tree 3: ✅
    <br>&nbsp;&nbsp;Count(✅) = 3, Count(❌) = 0
    <br>&nbsp;&nbsp;P(Successful) = 3/3 = <b>100%</b>
    <br>&nbsp;&nbsp;Final: ✅ Successful
    <br><br><b>Regression (Average):</b>
    <br>&nbsp;&nbsp;If predicting daily sales:
    <br>&nbsp;&nbsp;Tree 1: $520 | Tree 2: $480 | Tree 3: $510
    <br>&nbsp;&nbsp;Prediction = (520 + 480 + 510) / 3 = <b>$503.33</b>
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # STEP 6: FEATURE IMPORTANCE
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Step 6: Feature Importance")

    st.markdown("""<div class="dd-step" style="border-left-color: #5eaeff">
    <b style="color:#5eaeff">📊 What happens:</b> Random Forest tracks how much each feature reduces Gini impurity
    across ALL splits in ALL trees. More reduction = more important feature.
    </div>""", unsafe_allow_html=True)

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(y=["Employees", "Delivery_Min", "Rating"],
                             x=[0.219, 0.313, 0.469],
                             orientation='h',
                             marker_color=["#7c6aff", "#5eaeff", "#22d3a7"],
                             text=["21.9%", "31.3%", "46.9%"], textposition="auto"))
    fig_imp.update_layout(height=200, title="Feature Importance (sum = 100%)", xaxis_title="Importance", **DL)
    st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Feature Importance Calculation:</b>
    <br><br>For each feature, sum up Gini decrease across ALL splits in ALL trees:
    <br><br><b>Rating:</b>
    <br>&nbsp;&nbsp;Tree 1, Split 1: Gini decrease = 0.300
    <br>&nbsp;&nbsp;Tree 1, Split 3: Gini decrease = 0.100
    <br>&nbsp;&nbsp;Tree 2, Split 2: Gini decrease = 0.150
    <br>&nbsp;&nbsp;Tree 3, Split 2: Gini decrease = 0.200
    <br>&nbsp;&nbsp;Total = 0.750
    <br><br><b>Delivery_Min:</b> Total = 0.500
    <br><b>Employees:</b> Total = 0.350
    <br><br><b>Grand Total = 0.750 + 0.500 + 0.350 = 1.600</b>
    <br><br>Importance(Rating) = 0.750 / 1.600 = <b>46.9%</b>
    <br>Importance(Delivery) = 0.500 / 1.600 = <b>31.3%</b>
    <br>Importance(Employees) = 0.350 / 1.600 = <b>21.9%</b>
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # WHY IT WORKS
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Why Random Forest Works: Variance Reduction")

    st.markdown("""<div class="dd-math">
    <b>📐 The Math Behind the Magic:</b>
    <br><br><b>Single tree error</b> = Bias² + Variance + Noise
    <br><br><b>Random Forest reduces VARIANCE by averaging:</b>
    <br><br>If trees were independent:
    <br>&nbsp;&nbsp;Var(average of N trees) = Var(single tree) / N
    <br><br>But trees aren't fully independent, so:
    <br>&nbsp;&nbsp;<b>Var(RF) = ρ × σ² + (1-ρ)/N × σ²</b>
    <br><br>Where:
    <br>&nbsp;&nbsp;ρ = correlation between trees (lower is better)
    <br>&nbsp;&nbsp;σ² = variance of a single tree
    <br>&nbsp;&nbsp;N = number of trees
    <br><br>Random feature selection → lower ρ → lower variance → better predictions!
    </div>""", unsafe_allow_html=True)

    # Bias-Variance visual
    fig_bv = go.Figure()
    trees_range = [1, 5, 10, 25, 50, 100, 200, 500]
    single_var = 0.15
    rho = 0.3
    variances = [rho * single_var + (1-rho)/n * single_var for n in trees_range]
    bias = [0.05] * len(trees_range)
    total = [b + v for b, v in zip(bias, variances)]

    fig_bv.add_trace(go.Scatter(x=trees_range, y=variances, mode='lines+markers',
                                name='Variance', line=dict(color='#f45d6d', width=2)))
    fig_bv.add_trace(go.Scatter(x=trees_range, y=bias, mode='lines+markers',
                                name='Bias²', line=dict(color='#22d3a7', width=2)))
    fig_bv.add_trace(go.Scatter(x=trees_range, y=total, mode='lines+markers',
                                name='Total Error', line=dict(color='#f5b731', width=2)))
    fig_bv.update_layout(height=280, title="More Trees → Lower Variance → Lower Error",
                         xaxis_title="Number of Trees", yaxis_title="Error", **DL)
    st.plotly_chart(fig_bv, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-insight">
    💡 <b>Key insight:</b> Adding more trees reduces variance but NEVER increases bias.
    That's why Random Forest rarely overfits — more trees is almost always better (just slower).
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Single Tree vs Random Forest")

    comp_df = pd.DataFrame({
        "Aspect": ["Bias", "Variance", "Overfitting", "Interpretability", "Speed", "Accuracy"],
        "Single Tree": ["Low", "HIGH ⚠️", "Prone ⚠️", "Easy ✅", "Fast ✅", "Lower"],
        "Random Forest": ["Low", "LOW ✅", "Resistant ✅", "Harder", "Slower", "Higher ✅"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════
    # HYPERPARAMETERS
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Hyperparameters to Tune")

    params_df = pd.DataFrame({
        "Parameter": ["n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf"],
        "What it does": ["Number of trees", "Max depth per tree", "Features per split", "Min samples to split", "Min samples in leaf"],
        "Default": ["100", "None (unlimited)", "√p (classification)", "2", "1"],
        "Tip": ["100-500, more=better but slower", "5-20 to prevent overfitting", "Lower=more diverse trees", "Higher=simpler trees", "Higher=less overfitting"],
    })
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════
    # PYTHON CODE
    # ═══════════════════════════════════════
    st.divider()
    st.markdown("## Python Code")

    st.code("""from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # 100 trees
    max_depth=5,           # each tree max 5 levels deep
    max_features='sqrt',   # √p features per split
    random_state=42
)

# Train & Predict
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Feature Importance
for feat, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")""", language="python")

    st.markdown("""<div class="dd-warn">
    ⚠️ <b>Interview tip:</b> "Explain Random Forest" is one of the most common ML interview questions.
    Walk through: Bagging → Random Features → Gini splits → Majority Vote → Feature Importance.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# GRADIENT BOOSTING & XGBOOST DEEP DIVE
# ═══════════════════════════════════════
elif topic == "🚀 Gradient Boosting & XGBoost":
    st.markdown("# 🚀 Gradient Boosting & XGBoost: Complete Visual Guide")
    st.caption("Step-by-step visualization with math at every stage")

    st.markdown("""<div class="dd-card">
    <b>Boosting = Build trees SEQUENTIALLY, each one fixing the previous tree's mistakes.</b>
    <br><br>Unlike Random Forest (parallel, independent trees), boosting trees are <b>dependent</b> — each new tree
    learns from the errors of all previous trees combined.
    <br><br>🔹 <b>AdaBoost:</b> Reweights misclassified samples
    <br>🔹 <b>Gradient Boosting:</b> Fits new trees to the RESIDUALS (errors)
    <br>🔹 <b>XGBoost:</b> Optimized gradient boosting with regularization
    </div>""", unsafe_allow_html=True)

    # ── BAGGING vs BOOSTING ──
    st.divider()
    st.markdown("## Step 1: Bagging vs Boosting")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="dd-step" style="border-left-color: #7c6aff">
        <b style="color:#7c6aff">🌳 Random Forest (Bagging)</b>
        <br><br>• Trees built <b>in parallel</b> (independently)
        <br>• Each tree sees random data + random features
        <br>• Final: <b>majority vote</b>
        <br>• Reduces <b>variance</b>
        <br>• Hard to overfit
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="dd-step" style="border-left-color: #22d3a7">
        <b style="color:#22d3a7">🚀 Gradient Boosting</b>
        <br><br>• Trees built <b>sequentially</b> (each depends on previous)
        <br>• Each tree fixes previous errors
        <br>• Final: <b>weighted sum</b>
        <br>• Reduces <b>bias</b>
        <br>• CAN overfit (needs tuning)
        </div>""", unsafe_allow_html=True)

    # ── HOW GRADIENT BOOSTING WORKS ──
    st.divider()
    st.markdown("## Step 2: Gradient Boosting — Learning from Mistakes")

    st.markdown("""<div class="dd-step" style="border-left-color: #f5b731">
    <b style="color:#f5b731">🎯 The idea:</b> Start with a simple prediction. Calculate errors (residuals).
    Build a new tree to predict those errors. Add it to the model. Repeat.
    </div>""", unsafe_allow_html=True)

    # Visual: Iterative improvement
    stores = ["S1", "S2", "S3", "S4", "S5"]
    actual = [500, 300, 600, 200, 450]
    round0 = [410, 410, 410, 410, 410]  # mean
    resid1 = [90, -110, 190, -210, 40]
    round1_pred = [90, -80, 150, -180, 30]  # tree1 predictions of residuals
    after_r1 = [410+90*0.1, 410-80*0.1, 410+150*0.1, 410-180*0.1, 410+30*0.1]

    iter_df = pd.DataFrame({
        "Store": stores,
        "Actual Sales": actual,
        "Round 0 (mean)": round0,
        "Residual 1": resid1,
        "Tree 1 Predicts": round1_pred,
        "After Round 1": [f"${v:.0f}" for v in after_r1],
    })
    st.dataframe(iter_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class="dd-math">
    <b>📐 Gradient Boosting — Step by Step:</b>
    <br><br><b>Round 0: Start with mean</b>
    <br>&nbsp;&nbsp;F₀(x) = mean(y) = (500+300+600+200+450)/5 = <b>$410</b>
    <br><br><b>Round 1: Calculate residuals</b>
    <br>&nbsp;&nbsp;r₁ = actual - F₀(x)
    <br>&nbsp;&nbsp;S1: 500 - 410 = +90 (underpredicted by $90)
    <br>&nbsp;&nbsp;S2: 300 - 410 = -110 (overpredicted by $110)
    <br>&nbsp;&nbsp;S3: 600 - 410 = +190
    <br>&nbsp;&nbsp;S4: 200 - 410 = -210
    <br><br><b>Round 1: Fit Tree 1 to residuals</b>
    <br>&nbsp;&nbsp;Tree 1 learns to predict the errors
    <br><br><b>Round 1: Update model</b>
    <br>&nbsp;&nbsp;F₁(x) = F₀(x) + η × Tree₁(x)
    <br>&nbsp;&nbsp;where η = learning rate (e.g., 0.1)
    <br>&nbsp;&nbsp;S1: 410 + 0.1 × 90 = <b>$419</b> (closer to $500!)
    <br><br><b>Repeat for 100+ rounds...</b>
    </div>""", unsafe_allow_html=True)

    # Convergence chart
    np.random.seed(42)
    rounds = list(range(0, 101, 5))
    errors = [100 * np.exp(-0.03 * r) + 5 + np.random.normal(0, 2) for r in rounds]

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=rounds, y=errors, mode='lines+markers',
                                  line=dict(color='#22d3a7', width=2), marker=dict(size=5)))
    fig_conv.update_layout(height=250, title="Error Decreases with Each Round",
                           xaxis_title="Boosting Round", yaxis_title="Error (MSE)", **DL)
    st.plotly_chart(fig_conv, use_container_width=True, config={"displayModeBar": False})

    # ── LEARNING RATE ──
    st.divider()
    st.markdown("## Step 3: Learning Rate — The Speed Dial")

    lr_vals = [1.0, 0.3, 0.1, 0.01]
    fig_lr = go.Figure()
    colors = ['#f45d6d', '#f5b731', '#22d3a7', '#7c6aff']
    for lr, color in zip(lr_vals, colors):
        errs = [100 * np.exp(-lr * 0.3 * r) + 5 for r in range(50)]
        fig_lr.add_trace(go.Scatter(x=list(range(50)), y=errs, mode='lines',
                                    line=dict(color=color, width=2), name=f'η={lr}'))
    fig_lr.update_layout(height=280, title="Learning Rate: Slower = More Stable",
                         xaxis_title="Rounds", yaxis_title="Error", **DL)
    st.plotly_chart(fig_lr, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Learning Rate (η):</b>
    <br><br>F(x) = F(x) + <b>η</b> × new_tree(x)
    <br><br>η = 1.0 → Full correction each step → fast but overshoots → <b>overfits</b>
    <br>η = 0.1 → 10% correction each step → slow but stable → <b>better generalization</b>
    <br>η = 0.01 → 1% correction → very slow → needs many trees
    <br><br><b>Rule of thumb:</b> η = 0.1 with 100-1000 trees works well.
    <br>Lower η + more trees = better results (but slower training).
    </div>""", unsafe_allow_html=True)

    # ── XGBOOST ──
    st.divider()
    st.markdown("## Step 4: XGBoost — The Competition Winner")

    st.markdown("""<div class="dd-step" style="border-left-color: #f5b731">
    <b style="color:#f5b731">🏆 XGBoost = Gradient Boosting + Optimizations</b>
    </div>""", unsafe_allow_html=True)

    xg_df = pd.DataFrame({
        "Feature": ["Regularization (L1/L2)", "Tree Pruning", "Parallel Processing", "Missing Values", "Built-in CV"],
        "Gradient Boosting": ["❌ None", "❌ No", "❌ Sequential only", "❌ Must handle manually", "❌ No"],
        "XGBoost": ["✅ Prevents overfitting", "✅ Prunes weak branches", "✅ Parallel tree building", "✅ Handles automatically", "✅ Built-in"],
    })
    st.dataframe(xg_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class="dd-math">
    <b>📐 XGBoost Objective:</b>
    <br><br><b>Obj = Σ Loss(yᵢ, ŷᵢ) + Σ Ω(tree_k)</b>
    <br><br>Where:
    <br>&nbsp;&nbsp;Loss = how well the model fits (e.g., MSE, Log Loss)
    <br>&nbsp;&nbsp;Ω = regularization penalty = γT + ½λΣw²
    <br>&nbsp;&nbsp;T = number of leaves (penalizes complex trees)
    <br>&nbsp;&nbsp;w = leaf weights (penalizes large predictions)
    <br><br>🧠 The regularization term is what makes XGBoost resist overfitting!
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="dd-warn">
    ⚠️ <b>Interview tip:</b> "Why XGBoost over Random Forest?"
    → "XGBoost reduces bias through sequential learning, has built-in regularization,
    handles missing values, and typically achieves higher accuracy on structured data."
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# K-MEANS CLUSTERING DEEP DIVE
# ═══════════════════════════════════════
elif topic == "🧩 K-Means Clustering":
    st.markdown("# 🧩 K-Means Clustering: Complete Visual Guide")
    st.caption("Step-by-step visualization with math at every stage")

    st.markdown("""<div class="dd-card">
    <b>K-Means groups similar data points into K clusters — without any labels!</b>
    <br><br>It's <b>unsupervised learning</b>: no "right answer" to learn from.
    The algorithm discovers natural groupings in your data.
    <br><br>🔹 You choose K (number of clusters)
    <br>🔹 Algorithm assigns each point to nearest centroid
    <br>🔹 Iterates until centroids stop moving
    </div>""", unsafe_allow_html=True)

    # ── STEP 1: INITIAL DATA ──
    st.divider()
    st.markdown("## Step 1: The Data (No Labels!)")

    np.random.seed(42)
    c1_x, c1_y = np.random.normal(300, 60, 20), np.random.normal(4.5, 0.3, 20)
    c2_x, c2_y = np.random.normal(550, 50, 20), np.random.normal(3.5, 0.4, 20)
    c3_x, c3_y = np.random.normal(700, 40, 15), np.random.normal(2.5, 0.3, 15)
    all_x = np.concatenate([c1_x, c2_x, c3_x])
    all_y = np.concatenate([c1_y, c2_y, c3_y])

    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=all_x, y=all_y, mode='markers',
                                 marker=dict(color='#8892b0', size=8, opacity=0.7), name='Stores'))
    fig_raw.update_layout(height=300, title="55 Pizza Stores — Can you see the groups?",
                          xaxis_title="Daily Sales ($)", yaxis_title="Customer Rating", **DL)
    st.plotly_chart(fig_raw, use_container_width=True, config={"displayModeBar": False})

    # ── STEP 2: ALGORITHM ──
    st.divider()
    st.markdown("## Step 2: The K-Means Algorithm")

    # Simulate iterations
    centroids_history = [
        [(200, 4.0), (500, 3.0), (750, 3.5)],  # random init
        [(310, 4.4), (540, 3.4), (690, 2.6)],   # after iter 1
        [(300, 4.5), (550, 3.5), (700, 2.5)],   # converged
    ]
    colors_list = ['#f45d6d', '#22d3a7', '#7c6aff']
    true_labels = [0]*20 + [1]*20 + [2]*15

    tabs = st.tabs(["🎲 Init: Random Centroids", "🔄 Iteration 1", "✅ Converged"])

    for t_idx, tab in enumerate(tabs):
        with tab:
            fig_iter = go.Figure()
            cx, cy = zip(*centroids_history[t_idx])

            if t_idx == 0:
                fig_iter.add_trace(go.Scatter(x=all_x, y=all_y, mode='markers',
                                              marker=dict(color='#8892b0', size=7, opacity=0.5), name='Stores'))
            else:
                # Assign to nearest centroid
                for i in range(len(all_x)):
                    dists = [np.sqrt((all_x[i]-cx[j])**2 + (all_y[i]-cy[j])**2) for j in range(3)]
                    cluster = np.argmin(dists)
                    fig_iter.add_trace(go.Scatter(x=[all_x[i]], y=[all_y[i]], mode='markers',
                                                  marker=dict(color=colors_list[cluster], size=7, opacity=0.5),
                                                  showlegend=False))

            # Centroids
            for j in range(3):
                fig_iter.add_trace(go.Scatter(x=[cx[j]], y=[cy[j]], mode='markers',
                                              marker=dict(color=colors_list[j], size=18, symbol='x',
                                                         line=dict(width=3, color='white')),
                                              name=f'Centroid {j+1}'))

            titles = ["Step 0: Place 3 random centroids", "Step 1: Assign points → Recalculate centroids", "Converged! Centroids stopped moving"]
            fig_iter.update_layout(height=300, title=titles[t_idx],
                                   xaxis_title="Daily Sales ($)", yaxis_title="Rating", **DL)
            st.plotly_chart(fig_iter, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 K-Means Algorithm:</b>
    <br><br><b>1. Initialize:</b> Place K centroids randomly
    <br><br><b>2. Assign:</b> Each point → nearest centroid
    <br>&nbsp;&nbsp;Distance = √((x₁-c₁)² + (x₂-c₂)²)  (Euclidean)
    <br>&nbsp;&nbsp;Example: Store at (300, 4.5), Centroid at (200, 4.0)
    <br>&nbsp;&nbsp;Distance = √((300-200)² + (4.5-4.0)²) = √(10000 + 0.25) = <b>100.0</b>
    <br><br><b>3. Update:</b> Move each centroid to the mean of its assigned points
    <br>&nbsp;&nbsp;New centroid = (mean of x values, mean of y values)
    <br><br><b>4. Repeat</b> steps 2-3 until centroids stop moving (convergence)
    <br><br>Typically converges in <b>5-20 iterations</b>.
    </div>""", unsafe_allow_html=True)

    # ── ELBOW METHOD ──
    st.divider()
    st.markdown("## Step 3: Choosing K — The Elbow Method")

    k_range = range(1, 8)
    inertias = [45000, 18000, 5000, 4200, 3800, 3500, 3300]

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                   line=dict(color='#22d3a7', width=3), marker=dict(size=10)))
    fig_elbow.add_vline(x=3, line_dash="dash", line_color="#f5b731", annotation_text="Elbow: K=3")
    fig_elbow.update_layout(height=280, title="Elbow Method: Where does the curve bend?",
                            xaxis_title="Number of Clusters (K)", yaxis_title="Inertia (WCSS)", **DL)
    st.plotly_chart(fig_elbow, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="dd-math">
    <b>📐 Inertia (WCSS) = Within-Cluster Sum of Squares:</b>
    <br><br>Inertia = Σ Σ ||xᵢ - cₖ||²
    <br><br>For each point, calculate squared distance to its centroid. Sum them all.
    <br><br><b>Lower inertia = tighter clusters = better.</b>
    <br>But K = n (each point is its own cluster) gives inertia = 0 — that's useless!
    <br><br><b>The Elbow:</b> Where adding more clusters stops helping much.
    <br>&nbsp;&nbsp;K=1 → 45000 (terrible)
    <br>&nbsp;&nbsp;K=2 → 18000 (big improvement!)
    <br>&nbsp;&nbsp;K=3 → 5000 (big improvement!)
    <br>&nbsp;&nbsp;K=4 → 4200 (small improvement — diminishing returns)
    <br>&nbsp;&nbsp;→ <b>Elbow at K=3</b>
    </div>""", unsafe_allow_html=True)

    # ── FINAL CLUSTERS ──
    st.divider()
    st.markdown("## Step 4: Interpreting the Clusters")

    fig_final = go.Figure()
    names = ["⭐ Premium (High Sales, High Rating)", "📊 Average (Mid Sales, Mid Rating)", "⚠️ Struggling (High Sales, Low Rating)"]
    for j, (cx, cy, name) in enumerate(zip([c1_x, c2_x, c3_x], [c1_y, c2_y, c3_y], names)):
        fig_final.add_trace(go.Scatter(x=cx, y=cy, mode='markers',
                                       marker=dict(color=colors_list[j], size=9, opacity=0.7), name=name))
    fig_final.update_layout(height=350, title="3 Store Segments Discovered",
                            xaxis_title="Daily Sales ($)", yaxis_title="Rating", **DL)
    st.plotly_chart(fig_final, use_container_width=True, config={"displayModeBar": False})

    cluster_df = pd.DataFrame({
        "Cluster": ["⭐ Premium", "📊 Average", "⚠️ Struggling"],
        "Avg Sales": [f"${np.mean(c1_x):.0f}", f"${np.mean(c2_x):.0f}", f"${np.mean(c3_x):.0f}"],
        "Avg Rating": [f"{np.mean(c1_y):.1f}", f"{np.mean(c2_y):.1f}", f"{np.mean(c3_y):.1f}"],
        "Count": [len(c1_x), len(c2_x), len(c3_x)],
        "Action": ["Reward & replicate", "Improve ratings", "Investigate issues"],
    })
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class="dd-insight">
    💡 <b>Business value:</b> K-Means found 3 natural segments without any labels.
    Now you can create targeted strategies for each group!
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="dd-warn">
    ⚠️ <b>K-Means limitations:</b>
    <br>• Assumes spherical clusters (fails on weird shapes)
    <br>• Sensitive to initial centroid placement (run multiple times)
    <br>• Must choose K in advance
    <br>• Sensitive to scale (always standardize features first!)
    </div>""", unsafe_allow_html=True)
