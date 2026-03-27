# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.story-box {
    background: linear-gradient(135deg, #1a1d2e, #1f2235);
    border: 1px solid #2d3148; border-radius: 16px; padding: 1.4rem 1.6rem;
    margin: 0.8rem 0; line-height: 1.8; font-size: 0.93rem; color: #c8cfe0;
}
.story-box b, .story-box strong { color: #e2e8f0; }
.analogy-box {
    background: linear-gradient(135deg, #1a2a1f, #1f3528);
    border: 1px solid #2a5a3a; border-radius: 14px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; font-size: 0.9rem; color: #c8d8c0; line-height: 1.7;
}
.analogy-box b { color: #d0f0e0; }
.impact-box {
    background: linear-gradient(135deg, #2a1a1e, #351f25);
    border: 1px solid #5a2a3a; border-radius: 14px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; font-size: 0.9rem; color: #d8a8b8; line-height: 1.7;
}
.impact-box b { color: #f0c8d8; }
.key-point {
    background: #252840; border-left: 4px solid #7c6aff;
    border-radius: 0 10px 10px 0; padding: 0.8rem 1.2rem;
    margin: 0.6rem 0; font-size: 0.88rem; color: #c8cfe0; line-height: 1.7;
}
.key-point b { color: #e2e8f0; }
.section-header {
    font-size: 1.6rem; font-weight: 800; margin: 2rem 0 0.5rem;
    padding-bottom: 0.5rem; border-bottom: 2px solid #2d3148;
}
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
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
        with st.expander(f"💡 Answer  --  Q{i}"):
            st.markdown(f'<div class="iq-answer">{q["a"]}</div>', unsafe_allow_html=True)
            if q.get("t"):
                st.markdown(f'<div class="iq-tip">🎯 <b>Tip:</b> {q["t"]}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🤖 Phase 3: ML Concepts")
    st.caption("4-6 Weeks · How machines learn patterns")
    st.divider()
    concept = st.radio("Module:", [
        "🏠 Overview",
        "🔮 Prediction & Forecasting",
        "🎓 Supervised Learning",
        "🧩 Unsupervised Learning",
        "📖 Bias & Variance",
    ], label_visibility="collapsed")

if concept == "🏠 Overview":
    st.markdown("# 🤖 Phase 3: Machine Learning Concepts")
    st.caption("4-6 Weeks · Learn how machines learn patterns  --  no code, just deep intuition.")

    st.markdown("""<div class="story-box">
    Phase 1 gave you statistics. Phase 2 gave you data skills. Now we answer the big question:
    <b>"Can a machine learn patterns from data and make predictions on its own?"</b>
    <br><br>This phase covers the core ML concepts through storytelling and interactive visuals.
    </div>""", unsafe_allow_html=True)

    modules = [
        ("🔮", "Prediction & Forecasting", "#f45d6d", "Can we guess the future from the past? Prediction vs forecasting, overfitting vs underfitting."),
        ("🎓", "Supervised Learning", "#5eaeff", "Learning with a teacher  --  classification, regression, decision boundaries, common algorithms."),
        ("🧩", "Unsupervised Learning", "#a78bfa", "Finding hidden patterns without labels  --  clustering, anomaly detection, dimensionality reduction."),
        ("📖", "Bias & Variance", "#f5b731", "The two forces that make or break your model  --  the central tradeoff in all of ML."),
    ]
    for icon, title, color, desc in modules:
        st.markdown(f"""<div class="story-box" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span> <b>{title}</b>
        <br><span style="color:#8892b0">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-point">
    <b>✅ After Phase 3 you will:</b><br>
    • Understand ML models intuitively<br>
    • Know supervised vs unsupervised learning deeply<br>
    • Understand the bias-variance tradeoff<br>
    • Be ready for ML interview questions at any company
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
elif concept == "🔮 Prediction & Forecasting":
    st.markdown('<div class="section-header">🔮 Prediction & Forecasting  --  Seeing the Future with Data</div>', unsafe_allow_html=True)

    st.markdown("""<div class="story-box">
    Every day, you make predictions without thinking about it:
    <br>• "It's cloudy  --  I should bring an umbrella" <b>(weather prediction)</b>
    <br>• "This customer hasn't logged in for 3 months  --  they might leave" <b>(churn prediction)</b>
    <br>• "Sales went up every December  --  they'll probably go up this December too" <b>(forecasting)</b>
    <br><br>
    In data science, we formalize this intuition. We look at <b>patterns in past data</b> and use them
    to make educated guesses about the future.
    </div>""", unsafe_allow_html=True)

    # Prediction vs Forecasting
    st.markdown("### 🤔 Prediction vs Forecasting  --  What's the Difference?")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="story-box" style="border-color:#7c6aff">
        <b style="color:#7c6aff;font-size:1.1rem">🎯 Prediction</b><br><br>
        <b>What:</b> Guessing an <b>outcome</b> based on input features.<br>
        <b>When:</b> The answer exists but is unknown to us.<br>
        <b>Example:</b> "Will this customer churn?"  --  the customer either will or won't, we just don't know yet.<br><br>
        <b>Analogy:</b> A doctor predicting if a patient has a disease based on symptoms. The disease is already there (or not)  --  the doctor is just figuring it out.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="story-box" style="border-color:#22d3a7">
        <b style="color:#22d3a7;font-size:1.1rem">📈 Forecasting</b><br><br>
        <b>What:</b> Guessing a <b>future value</b> based on time patterns.<br>
        <b>When:</b> The answer doesn't exist yet  --  it's in the future.<br>
        <b>Example:</b> "What will sales be next quarter?"  --  it hasn't happened yet.<br><br>
        <b>Analogy:</b> A weather forecast. Tomorrow's weather doesn't exist yet  --  we're projecting based on patterns.
        </div>""", unsafe_allow_html=True)

    # Interactive: Simple prediction
    st.markdown("### 🎮 Try It: Make a Prediction")
    st.caption("You're a house price predictor. Adjust the house size and see the predicted price.")

    np.random.seed(42)
    sizes = np.random.uniform(500, 3000, 50)
    prices = sizes * 150 + np.random.normal(0, 30000, 50)

    house_size = st.slider("🏠 House size (sq ft):", 500, 3500, 1500, 50, key="pred_size")

    # Simple linear prediction
    slope, intercept = np.polyfit(sizes, prices, 1)
    predicted_price = slope * house_size + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=prices, mode='markers', marker=dict(color='#7c6aff', size=6, opacity=0.6), name='Past Sales'))
    x_line = np.linspace(400, 3600, 100)
    fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode='lines',
                              line=dict(color='#22d3a7', width=2, dash='dash'), name='Trend'))
    fig.add_trace(go.Scatter(x=[house_size], y=[predicted_price], mode='markers',
                              marker=dict(color='#f45d6d', size=15, symbol='star'), name=f'Your House: ${predicted_price:,.0f}'))
    fig.update_layout(height=400, title="House Price Prediction", xaxis_title="Size (sq ft)", yaxis_title="Price ($)", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="analogy-box">
    🏠 A <b>{house_size:,} sq ft</b> house is predicted to cost <b>${predicted_price:,.0f}</b>.<br>
    The model learned: for every extra square foot, the price goes up by about <b>${slope:,.0f}</b>.
    That's the "pattern" it found in past sales.
    </div>""", unsafe_allow_html=True)

    # Interactive: Forecasting
    st.markdown("### 📈 Try It: Forecast the Future")
    st.caption("See how past trends project into the future.")

    months = np.arange(1, 25)
    base_sales = 1000 + 50 * months + 200 * np.sin(months * np.pi / 6) + np.random.normal(0, 80, 24)

    forecast_months = st.slider("Forecast how many months ahead?", 1, 12, 6, key="forecast_m")

    # Simple trend + seasonal forecast
    all_months = np.arange(1, 25 + forecast_months)
    trend = 1000 + 50 * all_months
    seasonal = 200 * np.sin(all_months * np.pi / 6)
    forecast = trend + seasonal

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=months, y=base_sales, mode='lines+markers',
                                 line=dict(color='#7c6aff', width=2), name='Actual Sales'))
    fig_fc.add_trace(go.Scatter(x=all_months[23:], y=forecast[23:], mode='lines+markers',
                                 line=dict(color='#22d3a7', width=2, dash='dash'), name='Forecast'))
    # Confidence band
    upper = forecast[23:] + 150
    lower = forecast[23:] - 150
    fig_fc.add_trace(go.Scatter(x=np.concatenate([all_months[23:], all_months[23:][::-1]]),
                                 y=np.concatenate([upper, lower[::-1]]),
                                 fill='toself', fillcolor='rgba(34,211,167,0.1)',
                                 line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
    fig_fc.add_vline(x=24, line_dash="dot", line_color="#f5b731", annotation_text="Today")
    fig_fc.update_layout(height=400, title="Sales Forecast", xaxis_title="Month", yaxis_title="Sales ($)", **DARK_LAYOUT)
    st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="analogy-box">
    💡 The <b>solid line</b> is what actually happened. The <b>dashed line</b> is our forecast.
    The <b>shaded area</b> is the "confidence band"  --  we're saying "the real value will probably be somewhere in here."
    The further into the future, the wider the band  --  because <b>uncertainty grows with time</b>.
    </div>""", unsafe_allow_html=True)

    # Key concepts  --  explained in depth
    st.markdown("### 📖 Key Terms Explained")

    st.markdown("""<div class="story-box" style="border-color:#7c6aff">
    <b style="color:#7c6aff">📊 Training Data vs Test Data</b><br><br>
    Imagine you're studying for an exam. Your <b>textbook</b> is the training data  --  you learn patterns from it.
    The <b>actual exam</b> is the test data  --  it checks if you truly understood, or just memorized.
    <br><br>
    In data science, we split our data: ~80% for learning (training), ~20% for checking (testing).
    If a model scores 95% on training but 60% on testing, it <b>memorized</b> instead of <b>learned</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box" style="border-color:#f45d6d">
    <b style="color:#f45d6d">⚠️ Overfitting  --  The Student Who Memorizes</b><br><br>
    A student memorizes every answer from past exams word-for-word. On practice tests (training data),
    they score 100%. But on the real exam (test data) with slightly different questions, they score 40%.
    <br><br>
    That's <b>overfitting</b>. The model learned the <b>noise</b> (random quirks) in the training data,
    not the <b>signal</b> (real patterns). It's like memorizing that "the answer to question 3 is B"
    instead of understanding the concept behind question 3.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box" style="border-color:#f5b731">
    <b style="color:#f5b731">📉 Underfitting  --  The Student Who Barely Studied</b><br><br>
    A student reads only the chapter titles and skips all the content. They score poorly on
    <b>both</b> practice tests AND the real exam. The model is too simple to capture the patterns.
    <br><br>
    That's <b>underfitting</b>. It's like trying to predict house prices using only the color of the front door.
    You need more information (features) or a more capable model.
    </div>""", unsafe_allow_html=True)

    # Overfitting visual
    st.markdown("#### 🎮 See Overfitting vs Good Fit")
    st.caption("A good model captures the trend. An overfit model chases every tiny wiggle.")

    np.random.seed(42)
    x_fit = np.linspace(0, 10, 20)
    y_true = 2 * x_fit + 5 + np.random.normal(0, 3, 20)

    fit_type = st.radio("Model complexity:", ["📉 Underfit (too simple)", "✅ Good fit (just right)", "⚠️ Overfit (too complex)"], horizontal=True, key="fit_demo")

    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=x_fit, y=y_true, mode='markers', marker=dict(color='#22d3a7', size=8), name='Real Data'))

    x_smooth = np.linspace(0, 10, 200)
    if "Underfit" in fit_type:
        fig_fit.add_trace(go.Scatter(x=x_smooth, y=np.full_like(x_smooth, y_true.mean()),
                                      mode='lines', line=dict(color='#f5b731', width=3), name='Model (flat line)'))
        fit_msg = "The model is just a flat line  --  it doesn't capture the upward trend at all. Too simple."
    elif "Good" in fit_type:
        z = np.polyfit(x_fit, y_true, 1)
        fig_fit.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z, x_smooth),
                                      mode='lines', line=dict(color='#7c6aff', width=3), name='Model (straight line)'))
        fit_msg = "The model captures the overall upward trend without chasing every random wiggle. Just right."
    else:
        z = np.polyfit(x_fit, y_true, 15)
        fig_fit.add_trace(go.Scatter(x=x_smooth, y=np.polyval(z, x_smooth).clip(-10, 40),
                                      mode='lines', line=dict(color='#f45d6d', width=3), name='Model (wiggly)'))
        fit_msg = "The model bends to hit every single point  --  it memorized the noise. It'll fail badly on new data."

    fig_fit.update_layout(height=350, title="How Well Does the Model Fit?", **DARK_LAYOUT)
    st.plotly_chart(fig_fit, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""<div class="analogy-box">💡 {fit_msg}</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-point">
    <b>🎯 Key Takeaway:</b> Prediction and forecasting are about finding <b>patterns in the past</b> and
    assuming they'll continue. The art is knowing <b>which patterns are real</b> and which are just noise.
    A good model is like a good student  --  it understands the concepts, not just the specific examples.
    </div>""", unsafe_allow_html=True)

    iq([
        {"q": "What's the difference between prediction and forecasting?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Prediction:</b> Guessing an outcome based on input features. The answer exists but is unknown to us. Example: 'Will this customer churn?' <b>Forecasting:</b> Guessing a future value based on time patterns. The answer doesn't exist yet. Example: 'What will sales be next quarter?' <b>Key:</b> Prediction = cross-sectional. Forecasting = time-based.",
         "t": "Give one example of each to make the distinction concrete."},
        {"q": "What is overfitting? How do you detect and prevent it?", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "<b>Overfitting:</b> Model performs great on training data but poorly on new data  --  it memorized noise. <b>Detection:</b> Big gap between training accuracy (high) and test accuracy (low). <b>Prevention:</b> (1) Train/test split or cross-validation, (2) Regularization (L1/L2), (3) Simpler model, (4) More training data, (5) Early stopping, (6) Dropout (neural nets). <b>Analogy:</b> A student who memorizes answers instead of understanding concepts.",
         "t": "Use the student analogy  --  universally understood."},
        {"q": "What is underfitting and how is it different from overfitting?", "d": "Easy", "c": ["Amazon", "Apple"],
         "a": "<b>Underfitting:</b> Model is too simple to capture the real pattern. Bad on BOTH training and test data. <b>Overfitting:</b> Model is too complex  --  great on training, bad on test. <b>Analogy:</b> Underfitting = studying only chapter titles. Overfitting = memorizing every word including typos. <b>Fix underfitting:</b> More features, more complex model, less regularization.",
         "t": "Always contrast with overfitting  --  interviewers want to see you understand both sides."},
        {"q": "What is the difference between training data and test data?", "d": "Easy", "c": ["Google", "General"],
         "a": "<b>Training data:</b> The data the model learns from  --  like a textbook. <b>Test data:</b> New data the model has never seen  --  like the actual exam. <b>Why separate?</b> If you test on training data, you're checking if the model memorized, not if it learned. A model that scores 99% on training but 60% on test has overfitted. <b>Typical split:</b> 80% train, 20% test.",
         "t": "Mention cross-validation as a more robust alternative to a single train/test split."},
        {"q": "Your model has 95% training accuracy but 60% test accuracy. What happened?", "d": "Medium", "c": ["Meta", "Google", "Netflix"],
         "a": "Classic <b>overfitting</b>. The model memorized the training data including noise. <b>Diagnosis:</b> The 35% gap between train and test accuracy is the red flag. <b>Fixes:</b> (1) Simplify the model (fewer features, shallower tree). (2) Add regularization. (3) Get more training data. (4) Use cross-validation instead of a single split. (5) Check for data leakage  --  a feature that 'leaks' the target.",
         "t": "Always mention data leakage as a possible cause  --  it's a common gotcha."},
    ])


# ═══════════════════════════════════════════════════
# SUPERVISED LEARNING
# ═══════════════════════════════════════════════════
elif concept == "🎓 Supervised Learning":
    st.markdown('<div class="section-header">🎓 Supervised Learning  --  Learning with a Teacher</div>', unsafe_allow_html=True)

    st.markdown("""<div class="story-box">
    Imagine teaching a child to recognize animals. You show them pictures:
    <br>• "This is a 🐱 cat"  --  "This is a 🐶 dog"  --  "This is a 🐱 cat"  --  "This is a 🐶 dog"
    <br><br>
    After enough examples, the child can look at a <b>new picture</b> they've never seen and say "that's a cat!"
    <br><br>
    That's <b>supervised learning</b>. You give the machine <b>labeled examples</b> (inputs + correct answers),
    and it learns the pattern. Then it can predict answers for <b>new, unseen inputs</b>.
    <br><br>
    The "supervisor" is the <b>labeled data</b>  --  the answer key.
    </div>""", unsafe_allow_html=True)

    # Two types
    st.markdown("### 📋 Two Types of Supervised Learning")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="story-box" style="border-color:#7c6aff">
        <b style="color:#7c6aff;font-size:1.1rem">📊 Classification</b><br>
        <i>Predicting a category</i><br><br>
        <b>Question:</b> "Which group does this belong to?"<br>
        <b>Answer:</b> A label (Yes/No, Cat/Dog, Spam/Not Spam)<br><br>
        <b>Examples:</b><br>
        • Will this customer churn? -> <b>Yes / No</b><br>
        • Is this email spam? -> <b>Spam / Not Spam</b><br>
        • What disease does this patient have? -> <b>A / B / C</b><br>
        • Is this transaction fraud? -> <b>Fraud / Legit</b>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="story-box" style="border-color:#22d3a7">
        <b style="color:#22d3a7;font-size:1.1rem">📈 Regression</b><br>
        <i>Predicting a number</i><br><br>
        <b>Question:</b> "How much / how many?"<br>
        <b>Answer:</b> A continuous number<br><br>
        <b>Examples:</b><br>
        • What will this house sell for? -> <b>$350,000</b><br>
        • How many units will we sell? -> <b>1,247</b><br>
        • What's this patient's blood pressure? -> <b>128</b><br>
        • How long until this machine fails? -> <b>45 days</b>
        </div>""", unsafe_allow_html=True)

    # Interactive: Classification demo
    st.markdown("### 🎮 Try It: Classification in Action")
    st.caption("You're building a customer churn predictor. Adjust the customer's profile and see the prediction change.")

    c1, c2, c3 = st.columns(3)
    with c1:
        cust_tenure = st.slider("📅 Tenure (months):", 1, 72, 12, key="sl_tenure")
    with c2:
        cust_monthly = st.slider("💰 Monthly charge ($):", 20, 120, 70, key="sl_monthly")
    with c3:
        cust_tickets = st.slider("📞 Support tickets:", 0, 10, 2, key="sl_tickets")

    contract = st.radio("📋 Contract type:", ["Month-to-month", "One year", "Two year"], horizontal=True, key="sl_contract")
    contract_score = {"Month-to-month": 0.3, "One year": -0.1, "Two year": -0.3}[contract]

    # Simple logistic-style scoring
    churn_score = (-0.02 * cust_tenure + 0.01 * cust_monthly + 0.12 * cust_tickets + contract_score + 0.5)
    churn_prob = 1 / (1 + np.exp(-churn_score * 3))
    churn_prob = np.clip(churn_prob, 0.02, 0.98)

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=churn_prob * 100,
        number={"suffix": "%", "font": {"size": 50, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#f45d6d" if churn_prob > 0.5 else "#22d3a7"},
            "bgcolor": "#252840", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(34,211,167,0.15)"},
                {"range": [30, 60], "color": "rgba(245,183,49,0.15)"},
                {"range": [60, 100], "color": "rgba(244,93,109,0.15)"},
            ],
            "threshold": {"line": {"color": "#f5b731", "width": 3}, "thickness": 0.8, "value": 50},
        },
    ))
    fig_gauge.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                            title="Churn Probability", margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    prediction = "🔴 LIKELY TO CHURN" if churn_prob > 0.5 else "🟢 LIKELY TO STAY"
    risk = "High" if churn_prob > 0.7 else "Medium" if churn_prob > 0.4 else "Low"

    st.markdown(f"""<div class="{'impact-box' if churn_prob > 0.5 else 'analogy-box'}">
    <b>Prediction: {prediction}</b> (Confidence: {churn_prob*100:.0f}%, Risk: {risk})<br><br>
    <b>Why?</b> {'Short tenure + high charges + many tickets = high risk.' if churn_prob > 0.5 else 'Longer tenure and fewer issues = lower risk.'}
    {'A month-to-month contract adds risk  --  no commitment.' if contract == 'Month-to-month' else 'A longer contract reduces risk  --  the customer is committed.'}
    </div>""", unsafe_allow_html=True)

    # Interactive: Regression demo
    st.markdown("### 🎮 Try It: Regression in Action")
    st.caption("Predict house prices. Drag the features and watch the prediction update.")

    c1, c2 = st.columns(2)
    with c1:
        sqft = st.slider("📐 Square footage:", 500, 4000, 1500, 50, key="reg_sqft")
        bedrooms = st.slider("🛏️ Bedrooms:", 1, 6, 3, key="reg_bed")
    with c2:
        age = st.slider("📅 House age (years):", 0, 50, 10, key="reg_age")
        garage = st.radio("🚗 Garage:", ["No", "Yes"], horizontal=True, key="reg_garage")

    price = (sqft * 120 + bedrooms * 15000 - age * 2000 + (25000 if garage == "Yes" else 0) + 50000
             + np.random.normal(0, 5000))

    st.markdown(f"""<div class="analogy-box" style="text-align:center">
    <div style="font-size:0.8rem;color:#8892b0">PREDICTED PRICE</div>
    <div style="font-size:2.5rem;font-weight:800;color:#22d3a7">${price:,.0f}</div>
    <div style="font-size:0.85rem;color:#8892b0;margin-top:0.5rem">
    {sqft:,} sqft × $120/sqft + {bedrooms} beds × $15K + {'garage bonus' if garage == 'Yes' else 'no garage'} - {age}yr aging
    </div>
    </div>""", unsafe_allow_html=True)

    # Common algorithms  --  deep explanations
    st.markdown("### 🧰 Key Algorithms (Conceptual)")
    st.caption("You don't need to code these yet  --  just understand what each one does and when to use it.")

    with st.expander("📏 **Linear Regression**  --  Drawing the best straight line", expanded=True):
        st.markdown("""<div class="story-box">
        <b>What it does:</b> Finds the straight line that best fits your data. Predicts a <b>number</b>.
        <br><br>
        <div style="text-align:center;font-size:1.2rem;padding:0.6rem;background:rgba(124,106,255,0.08);border-radius:10px;margin:0.5rem 0">
        <b>y = mx + b</b>
        </div>
        <br>
        • <b>m (slope):</b> "For every 1-unit increase in X, Y changes by m." Example: each extra sq ft adds $150 to house price.
        <br>• <b>b (intercept):</b> "The value of Y when X = 0." Example: base price of a house with 0 sq ft (theoretical starting point).
        <br><br>
        <b>How it learns:</b> It tries many lines and picks the one that <b>minimizes the total error</b>
        (sum of squared distances between each point and the line). This is called <b>Least Squares</b>.
        </div>""", unsafe_allow_html=True)

        # Interactive linear regression visual
        st.caption("🎮 Drag the slope and intercept to fit the line. See how error changes.")
        lr_c1, lr_c2 = st.columns(2)
        lr_slope = lr_c1.slider("Slope (m):", 0.0, 300.0, 100.0, 10.0, key="algo_lr_m")
        lr_intercept = lr_c2.slider("Intercept (b):", -100000.0, 200000.0, 50000.0, 10000.0, key="algo_lr_b")

        np.random.seed(42)
        sqft = np.linspace(500, 3000, 30)
        true_price = 150 * sqft + 50000 + np.random.normal(0, 30000, 30)
        user_price = lr_slope * sqft + lr_intercept
        best_m, best_b = np.polyfit(sqft, true_price, 1)
        best_price = best_m * sqft + best_b
        user_mse = np.mean((true_price - user_price)**2)
        best_mse = np.mean((true_price - best_price)**2)

        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=sqft, y=true_price, mode='markers', marker=dict(color='#22d3a7', size=7), name='Houses'))
        fig_lr.add_trace(go.Scatter(x=sqft, y=user_price, mode='lines', line=dict(color='#f5b731', width=2), name=f'Your line (m={lr_slope:.0f}, b={lr_intercept:.0f})'))
        fig_lr.add_trace(go.Scatter(x=sqft, y=best_price, mode='lines', line=dict(color='#7c6aff', width=2, dash='dash'), name=f'Best fit (m={best_m:.0f}, b={best_b:.0f})'))
        # Show residuals for user line
        for i in range(0, len(sqft), 5):
            fig_lr.add_trace(go.Scatter(x=[sqft[i], sqft[i]], y=[true_price[i], user_price[i]],
                mode='lines', line=dict(color='#f45d6d', width=1, dash='dot'), showlegend=False))
        fig_lr.update_layout(height=380, title="House Price vs Size  --  Red dotted lines = errors (residuals)",
                             xaxis_title="Size (sq ft)", yaxis_title="Price ($)", **DARK_LAYOUT)
        st.plotly_chart(fig_lr, use_container_width=True, config={"displayModeBar": False})

        mc = st.columns(3)
        mc[0].metric("Your Error (MSE)", f"{user_mse/1e6:.1f}M")
        mc[1].metric("Best Fit Error", f"{best_mse/1e6:.1f}M")
        mc[2].metric("How close?", f"{user_mse/best_mse:.1f}×" if best_mse > 0 else "∞")

        st.markdown("""<div class="analogy-box">
        💡 The <b style="color:#f45d6d">red dotted lines</b> are <b>residuals</b>  --  the error for each point.
        The best line minimizes the total of these squared distances. That's what "least squares" means.
        <br><br><b>Strengths:</b> Simple, fast, interpretable ("each extra sq ft adds $150").
        <br><b>Weakness:</b> Can only draw straight lines. If the real pattern is curved, it'll miss it.
        </div>""", unsafe_allow_html=True)

    with st.expander("📊 **Logistic Regression**  --  Yes or No? (despite the name, it's classification)"):
        st.markdown("""<div class="story-box">
        <b>What it does:</b> Predicts the <b>probability</b> (0 to 1) of belonging to a category.
        <br><br>
        <b>The math intuition:</b> Start with a linear equation (like linear regression): z = mx + b.
        But z can be any number (-∞ to +∞). We need a probability (0 to 1). So we squash z through the
        <b>sigmoid function</b>:
        <br><br>
        <div style="text-align:center;font-size:1.2rem;padding:0.6rem;background:rgba(34,211,167,0.08);border-radius:10px;margin:0.5rem 0">
        <b>P(yes) = 1 / (1 + e<sup>-z</sup>)</b>
        </div>
        <br>
        This S-shaped curve maps any number to a probability between 0 and 1.
        </div>""", unsafe_allow_html=True)

        # Interactive sigmoid
        st.caption("🎮 See how the sigmoid transforms a linear score into a probability.")
        sig_shift = st.slider("Decision threshold (shifts the curve left/right):", -5.0, 5.0, 0.0, 0.5, key="algo_sig_shift")

        z_vals = np.linspace(-8, 8, 200)
        sigmoid = 1 / (1 + np.exp(-(z_vals - sig_shift)))

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=z_vals, y=sigmoid, mode='lines', line=dict(color='#22d3a7', width=3), name='Sigmoid'))
        fig_sig.add_hline(y=0.5, line_dash="dash", line_color="#f5b731", annotation_text="Decision boundary: 0.5")
        fig_sig.add_vline(x=sig_shift, line_dash="dot", line_color="#7c6aff", annotation_text=f"Threshold at z={sig_shift}")
        # Shade regions
        fig_sig.add_vrect(x0=-8, x1=sig_shift, fillcolor="rgba(244,93,109,0.06)", line_width=0)
        fig_sig.add_vrect(x0=sig_shift, x1=8, fillcolor="rgba(34,211,167,0.06)", line_width=0)
        fig_sig.add_annotation(x=-4, y=0.8, text="Predict: NO", font=dict(color="#f45d6d", size=12), showarrow=False)
        fig_sig.add_annotation(x=4, y=0.8, text="Predict: YES", font=dict(color="#22d3a7", size=12), showarrow=False)
        fig_sig.update_layout(height=350, title="The Sigmoid Function: Linear Score -> Probability",
                              xaxis_title="Linear score (z = mx + b)", yaxis_title="Probability P(yes)", **DARK_LAYOUT)
        st.plotly_chart(fig_sig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""<div class="analogy-box">
        💡 <b>How to read this:</b> The x-axis is the "raw score" from the linear equation (like a risk score).
        The y-axis is the probability. The sigmoid squashes everything into 0-1.
        <br>• Score very negative -> probability near 0 (definitely NO)
        <br>• Score near 0 -> probability near 0.5 (uncertain  --  coin flip)
        <br>• Score very positive -> probability near 1 (definitely YES)
        <br><br><b>When to use:</b> Binary classification (churn/stay, spam/not spam, fraud/legit).
        <br><b>Strengths:</b> Gives probabilities (not just yes/no), interpretable, fast.
        <br><b>Weakness:</b> Assumes a linear decision boundary. Can't capture complex patterns.
        </div>""", unsafe_allow_html=True)

    with st.expander("🌳 **Decision Trees**  --  A flowchart that makes decisions"):
        st.markdown("""<div class="story-box">
        <b>What it does:</b> Splits data by asking a series of <b>yes/no questions</b>, creating a tree-like flowchart.
        <br><br>
        <b>How it learns (the math):</b> At each node, the tree tries every possible split on every feature
        and picks the one that <b>best separates the classes</b>. "Best" is measured by <b>Gini Impurity</b>
        or <b>Information Gain</b>:
        <br><br>
        <div style="text-align:center;padding:0.5rem;background:rgba(124,106,255,0.08);border-radius:10px">
        <b>Gini = 1 - sum(p_i^2)</b> for each class i
        </div>
        <br>
        Gini = 0 means perfectly pure (all one class). Gini = 0.5 means maximum impurity (50/50 split).
        The tree picks the split that reduces Gini the most.
        </div>""", unsafe_allow_html=True)

        # Interactive: How a split works
        st.markdown("#### 🎮 See How a Tree Splits Data")
        st.caption("Drag the split point and watch how it separates churned vs stayed customers.")

        split_val = st.slider("Split: 'Is tenure > X months?'", 5, 60, 24, 1, key="tree_split")

        np.random.seed(42)
        n_tree = 200
        tenure_tree = np.random.randint(1, 72, n_tree)
        churn_prob_tree = 1 / (1 + np.exp(0.08 * (tenure_tree - 20)))
        churned_tree = (np.random.random(n_tree) < churn_prob_tree).astype(int)

        left_mask = tenure_tree <= split_val
        right_mask = ~left_mask

        # Calculate Gini for each side
        def gini(labels):
            if len(labels) == 0: return 0
            p = labels.mean()
            return 1 - p**2 - (1-p)**2

        gini_left = gini(churned_tree[left_mask])
        gini_right = gini(churned_tree[right_mask])
        gini_parent = gini(churned_tree)
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        weighted_gini = (n_left * gini_left + n_right * gini_right) / n_tree
        info_gain = gini_parent - weighted_gini

        fig_split = go.Figure()
        for label, color, name in [(0, '#22d3a7', 'Stayed'), (1, '#f45d6d', 'Churned')]:
            mask = churned_tree == label
            fig_split.add_trace(go.Scatter(
                x=tenure_tree[mask], y=np.random.normal(label, 0.15, mask.sum()),
                mode='markers', marker=dict(color=color, size=6, opacity=0.5), name=name))
        fig_split.add_vline(x=split_val, line_dash="dash", line_color="#f5b731", line_width=3,
                            annotation_text=f"Split at tenure = {split_val}")
        fig_split.add_vrect(x0=0, x1=split_val, fillcolor="rgba(124,106,255,0.05)", line_width=0)
        fig_split.add_vrect(x0=split_val, x1=75, fillcolor="rgba(34,211,167,0.05)", line_width=0)
        fig_split.update_layout(height=300, title="How the tree splits: left vs right",
                                xaxis_title="Tenure (months)", yaxis_title="Class (0=Stay, 1=Churn)", **DARK_LAYOUT)
        st.plotly_chart(fig_split, use_container_width=True, config={"displayModeBar": False})

        mc = st.columns(4)
        mc[0].metric("Left side (<=)", f"{n_left} pts", f"Gini={gini_left:.3f}")
        mc[1].metric("Right side (>)", f"{n_right} pts", f"Gini={gini_right:.3f}")
        mc[2].metric("Info Gain", f"{info_gain:.4f}", help="Higher = better split. The tree picks the split with max info gain.")
        mc[3].metric("Churn rate left", f"{churned_tree[left_mask].mean():.0%}" if n_left > 0 else "N/A")

        st.markdown(f"""<div class="analogy-box">
        💡 <b>Reading this:</b> The yellow line splits customers into two groups. Left side (tenure <= {split_val})
        has Gini={gini_left:.3f} and right side has Gini={gini_right:.3f}. Lower Gini = purer group.
        The tree picks the split that maximizes <b>Information Gain = {info_gain:.4f}</b>.
        Try moving the slider to find the best split point!
        </div>""", unsafe_allow_html=True)

        # Interactive: max_depth parameter
        st.markdown("#### 🎮 Key Parameter: max_depth (How Deep the Tree Goes)")
        st.caption("Shallow tree = underfitting. Deep tree = overfitting. Find the sweet spot.")

        max_depth = st.slider("max_depth:", 1, 10, 3, key="tree_depth")

        depth_desc = {
            1: ("Very shallow -- only 1 split. Underfitting. Like asking only 'Is it an animal?' and nothing else.", "#f5b731"),
            2: ("Shallow -- 2 levels of questions. Captures basic patterns but misses details.", "#f5b731"),
            3: ("Good balance -- captures main patterns without memorizing noise.", "#22d3a7"),
            4: ("Good balance -- slightly more detailed. Usually a good default.", "#22d3a7"),
            5: ("Getting complex -- starts to capture noise in training data.", "#f5b731"),
        }
        desc, color = depth_desc.get(max_depth, ("Very deep -- high risk of overfitting. The tree memorizes training data.", "#f45d6d") if max_depth > 5 else ("Moderate depth.", "#f5b731"))

        # Simulate train vs test accuracy at different depths
        depths = list(range(1, 11))
        train_acc = [0.60 + 0.04 * d for d in depths]  # keeps improving
        test_acc = [0.58 + 0.035 * d if d <= 4 else 0.72 - 0.02 * (d - 4) for d in depths]  # peaks then drops

        fig_depth = go.Figure()
        fig_depth.add_trace(go.Scatter(x=depths, y=train_acc, mode='lines+markers',
            line=dict(color='#7c6aff', width=2), name='Training accuracy'))
        fig_depth.add_trace(go.Scatter(x=depths, y=test_acc, mode='lines+markers',
            line=dict(color='#22d3a7', width=2), name='Test accuracy'))
        fig_depth.add_trace(go.Scatter(x=[max_depth], y=[test_acc[max_depth-1]], mode='markers',
            marker=dict(color='#f5b731', size=14, symbol='star'), name='Your choice'))
        fig_depth.add_vline(x=max_depth, line_dash="dot", line_color="#f5b731")
        fig_depth.update_layout(height=300, title="Training vs Test Accuracy by Tree Depth",
                                xaxis_title="max_depth", yaxis_title="Accuracy", **DARK_LAYOUT)
        st.plotly_chart(fig_depth, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""<div class="{'analogy-box' if 'Good' in desc else 'story-box'}" style="border-color:{color}">
        <b>max_depth = {max_depth}:</b> {desc}
        <br><br>Notice: training accuracy always improves with depth (purple line goes up). But test accuracy
        peaks around depth 3-4 then drops -- thats overfitting. The gap between the lines = overfitting amount.
        </div>""", unsafe_allow_html=True)

        # Parameter reference table
        st.markdown("#### 📋 Key Parameters at a Glance")
        params_table = pd.DataFrame({
            "Parameter": ["max_depth", "min_samples_split", "min_samples_leaf", "max_features", "criterion"],
            "What it controls": [
                "How deep the tree can grow",
                "Min samples needed to split a node",
                "Min samples required in a leaf node",
                "How many features to consider at each split",
                "How to measure split quality",
            ],
            "Low value effect": [
                "Underfitting (too simple)",
                "More splits, deeper tree",
                "More splits, deeper tree",
                "More randomness (good for forests)",
                "N/A",
            ],
            "High value effect": [
                "Overfitting (memorizes noise)",
                "Fewer splits, shallower tree",
                "Fewer splits, shallower tree",
                "Less randomness",
                "N/A",
            ],
            "Typical default": ["3-10", "2-20", "1-10", "sqrt(n_features)", "gini or entropy"],
        })
        st.dataframe(params_table, use_container_width=True, hide_index=True)

        # Other key parameters — interactive
        st.markdown("#### 🎛 Now See Each Parameter in Action")
        st.caption("Pick a parameter from the table above and see its effect visually.")

        param_pick = st.selectbox("Parameter to explore:", [
            "max_depth -- How deep the tree grows",
            "min_samples_split -- Min samples to make a split",
            "min_samples_leaf -- Min samples in each leaf",
            "criterion -- Gini vs Entropy",
        ], key="dt_param_pick")

        # Generate data for demo
        np.random.seed(42)
        n_demo = 200
        x_demo = np.random.uniform(0, 10, (n_demo, 2))
        y_demo = ((np.sin(x_demo[:, 0]) + x_demo[:, 1] / 5) > 1).astype(int)
        split_d = int(n_demo * 0.7)
        X_tr_d, X_te_d = x_demo[:split_d], x_demo[split_d:]
        y_tr_d, y_te_d = y_demo[:split_d], y_demo[split_d:]

        from sklearn.tree import DecisionTreeClassifier as DTC

        if "max_depth" in param_pick:
            val = st.slider("max_depth:", 1, 15, 3, key="dt_p_depth")
            models_range = range(1, 16)
            train_scores = []
            test_scores = []
            leaves_list = []
            for d in models_range:
                m = DTC(max_depth=d, random_state=42).fit(X_tr_d, y_tr_d)
                train_scores.append(m.score(X_tr_d, y_tr_d))
                test_scores.append(m.score(X_te_d, y_te_d))
                leaves_list.append(m.get_n_leaves())

            c1, c2 = st.columns(2)
            with c1:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=list(models_range), y=train_scores, mode='lines+markers',
                    line=dict(color='#7c6aff', width=2), name='Train accuracy'))
                fig_p.add_trace(go.Scatter(x=list(models_range), y=test_scores, mode='lines+markers',
                    line=dict(color='#22d3a7', width=2), name='Test accuracy'))
                fig_p.add_vline(x=val, line_dash="dot", line_color="#f5b731")
                fig_p.update_layout(height=300, title="Accuracy vs max_depth",
                    xaxis_title="max_depth", yaxis_title="Accuracy", **DARK_LAYOUT)
                st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})
            with c2:
                fig_l = go.Figure(go.Bar(x=list(models_range), y=leaves_list,
                    marker=dict(color=['#f5b731' if d == val else '#7c6aff' for d in models_range], cornerradius=4)))
                fig_l.update_layout(height=300, title="Number of leaves vs max_depth",
                    xaxis_title="max_depth", yaxis_title="Leaves", **DARK_LAYOUT)
                st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})

            m_cur = DTC(max_depth=val, random_state=42).fit(X_tr_d, y_tr_d)
            st.markdown(f"""<div class="analogy-box">
            💡 <b>max_depth={val}:</b> Tree has <b>{m_cur.get_n_leaves()} leaves</b> and actual depth {m_cur.get_depth()}.
            Train: {m_cur.score(X_tr_d, y_tr_d):.1%}, Test: {m_cur.score(X_te_d, y_te_d):.1%}.
            {'Underfitting -- tree is too shallow to capture the pattern.' if val <= 2 else 'Good balance -- captures pattern without memorizing noise.' if val <= 5 else 'Overfitting risk -- tree is very deep. Notice train accuracy is much higher than test.'}
            </div>""", unsafe_allow_html=True)

        elif "min_samples_split" in param_pick:
            val = st.slider("min_samples_split:", 2, 50, 2, key="dt_p_mss")
            models_range = [2, 5, 10, 15, 20, 30, 40, 50]
            train_scores = []
            test_scores = []
            leaves_list = []
            for ms in models_range:
                m = DTC(min_samples_split=ms, random_state=42).fit(X_tr_d, y_tr_d)
                train_scores.append(m.score(X_tr_d, y_tr_d))
                test_scores.append(m.score(X_te_d, y_te_d))
                leaves_list.append(m.get_n_leaves())

            c1, c2 = st.columns(2)
            with c1:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=models_range, y=train_scores, mode='lines+markers',
                    line=dict(color='#7c6aff', width=2), name='Train'))
                fig_p.add_trace(go.Scatter(x=models_range, y=test_scores, mode='lines+markers',
                    line=dict(color='#22d3a7', width=2), name='Test'))
                fig_p.update_layout(height=300, title="Accuracy vs min_samples_split",
                    xaxis_title="min_samples_split", yaxis_title="Accuracy", **DARK_LAYOUT)
                st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})
            with c2:
                fig_l = go.Figure(go.Bar(x=[str(x) for x in models_range], y=leaves_list,
                    marker=dict(color='#f5b731', cornerradius=4)))
                fig_l.update_layout(height=300, title="Leaves vs min_samples_split",
                    xaxis_title="min_samples_split", yaxis_title="Leaves", **DARK_LAYOUT)
                st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})

            m_cur = DTC(min_samples_split=val, random_state=42).fit(X_tr_d, y_tr_d)
            st.markdown(f"""<div class="analogy-box">
            💡 <b>min_samples_split={val}:</b> A node needs at least {val} samples to be split further.
            {'Low value -- tree splits aggressively, creating many small leaves. Risk of overfitting.' if val < 10 else 'High value -- tree stops splitting early. Simpler tree, less overfitting, but might miss patterns.'}
            Tree has {m_cur.get_n_leaves()} leaves. Train: {m_cur.score(X_tr_d, y_tr_d):.1%}, Test: {m_cur.score(X_te_d, y_te_d):.1%}.
            </div>""", unsafe_allow_html=True)

        elif "min_samples_leaf" in param_pick:
            val = st.slider("min_samples_leaf:", 1, 30, 1, key="dt_p_msl")
            models_range = [1, 2, 5, 10, 15, 20, 25, 30]
            train_scores = []
            test_scores = []
            leaves_list = []
            for ml in models_range:
                m = DTC(min_samples_leaf=ml, random_state=42).fit(X_tr_d, y_tr_d)
                train_scores.append(m.score(X_tr_d, y_tr_d))
                test_scores.append(m.score(X_te_d, y_te_d))
                leaves_list.append(m.get_n_leaves())

            c1, c2 = st.columns(2)
            with c1:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=models_range, y=train_scores, mode='lines+markers',
                    line=dict(color='#7c6aff', width=2), name='Train'))
                fig_p.add_trace(go.Scatter(x=models_range, y=test_scores, mode='lines+markers',
                    line=dict(color='#22d3a7', width=2), name='Test'))
                fig_p.update_layout(height=300, title="Accuracy vs min_samples_leaf",
                    xaxis_title="min_samples_leaf", yaxis_title="Accuracy", **DARK_LAYOUT)
                st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})
            with c2:
                fig_l = go.Figure(go.Bar(x=[str(x) for x in models_range], y=leaves_list,
                    marker=dict(color='#e879a8', cornerradius=4)))
                fig_l.update_layout(height=300, title="Leaves vs min_samples_leaf",
                    xaxis_title="min_samples_leaf", yaxis_title="Leaves", **DARK_LAYOUT)
                st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})

            m_cur = DTC(min_samples_leaf=val, random_state=42).fit(X_tr_d, y_tr_d)
            st.markdown(f"""<div class="analogy-box">
            💡 <b>min_samples_leaf={val}:</b> Every leaf must have at least {val} data points.
            {'Value of 1 means leaves can have a single point -- the tree memorizes individual examples. High overfitting risk.' if val == 1 else f'With {val} samples per leaf, the tree makes broader generalizations. Each prediction is based on at least {val} examples, not just one.'}
            Tree has {m_cur.get_n_leaves()} leaves. Train: {m_cur.score(X_tr_d, y_tr_d):.1%}, Test: {m_cur.score(X_te_d, y_te_d):.1%}.
            </div>""", unsafe_allow_html=True)

        else:  # criterion
            crit = st.radio("criterion:", ["gini", "entropy"], horizontal=True, key="dt_p_crit")

            m_gini = DTC(criterion="gini", max_depth=5, random_state=42).fit(X_tr_d, y_tr_d)
            m_entropy = DTC(criterion="entropy", max_depth=5, random_state=42).fit(X_tr_d, y_tr_d)

            # Show Gini vs Entropy curves
            p_range = np.linspace(0.01, 0.99, 100)
            gini_vals = 1 - p_range**2 - (1-p_range)**2
            entropy_vals = -(p_range * np.log2(p_range) + (1-p_range) * np.log2(1-p_range))

            fig_crit = go.Figure()
            fig_crit.add_trace(go.Scatter(x=p_range, y=gini_vals, mode='lines',
                line=dict(color='#7c6aff', width=3), name='Gini Impurity'))
            fig_crit.add_trace(go.Scatter(x=p_range, y=entropy_vals, mode='lines',
                line=dict(color='#22d3a7', width=3), name='Entropy'))
            fig_crit.update_layout(height=300, title="Gini vs Entropy: Both measure impurity",
                xaxis_title="P(class 1)", yaxis_title="Impurity", **DARK_LAYOUT)
            st.plotly_chart(fig_crit, use_container_width=True, config={"displayModeBar": False})

            mc = st.columns(2)
            mc[0].metric("Gini -- Train/Test", f"{m_gini.score(X_tr_d, y_tr_d):.1%} / {m_gini.score(X_te_d, y_te_d):.1%}")
            mc[1].metric("Entropy -- Train/Test", f"{m_entropy.score(X_tr_d, y_tr_d):.1%} / {m_entropy.score(X_te_d, y_te_d):.1%}")

            st.markdown("""<div class="analogy-box">
            💡 <b>Gini vs Entropy:</b> Both measure how "impure" a node is (how mixed the classes are).
            They produce very similar trees in practice. <b>Gini</b> is slightly faster to compute (no logarithm).
            <b>Entropy</b> tends to produce slightly more balanced trees. <b>In practice:</b> just use Gini (the default).
            The difference is almost never significant.
            </div>""", unsafe_allow_html=True)

    with st.expander("🌲 **Random Forest**  --  Many trees voting together"):
        st.markdown("""<div class="story-box">
        <b>What it does:</b> Builds <b>hundreds of decision trees</b>, each trained on a random subset of data
        and features. Final prediction = <b>majority vote</b> (classification) or <b>average</b> (regression).
        <br><br>
        <b>Two sources of randomness:</b>
        <br>1. <b>Bootstrap sampling:</b> Each tree gets a random sample of rows (with replacement)
        <br>2. <b>Random feature subsets:</b> At each split, only a random subset of features is considered
        <br><br>
        This double randomness makes each tree different, so their errors cancel when averaged.
        </div>""", unsafe_allow_html=True)

        # Interactive: n_estimators effect
        st.markdown("#### 🎮 Key Parameter: n_estimators (Number of Trees)")
        st.caption("More trees = more stable predictions. But diminishing returns after a point.")

        n_est = st.slider("Number of trees:", 1, 200, 10, 5, key="rf_n_est")

        np.random.seed(42)
        true_val_rf = 75.0
        all_tree_preds = np.random.normal(true_val_rf, 15, 200)

        # Show how ensemble prediction stabilizes
        ensemble_preds = []
        for k in range(1, n_est + 1):
            ensemble_preds.append(all_tree_preds[:k].mean())

        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(x=list(range(1, n_est+1)), y=all_tree_preds[:n_est], mode='markers',
            marker=dict(color='#7c6aff', size=5, opacity=0.4), name='Individual tree predictions'))
        fig_rf.add_trace(go.Scatter(x=list(range(1, n_est+1)), y=ensemble_preds, mode='lines',
            line=dict(color='#22d3a7', width=3), name='Ensemble average'))
        fig_rf.add_hline(y=true_val_rf, line_dash="dash", line_color="#f5b731",
                         annotation_text=f"True value: {true_val_rf}")
        fig_rf.update_layout(height=350, title=f"Random Forest: {n_est} trees",
                             xaxis_title="Tree #", yaxis_title="Prediction", **DARK_LAYOUT)
        st.plotly_chart(fig_rf, use_container_width=True, config={"displayModeBar": False})

        mc = st.columns(3)
        mc[0].metric("Ensemble prediction", f"{ensemble_preds[-1]:.1f}")
        mc[1].metric("Error", f"{abs(ensemble_preds[-1] - true_val_rf):.1f}")
        mc[2].metric("Std of individual trees", f"{all_tree_preds[:n_est].std():.1f}")

        st.markdown(f"""<div class="analogy-box">
        💡 With <b>{n_est} trees</b>, the ensemble prediction is {ensemble_preds[-1]:.1f} (true = {true_val_rf}).
        {'Only 1 tree -- no averaging, high variance!' if n_est == 1 else f'Notice how the green line stabilizes as you add more trees. After ~50-100 trees, adding more barely helps -- diminishing returns.'}
        </div>""", unsafe_allow_html=True)

        # max_features effect
        st.markdown("#### 🎮 Key Parameter: max_features (Randomness per Split)")
        st.caption("Controls how different each tree is from the others.")

        max_feat = st.radio("max_features:", [
            "All features (no randomness -- trees are similar)",
            "sqrt(n) -- default, good balance",
            "log2(n) -- more randomness",
            "1 feature -- maximum randomness",
        ], key="rf_max_feat", horizontal=True)

        feat_desc = {
            "All": "Each tree considers ALL features at every split. Trees end up very similar to each other, so averaging helps less. Closer to a single deep tree.",
            "sqrt": "Each tree considers only sqrt(n) random features per split. This is the <b>default and usually best</b>. Trees are different enough that averaging is powerful.",
            "log2": "Even fewer features per split. Trees are very different. Good when you have many correlated features.",
            "1 feature": "Extreme randomness. Each split uses only 1 random feature. Called 'Extra Trees.' Very fast, very random.",
        }
        key = "All" if "All" in max_feat else "sqrt" if "sqrt" in max_feat else "log2" if "log2" in max_feat else "1 feature"
        st.markdown(f"""<div class="story-box">{feat_desc[key]}</div>""", unsafe_allow_html=True)

        # RF parameters table
        st.markdown("#### 📋 All Key Parameters")
        rf_params = pd.DataFrame({
            "Parameter": ["n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf", "bootstrap"],
            "What it controls": [
                "Number of trees in the forest",
                "Max depth of each tree",
                "Features considered per split",
                "Min samples to split a node",
                "Min samples in a leaf",
                "Whether to use bootstrap sampling",
            ],
            "Typical range": ["100-500", "5-20 or None", "sqrt(n), log2(n), or n", "2-20", "1-10", "True (default)"],
            "Impact": [
                "More = more stable, slower. Diminishing returns after ~200.",
                "Deeper = more complex. None = grow until pure (risk overfitting).",
                "Lower = more random trees = better averaging.",
                "Higher = simpler trees = less overfitting.",
                "Higher = simpler trees = less overfitting.",
                "True = each tree sees different data. False = all see same data.",
            ],
        })
        st.dataframe(rf_params, use_container_width=True, hide_index=True)

        st.markdown("""<div class="key-point">
        <b>🎯 Tuning strategy:</b> Start with defaults (100 trees, max_features=sqrt, no max_depth limit).
        If overfitting: reduce max_depth or increase min_samples_leaf.
        If underfitting: increase n_estimators or reduce min_samples_leaf.
        Use cross-validation to find the sweet spot.
        </div>""", unsafe_allow_html=True)

    with st.expander("🚀 **Gradient Boosting (XGBoost/LightGBM)**  --  Trees that learn from mistakes"):
        st.markdown("""<div class="story-box">
        <b>What it does:</b> Builds trees <b>sequentially</b>  --  each new tree focuses on the <b>mistakes</b>
        the previous trees made. It's like a student who reviews their wrong answers after each practice test.
        <br><br>
        <b>How it's different from Random Forest:</b>
        <br>• Random Forest: trees are built <b>independently</b> (parallel), then averaged.
        <br>• Gradient Boosting: trees are built <b>one after another</b> (sequential), each correcting the last.
        <br><br>
        <b>Analogy:</b> A team of specialists. The first doctor makes a diagnosis. The second doctor looks at
        what the first got wrong and corrects it. The third corrects the second. Each one improves on the last.
        <br><br>
        <b>When to use:</b> When you need <b>maximum accuracy</b> on tabular data. Wins most Kaggle competitions.
        <br><b>Strengths:</b> State-of-the-art accuracy, handles missing data, feature importance.
        <br><b>Weakness:</b> Slower to train, more hyperparameters to tune, can overfit if not careful.
        </div>""", unsafe_allow_html=True)

    with st.expander("👥 **K-Nearest Neighbors (KNN)**  --  You are who your neighbors are"):
        st.markdown("""<div class="story-box">
        <b>What it does:</b> To predict for a new data point, it finds the <b>K closest</b> existing data points
        and takes a vote (classification) or average (regression).
        <br><br>
        <b>Analogy:</b> You move to a new neighborhood. To guess your income, someone looks at your 5 nearest
        neighbors' incomes and averages them. "You're probably similar to the people around you."
        <br><br>
        <b>When to use:</b> Small datasets, quick prototyping, recommendation systems.
        <br><b>Strengths:</b> No training needed (it just memorizes data), intuitive, works for any shape of data.
        <br><b>Weakness:</b> Slow for large datasets (has to compare against every point), sensitive to irrelevant features and scale.
        </div>""", unsafe_allow_html=True)

    with st.expander("🧠 **Neural Networks**  --  Inspired by the brain"):
        st.markdown("""<div class="story-box">
        <b>What it does:</b> Layers of connected "neurons" that transform inputs through weights and activation
        functions. Can learn <b>extremely complex patterns</b>.
        <br><br>
        <b>Analogy:</b> A factory assembly line. Raw materials (inputs) go through multiple processing stations
        (layers). Each station transforms the material a little. By the end, raw metal becomes a finished car.
        Each layer extracts increasingly abstract features.
        <br><br>
        <b>When to use:</b> Images (CNNs), text (Transformers/LLMs), audio, video  --  anything where patterns
        are too complex for traditional algorithms.
        <br><b>Strengths:</b> Can learn any pattern given enough data. Powers modern AI (ChatGPT, self-driving cars, image recognition).
        <br><b>Weakness:</b> Needs lots of data, expensive to train, hard to interpret ("black box"), many hyperparameters.
        <br><br>
        <b>For tabular data:</b> Usually overkill. XGBoost/Random Forest often beat neural nets on spreadsheet-style data.
        Neural nets shine on <b>unstructured data</b> (images, text, audio).
        </div>""", unsafe_allow_html=True)

    # ── Interactive Parameter Playground ──
    st.markdown("### 🎛 Tree Model Parameter Playground")
    st.caption("Tune every parameter and see how it changes model behavior in real-time. This is what hyperparameter tuning looks like.")

    st.markdown("""<div class="story-box">
    Every tree model has <b>knobs you can turn</b> (hyperparameters). Turning them wrong = overfitting or underfitting.
    Turning them right = a model that generalizes well. Below you can experiment with each parameter and see the effect instantly.
    </div>""", unsafe_allow_html=True)

    # Generate a dataset
    np.random.seed(42)
    n_pg = 300
    x1_pg = np.random.uniform(0, 10, n_pg)
    x2_pg = np.random.uniform(0, 10, n_pg)
    # True boundary: a curve
    y_true_pg = (np.sin(x1_pg) + x2_pg / 5 + np.random.normal(0, 0.3, n_pg) > 1).astype(int)

    # Split into train/test
    split_idx = int(n_pg * 0.7)
    x1_train, x1_test = x1_pg[:split_idx], x1_pg[split_idx:]
    x2_train, x2_test = x2_pg[:split_idx], x2_pg[split_idx:]
    y_train, y_test = y_true_pg[:split_idx], y_true_pg[split_idx:]

    param_model = st.radio("Model:", ["🌳 Decision Tree", "🌲 Random Forest"], horizontal=True, key="pg_model")

    st.markdown("#### Tune the parameters:")

    if "Decision Tree" in param_model:
        pc1, pc2, pc3 = st.columns(3)
        pg_depth = pc1.slider("max_depth (tree depth):", 1, 15, 3, key="pg_depth",
                              help="How many levels of questions. Low=simple, High=complex.")
        pg_min_split = pc2.slider("min_samples_split:", 2, 50, 2, key="pg_min_split",
                                  help="Min data points needed to make a split. Higher=simpler tree.")
        pg_min_leaf = pc3.slider("min_samples_leaf:", 1, 30, 1, key="pg_min_leaf",
                                 help="Min data points in each leaf. Higher=simpler tree.")

        from sklearn.tree import DecisionTreeClassifier
        X_train_pg = np.column_stack([x1_train, x2_train])
        X_test_pg = np.column_stack([x1_test, x2_test])

        model = DecisionTreeClassifier(max_depth=pg_depth, min_samples_split=pg_min_split,
                                        min_samples_leaf=pg_min_leaf, random_state=42)
        model.fit(X_train_pg, y_train)
        train_acc = model.score(X_train_pg, y_train)
        test_acc = model.score(X_test_pg, y_test)
        n_leaves = model.get_n_leaves()
        tree_depth_actual = model.get_depth()

    else:  # Random Forest
        pc1, pc2, pc3, pc4 = st.columns(4)
        pg_n_trees = pc1.slider("n_estimators (trees):", 1, 100, 10, 5, key="pg_n_trees",
                                help="Number of trees. More=more stable but slower.")
        pg_depth = pc2.slider("max_depth:", 1, 15, 5, key="pg_rf_depth",
                              help="Max depth per tree.")
        pg_max_feat = pc3.selectbox("max_features:", ["sqrt", "log2", "all"], key="pg_rf_feat",
                                    help="Features per split. sqrt=default.")
        pg_min_leaf = pc4.slider("min_samples_leaf:", 1, 30, 1, key="pg_rf_leaf",
                                 help="Min samples in each leaf.")

        from sklearn.ensemble import RandomForestClassifier
        X_train_pg = np.column_stack([x1_train, x2_train])
        X_test_pg = np.column_stack([x1_test, x2_test])

        mf = pg_max_feat if pg_max_feat != "all" else None
        model = RandomForestClassifier(n_estimators=pg_n_trees, max_depth=pg_depth,
                                        max_features=mf, min_samples_leaf=pg_min_leaf, random_state=42)
        model.fit(X_train_pg, y_train)
        train_acc = model.score(X_train_pg, y_train)
        test_acc = model.score(X_test_pg, y_test)
        n_leaves = sum(t.get_n_leaves() for t in model.estimators_)
        tree_depth_actual = max(t.get_depth() for t in model.estimators_)

    # Metrics
    mc = st.columns(4)
    mc[0].metric("Train Accuracy", f"{train_acc:.1%}")
    mc[1].metric("Test Accuracy", f"{test_acc:.1%}")
    mc[2].metric("Overfit Gap", f"{(train_acc - test_acc):.1%}",
                 help="Big gap = overfitting. Small gap = good generalization.")
    mc[3].metric("Complexity", f"{n_leaves} leaves, depth {tree_depth_actual}")

    # Decision boundary visualization
    xx1, xx2 = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    preds = model.predict(grid).reshape(xx1.shape)

    fig_pg = go.Figure()
    # Decision regions as heatmap
    fig_pg.add_trace(go.Heatmap(z=preds, x=np.linspace(0, 10, 100), y=np.linspace(0, 10, 100),
                                 colorscale=[[0, 'rgba(124,106,255,0.15)'], [1, 'rgba(244,93,109,0.15)']],
                                 showscale=False, hoverinfo='skip'))
    # Training points
    for label, color, name in [(0, '#7c6aff', 'Class 0 (train)'), (1, '#f45d6d', 'Class 1 (train)')]:
        mask = y_train == label
        fig_pg.add_trace(go.Scatter(x=x1_train[mask], y=x2_train[mask], mode='markers',
            marker=dict(color=color, size=5, opacity=0.5), name=name))
    # Test points (with border)
    for label, color, name in [(0, '#7c6aff', 'Class 0 (test)'), (1, '#f45d6d', 'Class 1 (test)')]:
        mask = y_test == label
        fig_pg.add_trace(go.Scatter(x=x1_test[mask], y=x2_test[mask], mode='markers',
            marker=dict(color=color, size=8, opacity=0.8, line=dict(width=2, color='white')), name=name))

    fig_pg.update_layout(height=450, title="Decision Boundary (shaded = model prediction, dots = data)",
                         xaxis_title="Feature 1", yaxis_title="Feature 2", **DARK_LAYOUT)
    st.plotly_chart(fig_pg, use_container_width=True, config={"displayModeBar": False})

    # Diagnosis
    gap = train_acc - test_acc
    if gap > 0.15:
        st.markdown(f"""<div class="impact-box">
        ⚠️ <b>Overfitting detected!</b> Train accuracy ({train_acc:.1%}) is much higher than test ({test_acc:.1%}).
        The model memorized training data. <b>Try:</b> reduce max_depth, increase min_samples_leaf, or add more trees (Random Forest).
        </div>""", unsafe_allow_html=True)
    elif test_acc < 0.65:
        st.markdown(f"""<div class="story-box" style="border-color:#f5b731">
        ⚠️ <b>Underfitting.</b> Both train ({train_acc:.1%}) and test ({test_acc:.1%}) accuracy are low.
        The model is too simple. <b>Try:</b> increase max_depth, decrease min_samples_leaf, or use a more complex model.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="analogy-box">
        ✅ <b>Good balance!</b> Train: {train_acc:.1%}, Test: {test_acc:.1%}. The gap is small ({gap:.1%}),
        meaning the model generalizes well. The decision boundary captures the real pattern without memorizing noise.
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-point">
    <b>🎯 What to observe:</b>
    <br>- Increase max_depth: boundary gets more complex (wiggly), train accuracy goes up, but test may drop
    <br>- Increase min_samples_leaf: boundary gets smoother, less overfitting
    <br>- For Random Forest: more trees = smoother boundary, more stable predictions
    <br>- Watch the <b>overfit gap</b> -- when it grows, you have gone too far
    </div>""", unsafe_allow_html=True)

    # ── Ensemble Methods: Bagging, Boosting, AdaBoost ──
    st.markdown("### 🏗 Ensemble Methods: Bagging, Boosting & AdaBoost")
    st.caption("The most powerful idea in ML: combine many weak models into one strong model.")

    st.markdown("""<div class="story-box">
    A single decision tree is like asking <b>one person</b> for directions -- they might be wrong.
    But if you ask <b>100 people</b> and go with the majority answer, you'll almost certainly get it right.
    That's the core idea behind <b>ensemble methods</b>: combine many "weak" models into one "strong" model.
    <br><br>
    There are two main strategies: <b>Bagging</b> (train models independently, then vote) and
    <b>Boosting</b> (train models sequentially, each fixing the previous one's mistakes).
    </div>""", unsafe_allow_html=True)

    ensemble_type = st.radio("Pick an ensemble method:", [
        "🎒 Bagging (Random Forest)",
        "🚀 Boosting (Gradient Boosting / XGBoost)",
        "🎯 AdaBoost (Adaptive Boosting)",
    ], key="ensemble_type")

    if "Bagging" in ensemble_type:
        st.markdown("""<div class="story-box" style="border-left:4px solid #7c6aff">
        <b style="color:#7c6aff;font-size:1.1rem">🎒 Bagging (Bootstrap Aggregating)</b>
        <br><br>
        <b>How it works:</b>
        <br>1. Take your training data and create <b>many random subsets</b> (with replacement -- some rows repeat, some are left out)
        <br>2. Train a <b>separate model</b> (usually a decision tree) on each subset
        <br>3. For prediction: each model votes, and the <b>majority wins</b> (classification) or <b>average</b> (regression)
        <br><br>
        <b>Why it works:</b> Each tree overfits differently (because of random subsets). When you average them,
        the random errors <b>cancel out</b>, leaving only the real signal. This <b>reduces variance</b> dramatically.
        <br><br>
        <b>Random Forest = Bagging + random feature selection.</b> At each split, the tree only considers a
        random subset of features. This makes the trees even more different from each other, which makes
        the averaging even more effective.
        <br><br>
        <div style="text-align:center;font-size:1.1rem;padding:0.6rem;background:rgba(124,106,255,0.08);border-radius:10px">
        <b>Final prediction = Average(Tree1, Tree2, ..., TreeN)</b>
        </div>
        </div>""", unsafe_allow_html=True)

        # Interactive bagging demo
        st.markdown("#### 🎮 See Bagging in Action")
        st.caption("Each tree makes a different prediction. The average is more stable than any individual tree.")

        n_trees = st.slider("Number of trees:", 1, 50, 10, 1, key="bag_trees")
        np.random.seed(42)
        true_val = 75.0
        tree_preds = np.random.normal(true_val, 12, n_trees)
        running_avg = np.cumsum(tree_preds) / np.arange(1, n_trees + 1)

        fig_bag = go.Figure()
        fig_bag.add_trace(go.Scatter(x=list(range(1, n_trees+1)), y=tree_preds, mode='markers',
            marker=dict(color='#7c6aff', size=8, opacity=0.6), name='Individual tree predictions'))
        fig_bag.add_trace(go.Scatter(x=list(range(1, n_trees+1)), y=running_avg, mode='lines+markers',
            line=dict(color='#22d3a7', width=3), marker=dict(size=5), name='Running average (ensemble)'))
        fig_bag.add_hline(y=true_val, line_dash="dash", line_color="#f5b731", annotation_text=f"True value: {true_val}")
        fig_bag.update_layout(height=350, title=f"Bagging: {n_trees} trees -- individual predictions vs ensemble average",
                              xaxis_title="Tree #", yaxis_title="Prediction", **DARK_LAYOUT)
        st.plotly_chart(fig_bag, use_container_width=True, config={"displayModeBar": False})

        mc = st.columns(3)
        mc[0].metric("Individual tree error (avg)", f"{np.mean(np.abs(tree_preds - true_val)):.1f}")
        mc[1].metric("Ensemble error", f"{abs(running_avg[-1] - true_val):.1f}")
        mc[2].metric("Improvement", f"{np.mean(np.abs(tree_preds - true_val)) / max(abs(running_avg[-1] - true_val), 0.1):.1f}x better")

        st.markdown("""<div class="analogy-box">
        💡 <b>Notice:</b> Individual trees (purple dots) are scattered wildly. But the running average (green line)
        converges toward the true value. More trees = more stable prediction. This is the <b>wisdom of crowds</b>.
        </div>""", unsafe_allow_html=True)

    elif "Boosting" in ensemble_type:
        st.markdown("""<div class="story-box" style="border-left:4px solid #22d3a7">
        <b style="color:#22d3a7;font-size:1.1rem">🚀 Boosting (Gradient Boosting / XGBoost / LightGBM)</b>
        <br><br>
        <b>How it works:</b>
        <br>1. Train a <b>weak model</b> (small tree) on the data
        <br>2. Look at the <b>errors</b> (residuals) -- where did it get wrong?
        <br>3. Train the <b>next model</b> specifically on those errors
        <br>4. Repeat: each new model <b>corrects the mistakes</b> of all previous models combined
        <br>5. Final prediction = sum of all models' predictions
        <br><br>
        <b>Key difference from Bagging:</b>
        <br>- Bagging: trees are built <b>independently</b> (parallel), then averaged. Reduces <b>variance</b>.
        <br>- Boosting: trees are built <b>sequentially</b> (one after another), each fixing errors. Reduces <b>bias</b>.
        <br><br>
        <div style="text-align:center;font-size:1.1rem;padding:0.6rem;background:rgba(34,211,167,0.08);border-radius:10px">
        <b>Final = Model1 + correction2 + correction3 + ... + correctionN</b>
        </div>
        <br>
        <b>XGBoost</b> and <b>LightGBM</b> are optimized implementations of gradient boosting.
        They add regularization to prevent overfitting and are extremely fast.
        <b>XGBoost wins most Kaggle competitions</b> on tabular data.
        </div>""", unsafe_allow_html=True)

        # Interactive boosting demo
        st.markdown("#### 🎮 See Boosting in Action")
        st.caption("Each round reduces the error. Watch the residuals shrink.")

        n_rounds = st.slider("Boosting rounds:", 1, 20, 5, 1, key="boost_rounds")
        np.random.seed(42)
        x_boost = np.linspace(0, 10, 50)
        y_true_boost = np.sin(x_boost) * 3 + x_boost * 0.5
        y_pred = np.zeros_like(x_boost)
        residuals_history = []

        for r in range(n_rounds):
            residuals = y_true_boost - y_pred
            residuals_history.append(np.mean(residuals**2))
            # Simple correction: fit a small step toward residuals
            correction = residuals * 0.3
            y_pred = y_pred + correction

        fig_boost = go.Figure()
        fig_boost.add_trace(go.Scatter(x=x_boost, y=y_true_boost, mode='markers',
            marker=dict(color='#22d3a7', size=6), name='True values'))
        fig_boost.add_trace(go.Scatter(x=x_boost, y=y_pred, mode='lines',
            line=dict(color='#f5b731', width=3), name=f'Prediction after {n_rounds} rounds'))
        fig_boost.update_layout(height=300, title=f"Boosting: {n_rounds} rounds of correction",
                                xaxis_title="X", yaxis_title="Y", **DARK_LAYOUT)
        st.plotly_chart(fig_boost, use_container_width=True, config={"displayModeBar": False})

        # Error reduction chart
        fig_err = go.Figure(go.Scatter(x=list(range(1, n_rounds+1)), y=residuals_history,
            mode='lines+markers', line=dict(color='#f45d6d', width=2), marker=dict(size=7)))
        fig_err.update_layout(height=200, title="Error decreases with each round",
                              xaxis_title="Round", yaxis_title="Mean Squared Error", **DARK_LAYOUT)
        st.plotly_chart(fig_err, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""<div class="analogy-box">
        💡 <b>Notice:</b> Each round, the prediction (yellow line) gets closer to the true data (green dots),
        and the error drops. Each new model focuses on what the previous ones got wrong.
        This is like a student reviewing their wrong answers after each practice test.
        </div>""", unsafe_allow_html=True)

    else:  # AdaBoost
        st.markdown("""<div class="story-box" style="border-left:4px solid #f5b731">
        <b style="color:#f5b731;font-size:1.1rem">🎯 AdaBoost (Adaptive Boosting)</b>
        <br><br>
        <b>How it works:</b>
        <br>1. Train a weak model on the data (all data points have <b>equal weight</b>)
        <br>2. Check which points it got <b>wrong</b>
        <br>3. <b>Increase the weight</b> of misclassified points (make them "louder")
        <br>4. Train the next model -- it now pays <b>more attention</b> to the hard cases
        <br>5. Repeat. Final prediction = weighted vote of all models
        <br><br>
        <b>Key difference from Gradient Boosting:</b>
        <br>- Gradient Boosting: fits new models to <b>residuals</b> (errors)
        <br>- AdaBoost: <b>reweights data points</b> so hard cases get more attention
        <br><br>
        <b>Analogy:</b> A teacher gives a test. Students who fail get <b>extra tutoring</b> (more weight).
        The next test focuses more on the topics those students struggled with. Each round, the class
        gets better at the hard problems.
        <br><br>
        <div style="text-align:center;font-size:1.1rem;padding:0.6rem;background:rgba(245,183,49,0.08);border-radius:10px">
        <b>Final = w1*Model1 + w2*Model2 + ... + wN*ModelN</b>
        <br><span style="font-size:0.85rem;color:#8892b0">where w = model weight (better models get higher weight)</span>
        </div>
        </div>""", unsafe_allow_html=True)

        # Interactive AdaBoost demo
        st.markdown("#### 🎮 See AdaBoost: Watch Misclassified Points Get Bigger")
        st.caption("Red = misclassified. Their size grows each round as AdaBoost pays more attention to them.")

        ada_round = st.slider("AdaBoost round:", 1, 5, 1, key="ada_round")
        np.random.seed(42)
        n_pts = 60
        x_ada = np.random.randn(n_pts)
        y_ada = np.random.randn(n_pts)
        labels = ((x_ada + y_ada) > 0).astype(int)
        # Simulate misclassification: points near the boundary
        distance_to_boundary = np.abs(x_ada + y_ada)
        misclassified = distance_to_boundary < (0.8 / ada_round)  # fewer misclassified as rounds increase
        weights = np.ones(n_pts) * 5
        weights[misclassified] = 5 + ada_round * 8  # misclassified get bigger

        fig_ada = go.Figure()
        for label, color, name in [(0, '#7c6aff', 'Class 0'), (1, '#22d3a7', 'Class 1')]:
            mask = labels == label
            fig_ada.add_trace(go.Scatter(x=x_ada[mask], y=y_ada[mask], mode='markers',
                marker=dict(
                    color=[('#f45d6d' if m else color) for m in misclassified[mask]],
                    size=weights[mask], opacity=0.7,
                ), name=name))
        # Boundary
        bx_ada = np.linspace(-3, 3, 100)
        fig_ada.add_trace(go.Scatter(x=bx_ada, y=-bx_ada, mode='lines',
            line=dict(color='#f5b731', width=2, dash='dash'), name='Decision boundary'))
        fig_ada.update_layout(height=380, title=f"AdaBoost Round {ada_round}: Red = misclassified (bigger = more weight)",
                              **DARK_LAYOUT)
        st.plotly_chart(fig_ada, use_container_width=True, config={"displayModeBar": False})

        mc = st.columns(2)
        mc[0].metric("Misclassified points", int(misclassified.sum()))
        mc[1].metric("Their weight", f"{5 + ada_round * 8}x normal")

        st.markdown(f"""<div class="analogy-box">
        💡 <b>Round {ada_round}:</b> The red points are the ones the model got wrong. Notice they're <b>bigger</b>
        -- AdaBoost is telling the next model: "Pay extra attention to these!" As rounds increase,
        fewer points are misclassified because each model focuses on the hard cases.
        </div>""", unsafe_allow_html=True)

    # Ensemble comparison
    st.markdown("### ⚖ Bagging vs Boosting vs AdaBoost")
    ensemble_compare = pd.DataFrame({
        "": ["Strategy", "How models are built", "What it reduces", "Risk", "Speed", "Best implementation"],
        "Bagging": ["Train independently, then average", "Parallel (all at once)", "Variance (overfitting)", "Low -- hard to overfit", "Fast (parallelizable)", "Random Forest"],
        "Gradient Boosting": ["Each model fixes previous errors", "Sequential (one after another)", "Bias (underfitting)", "Medium -- can overfit", "Slower (sequential)", "XGBoost, LightGBM"],
        "AdaBoost": ["Reweight hard examples", "Sequential (one after another)", "Bias", "Medium -- sensitive to noise", "Medium", "AdaBoostClassifier"],
    })
    st.dataframe(ensemble_compare, use_container_width=True, hide_index=True)

    # Algorithm comparison
    st.markdown("### 📋 Quick Comparison: When to Use What")
    algo_compare = pd.DataFrame({
        "Algorithm": ["Linear/Logistic Reg.", "Decision Tree", "Random Forest", "XGBoost", "KNN", "Neural Network"],
        "Best For": ["Simple relationships", "Interpretability", "General tabular data", "Max accuracy (tabular)", "Small data, prototyping", "Images, text, audio"],
        "Interpretable?": ["✅ Yes", "✅ Yes", "⚠️ Moderate", "⚠️ Moderate", "✅ Yes", "❌ No (black box)"],
        "Speed": ["⚡ Fast", "⚡ Fast", "🔵 Medium", "🔵 Medium", "🐌 Slow (large data)", "🐌 Slow to train"],
        "Overfitting Risk": ["Low", "High", "Low", "Medium", "Medium", "High"],
    })
    st.dataframe(algo_compare, use_container_width=True, hide_index=True)

    st.markdown("""<div class="key-point">
    <b>🎯 Rule of thumb:</b> Start with <b>Logistic Regression</b> (baseline). If not good enough, try
    <b>Random Forest</b>. If you need max accuracy, try <b>XGBoost</b>. Only use neural networks for
    images/text/audio or when you have massive datasets.
    </div>""", unsafe_allow_html=True)

    # Decision boundary visualization
    st.markdown("### 🗺️ How Different Algorithms Draw Decision Boundaries")
    st.caption("Each algorithm separates classes differently. See how the boundary changes.")

    np.random.seed(42)
    n = 200
    x0 = np.random.normal(20, 8, n // 2)
    y0 = np.random.normal(40, 12, n // 2)
    x1 = np.random.normal(45, 10, n // 2)
    y1 = np.random.normal(80, 15, n // 2)

    boundary_type = st.radio("Which algorithm draws the boundary?", [
        "📏 Logistic Regression (straight line)",
        "🌳 Decision Tree (axis-aligned splits)",
        "🌲 Random Forest / Neural Net (smooth curve)",
    ], horizontal=True, key="boundary")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x0, y=y0, mode='markers', marker=dict(color='#22d3a7', size=6, opacity=0.6), name='Stayed ✅'))
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', marker=dict(color='#f45d6d', size=6, opacity=0.6), name='Churned ❌'))

    bx = np.linspace(0, 70, 100)
    if "Logistic" in boundary_type:
        by = 1.2 * bx + 10
        boundary_desc = "Logistic Regression draws a <b>straight line</b>. Simple and interpretable, but can't capture curved patterns. If the real boundary is curved, it'll misclassify points near the bend."
    elif "Decision Tree" in boundary_type:
        # Axis-aligned splits (staircase)
        by = np.where(bx < 32, 55, np.where(bx < 50, 70, 90))
        boundary_desc = "Decision Trees make <b>axis-aligned splits</b> (horizontal/vertical cuts). The boundary looks like a staircase. Each split asks one question about one feature: 'Is tenure > 32?' then 'Is charge > 70?'"
    else:
        by = 0.03 * (bx - 35)**2 + 40
        boundary_desc = "Random Forests and Neural Networks can draw <b>smooth curves</b>. They capture complex, non-linear patterns. More accurate, but harder to explain to stakeholders."

    fig.add_trace(go.Scatter(x=bx, y=by, mode='lines', line=dict(color='#f5b731', width=3, dash='dash'), name='Decision Boundary'))
    fig.update_layout(height=400, title="How This Algorithm Separates the Classes",
                      xaxis_title="Tenure (months)", yaxis_title="Monthly Charges ($)", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="analogy-box">💡 {boundary_desc}</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-point">
    <b>🎯 Key Takeaway:</b> Supervised learning needs <b>labeled data</b> (the answer key). The quality and quantity
    of your labels determines how good your model can be. Garbage labels in -> garbage predictions out.
    </div>""", unsafe_allow_html=True)

    iq([
        {"q": "What's the difference between classification and regression?", "d": "Easy", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>Classification:</b> Predicting a category (Yes/No, Spam/Not Spam, Cat/Dog). Output is a label. <b>Regression:</b> Predicting a number (house price, temperature, revenue). Output is continuous. <b>Example:</b> 'Will this customer churn?' = classification. 'How much will this customer spend?' = regression.",
         "t": "Give one example of each. Interviewers want to see you can identify which is which."},
        {"q": "Explain the difference between logistic regression and linear regression.", "d": "Medium", "c": ["Google", "Meta", "Netflix"],
         "a": "<b>Linear regression:</b> Predicts a continuous number. Output can be any value. y = mx + b. <b>Logistic regression:</b> Predicts a probability (0 to 1) for classification. Uses a sigmoid function to squash output between 0 and 1. Despite the name, it's a CLASSIFICATION algorithm, not regression. <b>When to use:</b> Linear for 'how much?' Logistic for 'yes or no?'",
         "t": "Emphasize that logistic regression is for classification despite having 'regression' in the name."},
        {"q": "What is a decision boundary? Draw one.", "d": "Medium", "c": ["Google", "Apple"],
         "a": "A decision boundary is the line (or surface) that separates different classes in feature space. Points on one side get classified as Class A, points on the other as Class B. <b>Linear models</b> draw straight lines. <b>Trees and neural nets</b> can draw complex curves. <b>Key:</b> The model's job is to find the best position for this boundary that minimizes misclassification.",
         "t": "Sketch it  --  draw two clusters of dots and a line between them."},
        {"q": "Compare Random Forest vs Logistic Regression. When would you use each?", "d": "Medium", "c": ["Amazon", "Google", "Netflix"],
         "a": "<b>Logistic Regression:</b> Simple, fast, interpretable (coefficients tell a story). Best when relationship is roughly linear and you need to explain the model. <b>Random Forest:</b> Handles non-linear relationships, feature interactions, outliers. More accurate but less interpretable. <b>My approach:</b> Start with logistic regression as baseline. If it's not good enough, try Random Forest. If interpretability matters (healthcare, finance), stick with logistic.",
         "t": "Saying 'I start with a simple baseline' shows maturity."},
        {"q": "What is the precision-recall tradeoff?", "d": "Hard", "c": ["Meta", "Google", "Netflix"],
         "a": "<b>Precision:</b> Of all items I predicted as positive, how many actually are? (Don't cry wolf.) <b>Recall:</b> Of all actual positives, how many did I catch? (Don't miss any.) <b>Tradeoff:</b> Increasing one usually decreases the other. <b>Example:</b> Cancer screening  --  high recall (catch every cancer) means more false alarms (low precision). Spam filter  --  high precision (don't flag real emails) means some spam gets through (low recall). <b>Which matters more?</b> Depends on the cost of each error.",
         "t": "Always say 'it depends on the business context' and give an example where each matters more."},
        {"q": "How do you handle imbalanced classes? (e.g., 95% non-fraud, 5% fraud)", "d": "Hard", "c": ["Amazon", "Google", "Meta"],
         "a": "<b>Problem:</b> Model predicts 'not fraud' for everything and gets 95% accuracy  --  useless. <b>Solutions:</b> (1) <b>Resampling:</b> Oversample minority (SMOTE) or undersample majority. (2) <b>Class weights:</b> Tell the model that misclassifying fraud is 20× worse. (3) <b>Different metric:</b> Use F1, precision-recall AUC instead of accuracy. (4) <b>Anomaly detection:</b> Treat fraud as anomaly, not classification. (5) <b>Threshold tuning:</b> Lower the classification threshold to catch more fraud.",
         "t": "Mention that accuracy is misleading for imbalanced data  --  use F1 or AUC instead."},
    ])


# ═══════════════════════════════════════════════════
# UNSUPERVISED LEARNING
# ═══════════════════════════════════════════════════
elif concept == "🧩 Unsupervised Learning":
    st.markdown('<div class="section-header">🧩 Unsupervised Learning  --  Finding Hidden Patterns</div>', unsafe_allow_html=True)

    st.markdown("""<div class="story-box">
    Imagine you're given a <b>box of 1,000 buttons</b>  --  different sizes, colors, shapes, materials.
    Nobody tells you how to sort them. There are no labels. But naturally, you start grouping:
    "these are the big red ones," "these are the small metal ones," "these are the shiny round ones."
    <br><br>
    That's <b>unsupervised learning</b>. No teacher. No answer key. The machine looks at the data and
    discovers <b>hidden structure</b> on its own.
    <br><br>
    It answers the question: <b>"What natural groups exist in this data that I haven't noticed?"</b>
    </div>""", unsafe_allow_html=True)

    # Supervised vs Unsupervised comparison
    st.markdown("### ⚖️ Supervised vs Unsupervised  --  Side by Side")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="story-box" style="border-color:#7c6aff">
        <b style="color:#7c6aff">🎓 Supervised</b><br><br>
        ✅ Has labels (answer key)<br>
        ✅ "Predict this specific thing"<br>
        ✅ Clear right/wrong answers<br>
        ✅ Easy to measure accuracy<br><br>
        <b>Like:</b> Studying with an answer key<br>
        <b>Goal:</b> Predict a known target
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="story-box" style="border-color:#22d3a7">
        <b style="color:#22d3a7">🧩 Unsupervised</b><br><br>
        ❌ No labels<br>
        ✅ "Find interesting patterns"<br>
        ❌ No clear right/wrong<br>
        ⚠️ Harder to evaluate<br><br>
        <b>Like:</b> Exploring a new city without a map<br>
        <b>Goal:</b> Discover hidden structure
        </div>""", unsafe_allow_html=True)

    # Interactive: Clustering demo
    st.markdown("### 🎮 Try It: Customer Segmentation (Clustering)")
    st.caption("Adjust the number of clusters and watch the algorithm group customers automatically.")

    np.random.seed(42)
    n = 300
    # Generate 3 natural clusters
    c1_x, c1_y = np.random.normal(25, 8, 100), np.random.normal(70, 12, 100)
    c2_x, c2_y = np.random.normal(55, 10, 100), np.random.normal(30, 10, 100)
    c3_x, c3_y = np.random.normal(70, 8, 100), np.random.normal(80, 10, 100)

    all_x = np.concatenate([c1_x, c2_x, c3_x])
    all_y = np.concatenate([c1_y, c2_y, c3_y])

    n_clusters = st.slider("Number of clusters (K):", 2, 6, 3, key="kmeans_k")

    # Simple K-means implementation
    from numpy.random import choice
    centers_idx = choice(len(all_x), n_clusters, replace=False)
    centers_x = all_x[centers_idx].copy()
    centers_y = all_y[centers_idx].copy()

    for _ in range(20):  # iterations
        # Assign points to nearest center
        distances = np.array([np.sqrt((all_x - cx)**2 + (all_y - cy)**2) for cx, cy in zip(centers_x, centers_y)])
        labels = distances.argmin(axis=0)
        # Update centers
        for k in range(n_clusters):
            if (labels == k).sum() > 0:
                centers_x[k] = all_x[labels == k].mean()
                centers_y[k] = all_y[labels == k].mean()

    cluster_colors = ['#7c6aff', '#22d3a7', '#f5b731', '#f45d6d', '#e879a8', '#5eaeff']
    cluster_names = ['Segment A', 'Segment B', 'Segment C', 'Segment D', 'Segment E', 'Segment F']

    fig = go.Figure()
    for k in range(n_clusters):
        mask = labels == k
        fig.add_trace(go.Scatter(
            x=all_x[mask], y=all_y[mask], mode='markers',
            marker=dict(color=cluster_colors[k], size=6, opacity=0.6),
            name=cluster_names[k],
        ))
        # Center
        fig.add_trace(go.Scatter(
            x=[centers_x[k]], y=[centers_y[k]], mode='markers',
            marker=dict(color=cluster_colors[k], size=18, symbol='x', line=dict(width=2, color='white')),
            name=f'{cluster_names[k]} Center', showlegend=False,
        ))

    fig.update_layout(height=450, title=f"K-Means Clustering (K={n_clusters})",
                      xaxis_title="Monthly Spending ($)", yaxis_title="Satisfaction Score", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Cluster profiles
    st.markdown("#### 📊 Cluster Profiles")
    profile_data = []
    for k in range(n_clusters):
        mask = labels == k
        profile_data.append({
            "Segment": cluster_names[k],
            "Size": int(mask.sum()),
            "Avg Spending": f"${all_x[mask].mean():.0f}",
            "Avg Satisfaction": f"{all_y[mask].mean():.0f}",
        })
    st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="analogy-box">
    💡 With <b>K={n_clusters}</b>, the algorithm found {n_clusters} natural groups. Try changing K:
    <br>• <b>K=2</b>: Very broad segments (high-level view)
    <br>• <b>K=3</b>: Usually the "sweet spot" for this data
    <br>• <b>K=5+</b>: Very specific segments (might be over-splitting)
    <br><br>
    In business, these segments become <b>marketing strategies</b>: "Segment A gets discount offers,
    Segment B gets premium upsells, Segment C gets loyalty rewards."
    </div>""", unsafe_allow_html=True)

    # Types of unsupervised learning
    st.markdown("### 📚 Types of Unsupervised Learning")

    st.markdown("### 📚 Types of Unsupervised Learning")

    with st.expander("🎯 **Clustering** -- Group similar things together", expanded=True):
        st.markdown("""<div class="story-box">
        <b>K-Means:</b> Divides data into K groups based on distance to cluster centers.
        <br><b>How it works:</b> (1) Pick K random centers. (2) Assign each point to nearest center.
        (3) Move centers to the middle of their group. (4) Repeat until stable.
        <br><b>Analogy:</b> Sorting a pile of clothes into K piles by similarity -- keep rearranging until each pile makes sense.
        <br><b>Weakness:</b> You must choose K in advance. Assumes round-shaped clusters.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="story-box">
        <b>Hierarchical Clustering:</b> Builds a tree (dendrogram) of clusters by progressively merging the most similar groups.
        <br><b>How it works:</b> Start with each point as its own cluster. Merge the two closest clusters. Repeat until everything is one big cluster. Cut the tree at the desired level to get K groups.
        <br><b>Analogy:</b> A family tree -- start with individuals, merge into families, then neighborhoods, then cities.
        <br><b>Strength:</b> No need to choose K upfront -- you can cut the tree at any level. Shows relationships between clusters.
        <br><b>Weakness:</b> Slow for large datasets. Cannot undo a merge once made.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="story-box">
        <b>DBSCAN (Density-Based):</b> Finds clusters as dense regions separated by sparse areas.
        <br><b>How it works:</b> A point is a "core point" if it has enough neighbors within a radius. Core points that are close together form a cluster. Points with no nearby core points are labeled as noise/outliers.
        <br><b>Analogy:</b> Finding cities on a satellite photo at night -- bright dense areas are cities, dark sparse areas are countryside. Isolated lights are outliers.
        <br><b>Strength:</b> Finds clusters of any shape (not just round). Automatically detects outliers. No need to specify K.
        <br><b>Weakness:</b> Sensitive to the radius parameter. Struggles when clusters have very different densities.
        </div>""", unsafe_allow_html=True)

    with st.expander("📉 **Dimensionality Reduction** -- Simplify without losing meaning"):
        st.markdown("""<div class="story-box">
        <b>PCA (Principal Component Analysis):</b> Finds the directions in data that capture the most variation, then projects data onto those directions.
        <br><b>Analogy:</b> You have a 500-page book. PCA finds the 10 most important themes and summarizes the book using just those themes. You lose some detail, but keep the essence.
        <br><b>Math intuition:</b> If you have 50 features, many are correlated (redundant). PCA finds new "super-features" (principal components) that are combinations of the originals, ranked by how much variation they explain. PC1 explains the most, PC2 the second most, etc.
        <br><b>When to use:</b> Too many features (curse of dimensionality), features are correlated, need to visualize high-dimensional data, speed up model training.
        <br><b>Tradeoff:</b> Components are hard to interpret -- PC1 is a mix of original features, not a single meaningful variable.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="story-box">
        <b>t-SNE / UMAP:</b> Non-linear methods that squash high-dimensional data into 2D or 3D for visualization.
        <br><b>How they work:</b> They try to preserve the "neighborhood" structure -- points that are close in high dimensions should be close in 2D. Points that are far apart should stay far apart.
        <br><b>Analogy:</b> Projecting a 3D globe onto a flat map. You cannot do it perfectly (some distortion is inevitable), but you can preserve the relative positions of countries.
        <br><b>t-SNE vs UMAP:</b> UMAP is faster and better preserves global structure. t-SNE is better at showing local clusters but can distort distances between clusters.
        <br><b>When to use:</b> Visualizing clusters in high-dimensional data (e.g., "do my customer segments actually look different?").
        <br><b>Warning:</b> These are for visualization only -- do not use the 2D output as features for a model.
        </div>""", unsafe_allow_html=True)

    with st.expander("🔍 **Anomaly Detection** -- Find the weird ones"):
        st.markdown("""<div class="story-box">
        <b>Isolation Forest:</b> Randomly splits data with decision trees. Anomalies are easier to isolate (fewer splits needed) because they are far from everything else.
        <br><b>Analogy:</b> In a crowd of people standing together, the person standing alone in the corner is easy to "isolate" -- you only need to draw one line to separate them. Normal people in the crowd need many lines.
        <br><b>Strength:</b> Fast, works in high dimensions, no need to define "normal."
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="story-box">
        <b>Autoencoders (Neural Network-based):</b> A neural network learns to compress data into a small representation and then reconstruct it. Normal data reconstructs well. Anomalies reconstruct poorly (high reconstruction error).
        <br><b>Analogy:</b> A photocopier that works great for standard documents but produces garbled copies of unusual documents. The garbled output = anomaly detected.
        <br><b>When to use:</b> Complex data (images, time series, network traffic) where simple methods fail.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="story-box">
        <b>Use cases across industries:</b>
        <br>- <b>Finance:</b> Fraud detection (unusual transaction patterns)
        <br>- <b>Cybersecurity:</b> Network intrusion (abnormal traffic)
        <br>- <b>Manufacturing:</b> Defective products (sensor readings outside normal range)
        <br>- <b>Healthcare:</b> Unusual vital signs (early warning of deterioration)
        <br>- <b>Telecom:</b> Cell towers with abnormal dropped call rates
        </div>""", unsafe_allow_html=True)

    # Interactive: Anomaly detection
    st.markdown("### 🎮 Try It: Spot the Anomalies")
    st.caption("Normal transactions cluster together. Fraudulent ones stand out. Adjust the threshold.")

    np.random.seed(42)
    normal_x = np.random.normal(50, 12, 200)
    normal_y = np.random.normal(50, 12, 200)
    fraud_x = np.random.uniform(0, 100, 10)
    fraud_y = np.random.uniform(0, 100, 10)
    # Make fraud points far from center
    fraud_x = np.where(fraud_x > 50, fraud_x + 30, fraud_x - 30)
    fraud_y = np.where(fraud_y > 50, fraud_y + 30, fraud_y - 30)

    all_pts_x = np.concatenate([normal_x, fraud_x])
    all_pts_y = np.concatenate([normal_y, fraud_y])

    threshold = st.slider("Detection threshold (distance from center):", 10, 50, 30, key="anomaly_thresh")

    center_x, center_y = normal_x.mean(), normal_y.mean()
    distances = np.sqrt((all_pts_x - center_x)**2 + (all_pts_y - center_y)**2)
    detected = distances > threshold

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=all_pts_x[~detected], y=all_pts_y[~detected], mode='markers',
        marker=dict(color='#22d3a7', size=5, opacity=0.5), name='Normal',
    ))
    fig_anom.add_trace(go.Scatter(
        x=all_pts_x[detected], y=all_pts_y[detected], mode='markers',
        marker=dict(color='#f45d6d', size=10, symbol='x'), name='Detected Anomaly',
    ))
    # Threshold circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig_anom.add_trace(go.Scatter(
        x=center_x + threshold * np.cos(theta), y=center_y + threshold * np.sin(theta),
        mode='lines', line=dict(color='#f5b731', width=2, dash='dash'), name='Threshold',
    ))
    fig_anom.update_layout(height=400, title=f"Anomaly Detection  --  {detected.sum()} flagged",
                           xaxis_title="Transaction Amount", yaxis_title="Transaction Frequency", **DARK_LAYOUT)
    st.plotly_chart(fig_anom, use_container_width=True, config={"displayModeBar": False})

    actual_fraud = np.concatenate([np.zeros(200), np.ones(10)])
    true_positives = int((detected & (actual_fraud == 1)).sum())
    false_positives = int((detected & (actual_fraud == 0)).sum())

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Fraud Caught", f"{true_positives}/10")
    mc2.metric("False Alarms", f"{false_positives}")
    mc3.metric("Precision", f"{true_positives / max(detected.sum(), 1) * 100:.0f}%")

    st.markdown(f"""<div class="analogy-box">
    💡 <b>The tradeoff:</b> A tight threshold (small circle) catches fewer frauds but has fewer false alarms.
    A loose threshold (big circle) catches more fraud but flags innocent transactions too.
    This is the <b>precision-recall tradeoff</b>  --  one of the most important concepts in ML.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-point">
    <b>🎯 Key Takeaway:</b> Unsupervised learning finds <b>structure you didn't know existed</b>.
    It's like having a detective who doesn't need clues  --  they just look at the evidence and say
    "I see 3 groups here, and these 5 data points don't belong to any of them."
    It's exploration, not prediction.
    </div>""", unsafe_allow_html=True)

    # Final comparison table
    st.markdown("### 📋 Quick Reference: Supervised vs Unsupervised")

    comparison = pd.DataFrame({
        "Aspect": ["Labels needed?", "Goal", "Output", "Evaluation", "Example", "Analogy"],
        "Supervised 🎓": ["Yes", "Predict a target", "Prediction / Classification", "Accuracy, F1, RMSE", "Spam detection", "Studying with answer key"],
        "Unsupervised 🧩": ["No", "Find patterns", "Clusters / Anomalies", "Silhouette, visual inspection", "Customer segmentation", "Exploring without a map"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    iq([
        {"q": "What's the difference between supervised and unsupervised learning?", "d": "Easy", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>Supervised:</b> Has labeled data (input + correct answer). Goal: predict a known target. Example: spam detection (emails labeled spam/not spam). <b>Unsupervised:</b> No labels. Goal: find hidden patterns. Example: customer segmentation (group customers by behavior without predefined groups). <b>Key:</b> Supervised = 'teach me with examples.' Unsupervised = 'find structure on your own.'",
         "t": "Give one concrete example of each."},
        {"q": "Explain K-Means clustering to a non-technical person.", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "Imagine you have 1000 customers and want to group them into segments. K-Means: (1) Pick K random 'center points.' (2) Assign each customer to the nearest center. (3) Move each center to the middle of its group. (4) Repeat until stable. <b>Result:</b> K groups of similar customers. <b>Analogy:</b> Like sorting a pile of clothes into K piles by similarity  --  you keep rearranging until each pile makes sense.",
         "t": "Walk through the steps visually. Interviewers love when you can explain algorithms simply."},
        {"q": "How do you choose the right K in K-Means?", "d": "Medium", "c": ["Google", "Netflix", "Uber"],
         "a": "<b>Elbow method:</b> Plot total within-cluster distance vs K. Look for the 'elbow' where adding more clusters stops helping much. <b>Silhouette score:</b> Measures how similar each point is to its own cluster vs other clusters. Higher = better. <b>Business logic:</b> Sometimes K is determined by the business  --  'we want 3 customer tiers' or 'we need 5 market segments.' <b>My approach:</b> Try elbow method first, validate with silhouette, then check if clusters make business sense.",
         "t": "Mention business logic  --  shows you think beyond just the math."},
        {"q": "What is anomaly detection? Give a real-world example.", "d": "Easy", "c": ["Amazon", "Apple", "Google"],
         "a": "Finding data points that are significantly different from the majority. <b>Examples:</b> (1) Credit card fraud  --  unusual spending patterns. (2) Network intrusion  --  abnormal traffic. (3) Manufacturing  --  defective products on assembly line. (4) Healthcare  --  unusual vital signs. <b>Methods:</b> Isolation Forest, DBSCAN, autoencoders, statistical methods (z-score). <b>Key challenge:</b> Anomalies are rare, so you have very few examples to learn from.",
         "t": "Mention the rarity challenge  --  it connects to the imbalanced classes problem."},
        {"q": "When would you use PCA (dimensionality reduction)?", "d": "Medium", "c": ["Google", "Meta", "Netflix"],
         "a": "<b>PCA</b> reduces the number of features while keeping most of the information. <b>Use when:</b> (1) Too many features (curse of dimensionality). (2) Features are highly correlated (redundant). (3) You need to visualize high-dimensional data in 2D/3D. (4) Speed up model training. <b>How it works:</b> Finds the directions (principal components) that capture the most variance. <b>Tradeoff:</b> You lose interpretability  --  PC1 is a mix of original features, not a single meaningful variable.",
         "t": "Mention the interpretability tradeoff  --  PCA components are hard to explain to stakeholders."},
    ])


# ═══════════════════════════════════════════════════
# KEY TERMS: BIAS & VARIANCE
# ═══════════════════════════════════════════════════
elif concept == "📖 Bias & Variance":
    st.markdown('<div class="section-header">📖 Bias & Variance  --  The Two Forces That Make or Break Your Model</div>', unsafe_allow_html=True)

    st.markdown("""<div class="story-box">
    Every prediction your model makes can be wrong in two fundamentally different ways.
    Understanding these two ways  --  <b>bias</b> and <b>variance</b>  --  is one of the most important
    ideas in all of data science. Let's build the intuition with a simple story.
    </div>""", unsafe_allow_html=True)

    # ── BIAS ──
    st.markdown("---")
    st.markdown("## 🎯 Bias  --  Consistently Wrong in the Same Direction")

    st.markdown("""<div class="story-box">
    Imagine you have a <b>bathroom scale</b> that always reads <b>3 kg too heavy</b>.
    <br><br>
    You step on it Monday: <b>73 kg</b> (real: 70). Tuesday: <b>73.1 kg</b> (real: 70.1). Wednesday: <b>72.9 kg</b> (real: 69.9).
    <br><br>
    The readings are very <b>consistent</b> with each other (low variance)  --  but they're all <b>wrong
    in the same direction</b> (high bias). The scale isn't random  --  it has a <b>systematic lean</b>.
    <br><br>
    That's <b>bias</b>: a consistent, repeatable error that pushes your results away from the truth,
    always in the same direction. It's not random bad luck  --  it's a built-in flaw.
    </div>""", unsafe_allow_html=True)

    # Interactive bias demo: the biased scale
    st.markdown("### 🎮 Try It: The Biased Scale")
    st.caption("Your real weight is 70 kg. Adjust the bias and see how the scale lies to you.")

    scale_bias = st.slider("Scale bias (kg):", -5.0, 5.0, 3.0, 0.5, key="scale_bias",
                           help="Positive = reads too heavy. Negative = reads too light. Zero = perfect scale.")

    np.random.seed(42)
    real_weight = 70.0
    n_days = 20
    noise = np.random.normal(0, 0.3, n_days)  # small random variation
    readings = real_weight + scale_bias + noise

    fig_bias = go.Figure()
    fig_bias.add_trace(go.Scatter(
        x=list(range(1, n_days + 1)), y=readings, mode='lines+markers',
        marker=dict(color='#f45d6d' if abs(scale_bias) > 1 else '#22d3a7', size=7),
        line=dict(width=2), name='Scale readings',
    ))
    fig_bias.add_hline(y=real_weight, line_dash="dash", line_color="#7c6aff",
                       annotation_text=f"Real weight: {real_weight} kg")
    fig_bias.add_hline(y=np.mean(readings), line_dash="dot", line_color="#f5b731",
                       annotation_text=f"Avg reading: {np.mean(readings):.1f} kg")
    fig_bias.update_layout(height=350, title="Daily Scale Readings",
                           xaxis_title="Day", yaxis_title="Weight (kg)", **DARK_LAYOUT)
    st.plotly_chart(fig_bias, use_container_width=True, config={"displayModeBar": False})

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("🎯 Real Weight", f"{real_weight} kg")
    mc2.metric("📏 Avg Reading", f"{np.mean(readings):.1f} kg",
               f"{np.mean(readings) - real_weight:+.1f} kg off")
    mc3.metric("⚠️ Bias", f"{scale_bias:+.1f} kg",
               "No bias!" if abs(scale_bias) < 0.5 else "Biased!" )

    if abs(scale_bias) < 0.5:
        st.markdown("""<div class="analogy-box">
        ✅ The scale is <b>unbiased</b>  --  readings scatter around the true weight. Some days a bit high,
        some days a bit low, but on average it's correct. This is what we want from our models.
        </div>""", unsafe_allow_html=True)
    else:
        direction = "too heavy" if scale_bias > 0 else "too light"
        st.markdown(f"""<div class="impact-box">
        🚨 The scale is <b>biased by {abs(scale_bias):.1f} kg {direction}</b>. Every single reading is off
        in the same direction. No matter how many times you weigh yourself, you'll never get the right answer.
        <b>More data doesn't fix bias</b>  --  it just gives you more wrong answers.
        </div>""", unsafe_allow_html=True)

    # Types of bias in data science
    st.markdown("### 📚 Types of Bias You'll Encounter")

    with st.expander("📊 **Selection Bias**  --  Your data doesn't represent reality", expanded=True):
        st.markdown("""<div class="story-box">
        <b>What it is:</b> Your dataset is not a fair sample of the real world. Certain groups are
        over-represented or under-represented.
        <br><br>
        <b>Example  --  Survivor Bias:</b> During World War II, the military studied bullet holes in planes
        that <b>returned</b> from battle. They wanted to add armor where the holes were. A statistician
        named Abraham Wald said: <b>"No  --  add armor where the holes AREN'T."</b>
        <br><br>
        Why? Because the planes with holes in those spots <b>didn't come back</b>. They were studying
        survivors, not the full picture. The missing planes (the ones that crashed) had holes in
        the critical spots  --  engines and cockpit.
        <br><br>
        <b>In data science:</b> If you train a model on customers who stayed, you're missing the ones
        who already left. Your model learns what "staying" looks like, but not what "about to leave" looks like.
        </div>""", unsafe_allow_html=True)

    with st.expander("🤖 **Algorithmic Bias**  --  Your model learned unfair patterns"):
        st.markdown("""<div class="story-box">
        <b>What it is:</b> The model picks up biases that exist in the historical data and amplifies them.
        <br><br>
        <b>Example:</b> Amazon built a hiring AI trained on 10 years of resumes. Since the tech industry
        historically hired mostly men, the AI learned that male-associated words (like "captain" from
        sports teams) were positive signals. It <b>penalized resumes that mentioned "women's"</b>
        (like "women's chess club"). Amazon scrapped the tool.
        <br><br>
        <b>The lesson:</b> The AI wasn't "sexist"  --  it faithfully learned the patterns in biased data.
        <b>Biased data in -> biased predictions out.</b> The model is a mirror of its training data.
        </div>""", unsafe_allow_html=True)

    with st.expander("🧠 **Confirmation Bias**  --  You see what you want to see"):
        st.markdown("""<div class="story-box">
        <b>What it is:</b> You have a hypothesis, and you unconsciously look for data that supports it
        while ignoring data that contradicts it.
        <br><br>
        <b>Example:</b> A marketing team believes their new ad campaign is working. They find that
        sales went up 5% this month. "See? The campaign works!" But they ignore that sales went up
        8% in the same month last year (seasonal effect), and that a competitor went out of business
        (driving customers their way). The campaign might have had zero effect.
        <br><br>
        <b>The fix:</b> Always try to <b>disprove</b> your hypothesis, not prove it. Look for evidence
        that you're wrong. If you can't find any, then maybe you're right.
        </div>""", unsafe_allow_html=True)

    # ── VARIANCE ──
    st.markdown("---")
    st.markdown("## 🎲 Variance  --  Wildly Different Every Time")

    st.markdown("""<div class="story-box">
    Now imagine a <b>different scale</b>. This one has no bias  --  on average, it's correct.
    But it's <b>wildly inconsistent</b>:
    <br><br>
    Monday: <b>65 kg</b>. Tuesday: <b>76 kg</b>. Wednesday: <b>68 kg</b>. Thursday: <b>74 kg</b>.
    <br><br>
    The average of all readings might be close to 70 kg (your real weight), but any <b>single reading</b>
    is unreliable. You can't trust it on any given day.
    <br><br>
    That's <b>high variance</b>: the results are scattered all over the place. The model isn't
    consistently wrong  --  it's <b>unpredictably wrong</b>. Sometimes too high, sometimes too low,
    with no pattern.
    </div>""", unsafe_allow_html=True)

    # Interactive variance demo
    st.markdown("### 🎮 Try It: The Shaky Scale")
    st.caption("Adjust the variance (shakiness) and see how unreliable the readings become.")

    scale_var = st.slider("Scale shakiness (variance):", 0.1, 8.0, 0.5, 0.1, key="scale_var",
                          help="Low = consistent readings. High = readings jump around wildly.")

    np.random.seed(42)
    noise_var = np.random.normal(0, scale_var, n_days)
    readings_var = real_weight + noise_var  # no bias, just variance

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=list(range(1, n_days + 1)), y=readings_var, mode='lines+markers',
        marker=dict(color='#f5b731' if scale_var > 3 else '#22d3a7', size=7),
        line=dict(width=2), name='Scale readings',
    ))
    fig_var.add_hline(y=real_weight, line_dash="dash", line_color="#7c6aff",
                      annotation_text=f"Real weight: {real_weight} kg")
    # Show the spread
    fig_var.add_hrect(y0=real_weight - scale_var, y1=real_weight + scale_var,
                      fillcolor="rgba(124,106,255,0.08)", line_width=0,
                      annotation_text=f"±{scale_var:.1f} kg spread")
    fig_var.update_layout(height=350, title="Daily Readings (No Bias, Just Variance)",
                          xaxis_title="Day", yaxis_title="Weight (kg)", **DARK_LAYOUT)
    st.plotly_chart(fig_var, use_container_width=True, config={"displayModeBar": False})

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("🎯 Real Weight", f"{real_weight} kg")
    mc2.metric("📏 Avg Reading", f"{np.mean(readings_var):.1f} kg",
               f"{np.mean(readings_var) - real_weight:+.1f} kg off",
               help="With enough readings, the average is close to truth (low bias)")
    mc3.metric("🎲 Spread (Std Dev)", f"±{np.std(readings_var):.1f} kg",
               help="How much individual readings jump around")

    if scale_var < 1.5:
        st.markdown("""<div class="analogy-box">
        ✅ <b>Low variance</b>  --  readings are tightly clustered around the truth. Each individual reading
        is trustworthy. This is a reliable scale (and a reliable model).
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="impact-box">
        ⚠️ <b>High variance</b>  --  readings are scattered ±{scale_var:.1f} kg. The average might be okay,
        but any <b>single reading is unreliable</b>. You'd need to weigh yourself many times and average
        to get a useful number. In ML, this means the model gives very different predictions for
        similar inputs  --  it's <b>unstable and overfitting</b>.
        </div>""", unsafe_allow_html=True)

    # ── THE TRADEOFF ──
    st.markdown("---")
    st.markdown("## ⚖️ The Bias-Variance Tradeoff  --  The Central Tension")

    st.markdown("""<div class="story-box">
    Here's the cruel truth: <b>reducing bias often increases variance, and vice versa.</b>
    <br><br>
    Think of it as a <b>dartboard</b>:
    </div>""", unsafe_allow_html=True)

    # Dartboard visualization
    st.markdown("### 🎯 The Dartboard Analogy")

    dart_scenario = st.radio("Pick a scenario:", [
        "🎯 Low Bias, Low Variance (the goal)",
        "↗️ High Bias, Low Variance (consistently wrong)",
        "💥 Low Bias, High Variance (scattered but centered)",
        "🌪️ High Bias, High Variance (the worst)",
    ], key="dart_scenario", horizontal=True)

    np.random.seed(42)
    n_darts = 20

    if "Low Bias, Low Variance" in dart_scenario:
        dx, dy = np.random.normal(0, 0.3, n_darts), np.random.normal(0, 0.3, n_darts)
        msg = ("All darts land <b>close together</b> and <b>near the bullseye</b>. "
               "This is the ideal model  --  accurate and consistent. Hard to achieve, but that's the goal.")
        msg_color = "#22d3a7"
    elif "High Bias, Low Variance" in dart_scenario:
        dx, dy = np.random.normal(2, 0.3, n_darts), np.random.normal(2, 0.3, n_darts)
        msg = ("All darts land <b>close together</b> but <b>far from the bullseye</b>. "
               "The model is consistent but systematically wrong. Like an underfitting model  --  "
               "it's too simple to capture the real pattern, so it's always off in the same way.")
        msg_color = "#f5b731"
    elif "Low Bias, High Variance" in dart_scenario:
        dx, dy = np.random.normal(0, 1.5, n_darts), np.random.normal(0, 1.5, n_darts)
        msg = ("Darts are <b>scattered everywhere</b> but <b>centered on the bullseye</b>. "
               "On average the model is right, but any single prediction is unreliable. "
               "Like an overfitting model  --  it memorized the training data and is unstable on new data.")
        msg_color = "#f5b731"
    else:
        dx, dy = np.random.normal(2, 1.5, n_darts), np.random.normal(-2, 1.5, n_darts)
        msg = ("Darts are <b>scattered everywhere</b> AND <b>far from the bullseye</b>. "
               "The worst case  --  the model is both wrong and unpredictable. Time to start over.")
        msg_color = "#f45d6d"

    fig_dart = go.Figure()
    # Target circles
    for r in [3, 2, 1, 0.5]:
        theta_c = np.linspace(0, 2 * np.pi, 100)
        fig_dart.add_trace(go.Scatter(
            x=r * np.cos(theta_c), y=r * np.sin(theta_c), mode='lines',
            line=dict(color='#2d3148', width=1), showlegend=False, hoverinfo='skip',
        ))
    # Bullseye
    fig_dart.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                                   marker=dict(color='#f45d6d', size=10, symbol='x'),
                                   name='Bullseye (truth)', showlegend=True))
    # Darts
    fig_dart.add_trace(go.Scatter(x=dx, y=dy, mode='markers',
                                   marker=dict(color='#22d3a7', size=8, opacity=0.8),
                                   name='Predictions (darts)'))
    # Average of darts
    fig_dart.add_trace(go.Scatter(x=[np.mean(dx)], y=[np.mean(dy)], mode='markers',
                                   marker=dict(color='#f5b731', size=14, symbol='diamond'),
                                   name='Average prediction'))
    fig_dart.update_layout(
        height=420, title="🎯 Where Do Your Predictions Land?",
        xaxis=dict(range=[-4, 4], scaleanchor="y", gridcolor='#2d3148',
                   zeroline=True, zerolinecolor='#2d3148', tickfont=dict(color='#8892b0')),
        yaxis=dict(range=[-4, 4], gridcolor='#2d3148',
                   zeroline=True, zerolinecolor='#2d3148', tickfont=dict(color='#8892b0')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'), legend=dict(font=dict(color='#8892b0')),
    )
    st.plotly_chart(fig_dart, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="analogy-box" style="border-color:{msg_color}">
    💡 {msg}
    </div>""", unsafe_allow_html=True)

    # The tradeoff explained
    st.markdown("### 🤔 Why Can't We Have Both Low Bias AND Low Variance?")

    st.markdown("""<div class="story-box">
    Think of it like a <b>student preparing for an exam</b>:
    <br><br>
    📉 <b>Too simple (high bias, low variance):</b> The student only learns the chapter summaries.
    They'll give the same mediocre answer every time  --  consistent, but always missing the details.
    They <b>underfit</b> the material.
    <br><br>
    📈 <b>Too complex (low bias, high variance):</b> The student memorizes every word of the textbook,
    including typos and footnotes. On the practice exam (training data), they score 100%.
    But on the real exam with slightly different questions, they panic  --  they memorized specifics,
    not concepts. They <b>overfit</b> the material.
    <br><br>
    ✅ <b>Just right:</b> The student understands the core concepts and can apply them to new questions.
    They won't score 100% on practice exams, but they'll do well on any exam. That's the <b>sweet spot</b>.
    </div>""", unsafe_allow_html=True)

    # Interactive: see the tradeoff
    st.markdown("### 🎮 Try It: The Bias-Variance Slider")
    st.caption("Drag the slider from simple to complex and watch bias and variance change.")

    complexity = st.slider("Model complexity:", 1, 10, 3, key="bv_complexity",
                           help="1 = very simple model, 10 = very complex model")

    bias_val = max(0.5, 5.0 - complexity * 0.5)
    var_val = max(0.3, complexity * 0.5 - 0.5)
    total_error = bias_val ** 2 + var_val ** 2

    fig_bv = go.Figure()
    complexities = list(range(1, 11))
    biases = [max(0.5, 5.0 - c * 0.5) for c in complexities]
    variances = [max(0.3, c * 0.5 - 0.5) for c in complexities]
    totals = [b**2 + v**2 for b, v in zip(biases, variances)]

    fig_bv.add_trace(go.Scatter(x=complexities, y=biases, mode='lines+markers',
                                 name='Bias', line=dict(color='#7c6aff', width=3)))
    fig_bv.add_trace(go.Scatter(x=complexities, y=variances, mode='lines+markers',
                                 name='Variance', line=dict(color='#f45d6d', width=3)))
    fig_bv.add_trace(go.Scatter(x=complexities, y=totals, mode='lines+markers',
                                 name='Total Error', line=dict(color='#f5b731', width=3, dash='dash')))
    # Current position
    fig_bv.add_trace(go.Scatter(x=[complexity], y=[total_error], mode='markers',
                                 marker=dict(color='#22d3a7', size=16, symbol='star'),
                                 name='Your model'))
    # Sweet spot
    best_c = complexities[totals.index(min(totals))]
    fig_bv.add_vline(x=best_c, line_dash="dot", line_color="#22d3a7",
                     annotation_text=f"Sweet spot: {best_c}")
    fig_bv.update_layout(height=400, title="The Bias-Variance Tradeoff",
                         xaxis_title="Model Complexity ->", yaxis_title="Error ->", **DARK_LAYOUT)
    st.plotly_chart(fig_bv, use_container_width=True, config={"displayModeBar": False})

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🟣 Bias", f"{bias_val:.1f}", help="How far off the model is on average")
    mc2.metric("🔴 Variance", f"{var_val:.1f}", help="How much predictions jump around")
    mc3.metric("🟡 Total Error", f"{total_error:.1f}", help="Bias² + Variance²")
    zone = "Underfitting" if complexity <= 3 else "Overfitting" if complexity >= 7 else "Sweet Spot ✅"
    mc4.metric("📍 Zone", zone)

    if complexity <= 3:
        st.markdown("""<div class="impact-box">
        📉 <b>Underfitting zone:</b> Your model is too simple. It has high bias (consistently wrong)
        but low variance (at least it's consistent). It's like predicting everyone's salary is $50K  -- 
        you'll be wrong, but you'll be wrong the same way every time. <b>Solution: make the model more complex.</b>
        </div>""", unsafe_allow_html=True)
    elif complexity >= 7:
        st.markdown("""<div class="impact-box">
        📈 <b>Overfitting zone:</b> Your model is too complex. It has low bias (on average it's close)
        but high variance (individual predictions are unreliable). It memorized the training data
        including the noise. <b>Solution: simplify the model, get more data, or use regularization.</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="analogy-box">
        ✅ <b>Sweet spot:</b> Your model balances bias and variance. It's complex enough to capture
        real patterns but simple enough to not memorize noise. This is where the total error is minimized.
        </div>""", unsafe_allow_html=True)

    # Summary
    st.markdown("### 📋 Quick Reference")


    comparison = pd.DataFrame({
        "": ["What is it?", "Analogy", "Caused by", "Symptom", "Fix"],
        "Bias 🟣": [
            "Consistently wrong in the same direction",
            "A scale that always reads 3 kg too heavy",
            "Model too simple, wrong assumptions, biased data",
            "Bad on training AND test data",
            "More features, more complex model, better data",
        ],
        "Variance 🔴": [
            "Predictions jump around unpredictably",
            "A shaky scale giving different readings each time",
            "Model too complex, memorizing noise",
            "Great on training, bad on test data",
            "Simpler model, more training data, regularization",
        ],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("""<div class="key-point">
    <b>🎯 Key Takeaway:</b> Bias and variance are the <b>yin and yang</b> of ML.
    You can't eliminate both  --  but you can find the <b>sweet spot</b> where their combined error is smallest.
    </div>""", unsafe_allow_html=True)

    iq([
        {"q": "Explain the bias-variance tradeoff to a non-technical person.", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "<b>Bias</b> = consistently wrong in the same direction (scale always reads 3 kg too heavy). <b>Variance</b> = wildly different answers each time (shaky scale). <b>Tradeoff:</b> Simple models = high bias, low variance. Complex models = low bias, high variance. The goal is the <b>sweet spot</b> in between.",
         "t": "Use the scale analogy  --  clearest way to explain it."},
        {"q": "Your model has high bias. What do you do?", "d": "Medium", "c": ["Google", "Amazon", "Netflix"],
         "a": "High bias = underfitting. Model too simple. <b>Fixes:</b> (1) Add more features. (2) More complex model. (3) Reduce regularization. (4) Add polynomial/interaction features. <b>Diagnosis:</b> Both training and test error are high.",
         "t": "Start with 'high bias means underfitting' then list fixes."},
        {"q": "Your model has high variance. What do you do?", "d": "Medium", "c": ["Google", "Meta", "Apple"],
         "a": "High variance = overfitting. Model memorized noise. <b>Fixes:</b> (1) More training data. (2) Simplify model. (3) Add regularization (L1/L2). (4) Ensemble methods (bagging). (5) Cross-validation. (6) Early stopping. <b>Diagnosis:</b> Training error low, test error high.",
         "t": "Mention Random Forest reduces variance through bagging."},
        {"q": "What is regularization and why does it help?", "d": "Hard", "c": ["Google", "Meta", "Netflix"],
         "a": "Adds a <b>penalty for complexity</b> to the loss function. <b>L1 (Lasso):</b> Pushes coefficients to zero -> feature selection. <b>L2 (Ridge):</b> Shrinks coefficients -> smoother model. <b>Elastic Net:</b> L1 + L2. <b>Analogy:</b> Penalizing a student for writing too much  --  forces focus on what matters.",
         "t": "Mention L1 does feature selection  --  common follow-up."},
        {"q": "How does ensemble learning reduce variance?", "d": "Hard", "c": ["Google", "Amazon"],
         "a": "<b>Bagging:</b> Train many models on random data subsets, average predictions. Individual models overfit differently, but averaging cancels noise. <b>Random Forest</b> = bagging + random features. <b>Analogy:</b> 100 people guess jellybeans  --  individual guesses are wild, but the average is surprisingly accurate (wisdom of crowds).",
         "t": "The jellybean analogy is memorable."},
    ])
