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
.green-box {
    background: linear-gradient(135deg, #1a2a1f, #1f3528);
    border: 1px solid #2a5a3a; border-radius: 14px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; font-size: 0.9rem; color: #c8d8c0; line-height: 1.7;
}
.green-box b { color: #d0f0e0; }
.red-box {
    background: linear-gradient(135deg, #2a1a1e, #351f25);
    border: 1px solid #5a2a3a; border-radius: 14px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; font-size: 0.9rem; color: #d8a8b8; line-height: 1.7;
}
.red-box b { color: #f0c8d8; }
.key-box {
    background: #252840; border-left: 4px solid #7c6aff;
    border-radius: 0 10px 10px 0; padding: 0.8rem 1.2rem;
    margin: 0.6rem 0; font-size: 0.88rem; color: #c8cfe0; line-height: 1.7;
}
.key-box b { color: #e2e8f0; }
.iq-card {
    background: #181a27; border: 1px solid #2d3148; border-radius: 14px;
    padding: 1.1rem 1.4rem; margin-bottom: 0.5rem; border-left: 4px solid;
}
.iq-title { font-size: 0.93rem; font-weight: 700; color: #e2e8f0; line-height: 1.5; }
.iq-meta { font-size: 0.7rem; margin-top: 5px; }
.iq-tag { display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 0.67rem; font-weight: 600; }
.iq-answer {
    background: #1a2a1f; border: 1px solid #2a5a3a; border-radius: 12px;
    padding: 1rem 1.3rem; margin-top: 0.3rem; color: #c8d8c0; font-size: 0.88rem; line-height: 1.8;
}
.iq-answer b { color: #d0f0e0; }
.iq-tip {
    background: #252840; border-left: 4px solid #f5b731; border-radius: 0 10px 10px 0;
    padding: 0.6rem 1rem; margin-top: 0.3rem; font-size: 0.84rem; color: #c8cfe0; line-height: 1.6;
}
.iq-tip b { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

DL = dict(
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
    st.caption("Real FAANG-level questions on this topic.")
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

with st.sidebar:
    st.markdown("## 📊 Phase 1: Statistics")
    st.caption("3–4 Weeks · The foundation for everything")
    st.divider()
    module = st.radio("Module:", [
        "🏠 Overview",
        "🧩 M1: Descriptive Statistics",
        "🎲 M2: Probability",
        "📈 M3: Distributions",
        "🧪 M4: Inferential Statistics",
        "🔗 M5: Correlation",
        "📉 M6: Regression Intuition",
    ], label_visibility="collapsed")


# ═══════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════
if module == "🏠 Overview":
    st.markdown("# 📊 Phase 1: Statistics Foundation")
    st.caption("3–4 Weeks · Understand data, uncertainty, and decision-making — this is the base for everything.")

    st.markdown("""<div class="story-box">
    <b>Why start with statistics?</b> Every machine learning model, every A/B test, every data-driven decision
    is built on statistics. Skip this, and everything later will feel like magic you can't control.
    Nail this, and ML becomes intuitive.
    </div>""", unsafe_allow_html=True)

    modules = [
        ("🧩", "Descriptive Statistics", "Week 1", "#7c6aff", "Mean, median, variance, std dev, quartiles, outliers — summarizing data in numbers."),
        ("🎲", "Probability", "Week 1–2", "#22d3a7", "Probability basics, conditional probability, Bayes' theorem — reasoning under uncertainty."),
        ("📈", "Distributions", "Week 2", "#f5b731", "Normal, skewed, binomial, Poisson — the shapes data takes and why they matter."),
        ("🧪", "Inferential Statistics", "Week 3", "#f45d6d", "Sampling, confidence intervals, hypothesis testing, p-values — is this result real or random?"),
        ("🔗", "Correlation", "Week 3", "#e879a8", "Correlation, covariance, causation trap — which factors move together?"),
        ("📉", "Regression Intuition", "Week 4", "#5eaeff", "Linear relationships, residuals, overfitting — predicting output from input."),
    ]

    for icon, title, week, color, desc in modules:
        st.markdown(f"""<div class="story-box" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span>
        <span class="iq-tag" style="background:{color}22;color:{color}">{week}</span>
        <b style="margin-left:6px">{title}</b>
        <br><span style="color:#8892b0">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-box">
    <b>✅ After Phase 1 you will:</b><br>
    • Understand any dataset conceptually<br>
    • Interpret trends and variability<br>
    • Reason about uncertainty<br>
    • Understand ML concepts easily later<br>
    • Be ready for statistics interview questions at any company
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# MODULE 1: DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════
elif module == "🧩 M1: Descriptive Statistics":
    st.markdown("# 🧩 Module 1: Descriptive Statistics")
    st.caption("Week 1 · Summarize data in a few meaningful numbers.")

    st.markdown("""<div class="story-box">
    You have 10,000 customer records. You can't look at each one. <b>Descriptive statistics</b> lets you
    summarize the entire dataset in 5–6 numbers that tell the story: What's typical? How spread out?
    Anything unusual?
    </div>""", unsafe_allow_html=True)

    # ── Mean, Median, Mode ──
    st.markdown("### 📖 Mean, Median, Mode")
    st.markdown("""<div class="story-box">
    <b>Mean (Average):</b> Add all values, divide by count. The "balance point." Every value pulls on it.
    <br><b>Median (Middle):</b> Sort values, pick the center one. Ignores extremes — robust to outliers.
    <br><b>Mode (Most frequent):</b> The value that appears most often. Best for categorical data.
    <br><br>
    <b>When mean ≠ median:</b> Your data is skewed. If mean > median → right-skewed (a few large values pulling the mean up, like income data).
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Enter data and see all three")
    user_data = st.text_input("Enter numbers (comma-separated):", "25, 30, 35, 40, 45, 50, 200", key="m1_data")
    try:
        vals = np.array([float(x.strip()) for x in user_data.split(",") if x.strip()])
        sorted_v = np.sort(vals)
        mc = st.columns(4)
        mc[0].metric("🔵 Mean", f"{vals.mean():.1f}", help="Sum ÷ count")
        mc[1].metric("🟣 Median", f"{np.median(vals):.1f}", help="Middle value when sorted")
        from scipy import stats as sp_stats
        mode_result = sp_stats.mode(vals, keepdims=True)
        mc[2].metric("🟠 Mode", f"{mode_result.mode[0]:.1f}", help="Most frequent value")
        mc[3].metric("Gap (Mean-Median)", f"{vals.mean() - np.median(vals):.1f}",
                     help="Large gap = skewed data")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=15, marker_color="#7c6aff", opacity=0.7))
        fig.add_vline(x=vals.mean(), line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: {vals.mean():.1f}")
        fig.add_vline(x=np.median(vals), line_dash="dash", line_color="#22d3a7", annotation_text=f"Median: {np.median(vals):.1f}")
        fig.update_layout(height=280, title="Your Data Distribution", xaxis_title="Value", yaxis_title="Count", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if abs(vals.mean() - np.median(vals)) > vals.std() * 0.3:
            st.markdown("""<div class="red-box">⚠️ <b>Mean and median are far apart</b> — your data is skewed or has outliers. The median better represents the "typical" value here.</div>""", unsafe_allow_html=True)
    except:
        st.error("Enter valid numbers separated by commas.")

    # ── Variance & Std Dev ──
    st.markdown("### 📖 Variance & Standard Deviation")
    st.markdown("""<div class="story-box">
    <b>Variance</b> = average of squared distances from the mean. Tells you how spread out data is, but in squared units (hard to interpret).
    <br><b>Standard Deviation (Std Dev)</b> = square root of variance. Same units as your data — much easier to understand.
    <br><br><b>Intuition:</b> Std dev answers: <b>"How far does a typical value sit from the average?"</b>
    <br>• Std dev = 5 on exam scores → most students scored within 5 points of the average
    <br>• Std dev = 25 → scores were all over the place
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Two teams, same average, different spread")
    c1, c2 = st.columns(2)
    with c1:
        sd_a = st.slider("Team A — Std Dev (how spread out):", 1, 25, 5, key="m1_sda")
    with c2:
        sd_b = st.slider("Team B — Std Dev (how spread out):", 1, 25, 18, key="m1_sdb")

    np.random.seed(42)
    team_a = np.random.normal(75, sd_a, 300).clip(0, 100)
    team_b = np.random.normal(75, sd_b, 300).clip(0, 100)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=team_a, name=f"Team A (std={team_a.std():.1f})", marker_color="#7c6aff", opacity=0.5, nbinsx=25))
    fig2.add_trace(go.Histogram(x=team_b, name=f"Team B (std={team_b.std():.1f})", marker_color="#22d3a7", opacity=0.5, nbinsx=25))
    fig2.update_layout(barmode="overlay", height=300, title="Same Average (75), Different Spread", xaxis_title="Score", yaxis_title="Count", **DL)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### 📖 The 68-95-99.7 Rule (Empirical Rule)")
    st.caption("For bell-shaped (normal) data, this rule tells you exactly how data spreads around the mean.")

    st.markdown("""<div class="story-box">
    If your data follows a bell curve:
    <br>• <b>68%</b> of values fall within <b>±1 std dev</b> of the mean
    <br>• <b>95%</b> fall within <b>±2 std devs</b>
    <br>• <b>99.7%</b> fall within <b>±3 std devs</b>
    <br><br>Anything beyond 3 std devs is <b>extremely rare</b> (0.3% chance) — that's where outliers live.
    <br><br><b>Example:</b> If exam scores have mean=75 and std dev=10:
    <br>• 68% of students scored <b>65–85</b>
    <br>• 95% scored <b>55–95</b>
    <br>• 99.7% scored <b>45–105</b>
    <br>• A score of 110? That's beyond 3σ — either a genius or a grading error.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: See the Rule on a Bell Curve")
    rule_mean = st.slider("Mean:", 40, 100, 75, 5, key="m1_rule_mean")
    rule_std = st.slider("Std Dev:", 2, 20, 10, 1, key="m1_rule_std")

    x_rule = np.linspace(rule_mean - 4*rule_std, rule_mean + 4*rule_std, 500)
    from scipy.stats import norm
    y_rule = norm.pdf(x_rule, rule_mean, rule_std)

    fig_rule = go.Figure()
    fig_rule.add_trace(go.Scatter(x=x_rule, y=y_rule, mode='lines', line=dict(color='#7c6aff', width=2), name='Distribution'))
    # 1σ band (68%)
    fig_rule.add_vrect(x0=rule_mean-rule_std, x1=rule_mean+rule_std, fillcolor="rgba(34,211,167,0.15)", line_width=0)
    # 2σ band (95%)
    fig_rule.add_vrect(x0=rule_mean-2*rule_std, x1=rule_mean+2*rule_std, fillcolor="rgba(124,106,255,0.08)", line_width=0)
    # Labels
    fig_rule.add_annotation(x=rule_mean, y=max(y_rule)*0.5, text="<b>68%</b>", showarrow=False, font=dict(color="#22d3a7", size=14))
    fig_rule.add_annotation(x=rule_mean-1.5*rule_std, y=max(y_rule)*0.15, text="95%", showarrow=False, font=dict(color="#7c6aff", size=11))
    fig_rule.add_annotation(x=rule_mean+1.5*rule_std, y=max(y_rule)*0.15, text="95%", showarrow=False, font=dict(color="#7c6aff", size=11))
    # Boundary lines
    for mult, label in [(1, "±1σ"), (2, "±2σ"), (3, "±3σ")]:
        fig_rule.add_vline(x=rule_mean+mult*rule_std, line_dash="dot", line_color="#2d3148")
        fig_rule.add_vline(x=rule_mean-mult*rule_std, line_dash="dot", line_color="#2d3148")

    fig_rule.update_layout(height=300, title=f"68-95-99.7 Rule: Mean={rule_mean}, Std Dev={rule_std}", **DL)
    st.plotly_chart(fig_rule, use_container_width=True, config={"displayModeBar": False})

    mc = st.columns(3)
    mc[0].metric("68% range", f"{rule_mean-rule_std} – {rule_mean+rule_std}")
    mc[1].metric("95% range", f"{rule_mean-2*rule_std} – {rule_mean+2*rule_std}")
    mc[2].metric("99.7% range", f"{rule_mean-3*rule_std} – {rule_mean+3*rule_std}")

    # ── Quartiles & Percentiles ──
    st.markdown("### 📖 Quartiles, Percentiles & IQR")
    st.markdown("""<div class="story-box">
    <b>Percentile:</b> "What percentage of data falls below this value?" If you're in the 90th percentile on an exam, 90% of students scored lower than you.
    <br><br><b>Quartiles</b> split data into 4 equal parts:
    <br>• <b>Q1 (25th %ile):</b> 25% of data is below this
    <br>• <b>Q2 (50th %ile):</b> The median — 50% below
    <br>• <b>Q3 (75th %ile):</b> 75% of data is below this
    <br><br><b>IQR = Q3 - Q1:</b> The range of the middle 50%. Used to detect outliers: anything below Q1 - 1.5×IQR or above Q3 + 1.5×IQR is suspicious.
    </div>""", unsafe_allow_html=True)

    # ── Outliers ──
    st.markdown("### 📖 Outliers")
    st.markdown("""<div class="story-box">
    An <b>outlier</b> is a data point far from the rest. It could be an error (typo), a rare event (fraud), or genuinely unusual data (a billionaire in income data).
    <br><br><b>Key question:</b> Don't ask "should I remove it?" Ask <b>"why is it there?"</b>
    <br>• Error → remove it
    <br>• Fraud → keep it (it's the signal!)
    <br>• Rare but real → investigate, then decide
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Add an outlier and watch the damage")
    outlier_v = st.slider("Add an outlier with value:", 75, 500, 75, 25, key="m1_out")
    np.random.seed(42)
    base = np.random.normal(75, 10, 50)
    data_out = np.append(base, outlier_v) if outlier_v > 100 else base

    mc = st.columns(4)
    mc[0].metric("Mean", f"{data_out.mean():.1f}", f"{data_out.mean()-base.mean():+.1f}" if outlier_v > 100 else None)
    mc[1].metric("Median", f"{np.median(data_out):.1f}", f"{np.median(data_out)-np.median(base):+.1f}" if outlier_v > 100 else None)
    mc[2].metric("Std Dev", f"{data_out.std():.1f}", f"{data_out.std()-base.std():+.1f}" if outlier_v > 100 else None)
    mc[3].metric("Mean Distortion", f"{((data_out.mean()/base.mean())-1)*100:+.0f}%" if outlier_v > 100 else "0%")

    fig3 = go.Figure()
    colors = ['#f45d6d' if v == outlier_v and outlier_v > 100 else '#22d3a7' for v in data_out]
    fig3.add_trace(go.Scatter(x=list(range(len(data_out))), y=data_out, mode='markers', marker=dict(color=colors, size=[12 if c == '#f45d6d' else 6 for c in colors])))
    fig3.add_hline(y=data_out.mean(), line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: {data_out.mean():.1f}")
    fig3.add_hline(y=np.median(data_out), line_dash="dash", line_color="#7c6aff", annotation_text=f"Median: {np.median(data_out):.1f}")
    fig3.update_layout(height=300, title="Outlier Impact: Mean gets pulled, Median stays", **DL)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Domain example ──
    st.markdown("""<div class="green-box">
    👉 <b>Your domain example (Telecom):</b><br>
    • <b>Mean latency</b> = 45ms looks fine, but <b>std dev = 80ms</b> means some users experience 200ms+ spikes<br>
    • <b>KPI variability across cells:</b> Low std dev = consistent network. High std dev = some cells are great, others are terrible<br>
    • <b>Outlier detection:</b> A cell tower with 10× the average dropped calls is an outlier — investigate before removing
    </div>""", unsafe_allow_html=True)

    iq([
        {"q": "Explain mean, median, and mode. When would you use each?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Mean:</b> Use when data is symmetric, no outliers. <b>Median:</b> Use when data is skewed or has outliers (income, house prices). <b>Mode:</b> Use for categorical data (most popular product, most common error type). <b>Key insight:</b> If mean >> median, data is right-skewed. Economists report median income, not mean, because billionaires skew the average.",
         "t": "Always mention outlier sensitivity of the mean. Give a concrete example."},
        {"q": "What is standard deviation? Explain it to a 10-year-old.", "d": "Easy", "c": ["Meta", "Apple"],
         "a": "Imagine your class takes a test. The average score is 75. Std dev tells you <b>how far most students scored from 75</b>. If std dev is 5, almost everyone got 70–80 (consistent class). If std dev is 20, scores ranged from 55 to 95 (wild class). It measures <b>how spread out</b> the data is.",
         "t": "Use the exam analogy — it's universally understood."},
        {"q": "What's the difference between variance and standard deviation?", "d": "Easy", "c": ["Google", "General"],
         "a": "Variance = average of squared distances from the mean. Std dev = square root of variance. <b>Why both?</b> Variance is useful in math (it's additive), but std dev is in the <b>same units as your data</b>, making it interpretable. If exam scores are in points, std dev is in points too, but variance is in 'points squared' — meaningless to humans.",
         "t": "Mention that variance is used internally in formulas, std dev is what you report."},
        {"q": "How do you detect outliers? Compare IQR and Z-score methods.", "d": "Medium", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>IQR:</b> Q1 - 1.5×IQR to Q3 + 1.5×IQR. Doesn't assume normal distribution. <b>Z-score:</b> Flag |z| > 2-3. Assumes roughly normal data. <b>Key difference:</b> IQR is robust (outliers don't affect Q1/Q3 much). Z-score uses mean and std, which ARE affected by outliers — so outliers can hide themselves! <b>My preference:</b> Start with box plots (visual), then IQR for formal detection.",
         "t": "Mention that you'd always VISUALIZE first. Box plots catch things formulas miss."},
        {"q": "You're told the average response time is 200ms. Is the system healthy?", "d": "Medium", "c": ["Amazon", "Google", "Netflix"],
         "a": "Not enough information! I'd ask: <b>What's the median?</b> (If median is 50ms but mean is 200ms, a few extreme outliers are pulling the average up.) <b>What's the std dev?</b> (Low std = consistent, high std = some requests are very slow.) <b>What's the 95th/99th percentile?</b> (P99 = 2000ms means 1% of users wait 2 seconds — that's terrible even if the average looks fine.) <b>The average alone is never enough.</b>",
         "t": "This is a CLASSIC system design question. Always ask for percentiles, not just averages."},
        {"q": "What is the 68-95-99.7 rule and when does it apply?", "d": "Medium", "c": ["Google", "Netflix"],
         "a": "For <b>normally distributed</b> data: 68% falls within ±1σ of the mean, 95% within ±2σ, 99.7% within ±3σ. <b>When it applies:</b> Only for bell-shaped (normal) distributions. Does NOT apply to skewed data, bimodal data, or uniform distributions. <b>Practical use:</b> If a server's response time has mean=100ms and std=20ms, then 95% of requests complete in 60–140ms. Anything above 160ms (3σ) is an anomaly worth investigating.",
         "t": "Emphasize it ONLY works for normal distributions. Interviewers test if you know the limitation."},
        {"q": "A dataset has mean=50, median=50, mode=50. What can you infer?", "d": "Easy", "c": ["General"],
         "a": "The data is likely <b>symmetric</b> (bell-shaped). When all three measures of central tendency are equal, there's no skew. But it doesn't guarantee normality — a uniform distribution can also have equal mean/median/mode. You'd need to check the <b>shape</b> (histogram) and <b>kurtosis</b> (how peaked/flat) to confirm.",
         "t": "Don't just say 'it's normal.' Say 'it's symmetric' and mention you'd verify with a histogram."},
    ])


# ═══════════════════════════════════════
# MODULE 2: PROBABILITY
# ═══════════════════════════════════════
elif module == "🎲 M2: Probability":
    st.markdown("# 🎲 Module 2: Probability")
    st.caption("Week 1–2 · Reasoning under uncertainty.")

    st.markdown("""<div class="story-box">
    Probability answers: <b>"How likely is something to happen?"</b> It's a number from 0 (impossible) to 1 (certain).
    Every prediction, every risk assessment, every A/B test is built on probability.
    </div>""", unsafe_allow_html=True)

    # ── Basics ──
    st.markdown("### 📖 Probability Basics")
    st.markdown("""<div class="story-box">
    <b>P(event) = favorable outcomes ÷ total outcomes</b>
    <br><br>Coin flip: P(heads) = 1/2 = 0.5. Die roll: P(six) = 1/6 ≈ 0.167.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 📖 The Three Key Rules (with intuition)")

    st.markdown("""<div class="story-box" style="border-left:4px solid #7c6aff">
    <b style="color:#7c6aff">Rule 1: P(A or B) = P(A) + P(B) - P(A∩B)</b>
    <br><br>
    <b>In English:</b> "What's the chance of A happening OR B happening (or both)?"
    <br><br>
    <b>Why subtract the overlap?</b> Imagine a class of 30 students. 10 play football, 8 play basketball,
    and 3 play both. If you just add 10 + 8 = 18, you've counted those 3 students <b>twice</b>.
    So: P(football or basketball) = 10 + 8 - 3 = 15 out of 30 = 0.5.
    <br><br>
    <b>Telecom:</b> "What's the probability a customer has a billing complaint OR a network complaint?"
    Some customers have both — don't double-count them.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box" style="border-left:4px solid #22d3a7">
    <b style="color:#22d3a7">Rule 2: P(A and B) = P(A) × P(B) — if independent</b>
    <br><br>
    <b>In English:</b> "What's the chance of A AND B both happening?"
    <br><br>
    <b>Only works when independent</b> — meaning A happening doesn't change the probability of B.
    <br><br>
    <b>Example:</b> You flip a coin and roll a die. P(heads AND six) = 1/2 × 1/6 = 1/12.
    The coin doesn't know what the die is doing — they're independent.
    <br><br>
    <b>Counter-example (NOT independent):</b> P(umbrella AND rain). If it's raining, you're much more
    likely to carry an umbrella. These are <b>dependent</b> — you can't just multiply.
    For dependent events: P(A and B) = P(A) × P(B|A).
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box" style="border-left:4px solid #f5b731">
    <b style="color:#f5b731">Rule 3: P(not A) = 1 - P(A)</b>
    <br><br>
    <b>In English:</b> "The chance of something NOT happening = 1 minus the chance it does happen."
    <br><br>
    <b>Why it's useful:</b> Sometimes it's easier to calculate the opposite. "What's the probability
    of getting at least one head in 10 flips?" is hard to calculate directly. But P(zero heads) = (1/2)¹⁰ = 0.001.
    So P(at least one head) = 1 - 0.001 = <b>0.999</b>. Much easier!
    <br><br>
    <b>Telecom:</b> "What's the probability at least one server fails this month?" Calculate P(none fail) first, then subtract from 1.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Coin Flip Simulator")
    n_flips = st.slider("Number of flips:", 10, 50000, 1000, 100, key="m2_flips")
    if st.button("🪙 Flip!", key="m2_flip_btn"):
        np.random.seed(None)
        results = np.random.choice([0, 1], n_flips)
        running = np.cumsum(results) / np.arange(1, n_flips + 1)
        mc = st.columns(2)
        mc[0].metric("P(Heads)", f"{results.mean():.4f}", f"{results.mean()-0.5:+.4f} from theory")
        mc[1].metric("Flips", f"{n_flips:,}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=running, mode='lines', line=dict(color='#7c6aff', width=1.5)))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#22d3a7", annotation_text="Theory: 0.5")
        fig.update_layout(height=280, title="Convergence: More flips → closer to 0.5", xaxis_title="Flip #", yaxis_title="P(Heads)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Conditional Probability ──
    st.markdown("### 📖 Conditional Probability — \"Given that X happened, what are the chances of Y?\"")

    st.markdown("""<div class="story-box">
    Imagine a school with <b>200 students</b>. You know:
    <br>• <b>120 students</b> play sports (60%)
    <br>• <b>80 students</b> get A grades (40%)
    <br>• <b>50 students</b> play sports AND get A grades (25%)
    <br><br>
    Now someone tells you: <i>"I picked a random student, and they play sports."</i>
    <br>What is the probability this student ALSO gets A grades?
    <br><br>
    You are NOT asking about all 200 students anymore. You are asking about the <b>120 who play sports</b>.
    Out of those 120, how many get A grades? <b>50.</b>
    <br><br>
    <div style="text-align:center;font-size:1.2rem;padding:0.6rem;background:rgba(124,106,255,0.08);border-radius:10px;margin:0.5rem 0">
    <b>P(A grades | plays sports) = 50 / 120 = 0.417 (41.7%)</b>
    </div>
    <br>
    Compare this to P(A grades) for ALL students = 80/200 = 40%. Knowing they play sports <b>slightly increased</b>
    the probability of A grades (from 40% to 41.7%). The "given" information <b>narrows the world</b> you are looking at.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box" style="border-left:4px solid #7c6aff">
    <b style="color:#7c6aff">The formula explained step by step:</b>
    <br><br>
    <b>P(A|B) = P(A and B) / P(B)</b>
    <br><br>
    In words: <i>"The probability of A happening GIVEN B happened = (the probability of BOTH happening) divided by (the probability of B happening)."</i>
    <br><br>
    <b>Why divide by P(B)?</b> Because once you know B happened, B becomes your <b>new universe</b>.
    You are no longer looking at all possible outcomes — only the ones where B is true.
    Dividing by P(B) "zooms in" to that smaller world.
    <br><br>
    <b>Our example:</b>
    <br>• P(A and B) = P(sports AND A grades) = 50/200 = 0.25
    <br>• P(B) = P(sports) = 120/200 = 0.60
    <br>• P(A|B) = 0.25 / 0.60 = <b>0.417</b>
    </div>""", unsafe_allow_html=True)

    # Venn diagram visualization
    st.markdown("#### 🎮 See It Visually: The Venn Diagram")
    st.caption("The blue circle = students who play sports. The green circle = students who get A grades. The overlap = both.")

    fig_venn = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    # Circle B (sports) - left
    fig_venn.add_trace(go.Scatter(x=2.5*np.cos(theta)-0.8, y=2.5*np.sin(theta), mode='lines',
        fill='toself', fillcolor='rgba(124,106,255,0.12)', line=dict(color='#7c6aff', width=2), name='Sports (120)'))
    # Circle A (A grades) - right
    fig_venn.add_trace(go.Scatter(x=2.5*np.cos(theta)+0.8, y=2.5*np.sin(theta), mode='lines',
        fill='toself', fillcolor='rgba(34,211,167,0.12)', line=dict(color='#22d3a7', width=2), name='A Grades (80)'))
    # Labels
    fig_venn.add_annotation(x=-2.0, y=0, text="<b>Sports only<br>70</b>", showarrow=False, font=dict(color='#7c6aff', size=14))
    fig_venn.add_annotation(x=0, y=0, text="<b>Both<br>50</b>", showarrow=False, font=dict(color='#f5b731', size=16))
    fig_venn.add_annotation(x=2.0, y=0, text="<b>A grades only<br>30</b>", showarrow=False, font=dict(color='#22d3a7', size=14))
    fig_venn.add_annotation(x=0, y=-3.2, text="Neither: 50 students", showarrow=False, font=dict(color='#8892b0', size=11))
    fig_venn.update_layout(height=350, showlegend=True,
        xaxis=dict(visible=False, scaleanchor='y', range=[-4.5, 4.5]),
        yaxis=dict(visible=False, range=[-3.8, 3.5]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'), legend=dict(font=dict(color='#8892b0')))
    st.plotly_chart(fig_venn, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="green-box">
    💡 <b>How to read this:</b> When someone says "this student plays sports," you <b>zoom into the blue circle</b>
    (120 students). Within that circle, 50 are in the overlap (also get A grades). So P(A|Sports) = 50/120 = 41.7%.
    <br><br>
    <b>Without</b> the condition: P(A grades) = 80/200 = 40% (looking at everyone).
    <br><b>With</b> the condition: P(A grades | Sports) = 50/120 = 41.7% (looking only at athletes).
    <br><br>
    The condition <b>changed our perspective</b> — we went from looking at the whole school to looking at just the athletes.
    </div>""", unsafe_allow_html=True)

    # Interactive version with telecom example
    st.markdown("#### 🎮 Try It: Telecom Customer Example")
    st.caption("Adjust the percentages and see how conditional probability changes.")

    st.markdown("""<div class="story-box">
    <b>Scenario:</b> You have 1000 telecom customers.
    <br>• Some have a <b>data plan</b> (event B)
    <br>• Some <b>stream video</b> (event A)
    <br>• Some do <b>both</b> (A and B)
    <br><br>
    <b>Question:</b> If I pick a customer who HAS a data plan, what is the probability they also stream video?
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    p_b = c1.slider("% with data plan (B):", 10, 100, 65, 5, key="m2_pb",
                    help="Out of 1000 customers, how many have a data plan?")
    p_ab = c2.slider("% with data plan AND stream (A and B):", 5, p_b, min(30, p_b), 5, key="m2_pab",
                     help="Out of 1000 customers, how many have a data plan AND stream video?")

    p_a_given_b = p_ab / p_b
    n_b = p_b * 10  # out of 1000
    n_ab = p_ab * 10
    n_b_only = n_b - n_ab

    mc = st.columns(4)
    mc[0].metric("Customers with plan", f"{int(n_b)}")
    mc[1].metric("Of those, who stream", f"{int(n_ab)}")
    mc[2].metric("P(stream | has plan)", f"{p_a_given_b:.1%}")
    mc[3].metric("vs P(stream) overall", f"{p_ab}%", f"{p_a_given_b*100 - p_ab:+.1f}%")

    st.markdown(f"""<div class="green-box">
    <b>Calculation:</b> P(stream | data plan) = P(both) / P(data plan) = {p_ab}% / {p_b}% = <b>{p_a_given_b:.1%}</b>
    <br><br>
    Out of <b>{int(n_b)}</b> customers with a data plan, <b>{int(n_ab)}</b> also stream video.
    {'<br><br>💡 Having a data plan makes streaming MORE likely than the base rate!' if p_a_given_b * 100 > p_ab else '<br><br>💡 Interestingly, the conditional probability equals the base rate -- the events might be independent!'}
    </div>""", unsafe_allow_html=True)

    # ── Probability Tree Diagrams ──
    st.markdown("### 🌳 Probability Trees: Independent vs Dependent Events")
    st.caption("Tree diagrams show how probabilities branch at each step. They make complex probability problems visual and intuitive.")

    tree_type = st.radio("Pick an example:", [
        "🪙 Independent: Two coin flips",
        "🃏 Dependent: Drawing cards without replacement",
        "📞 Real-world: Customer churn path",
    ], key="m2_tree_type")

    if "Independent" in tree_type:
        st.markdown("""<div class="story-box">
        <b>Independent events:</b> The outcome of one does NOT affect the other.
        <br>Flipping a coin twice -- the first flip has no effect on the second.
        </div>""", unsafe_allow_html=True)

        # Tree diagram using plotly
        fig_tree = go.Figure()

        # Nodes
        nodes = {
            "Start": (0.5, 1.0),
            "H1": (0.25, 0.65), "T1": (0.75, 0.65),
            "HH": (0.1, 0.3), "HT": (0.35, 0.3),
            "TH": (0.65, 0.3), "TT": (0.9, 0.3),
        }

        # Edges with probabilities
        edges = [
            ("Start", "H1", "H (0.5)"), ("Start", "T1", "T (0.5)"),
            ("H1", "HH", "H (0.5)"), ("H1", "HT", "T (0.5)"),
            ("T1", "TH", "H (0.5)"), ("T1", "TT", "T (0.5)"),
        ]

        for src, dst, label in edges:
            x0, y0 = nodes[src]
            x1, y1 = nodes[dst]
            fig_tree.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                line=dict(color='#7c6aff', width=2), showlegend=False, hoverinfo='skip'))
            fig_tree.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2 + 0.03, text=f"<b>{label}</b>",
                showarrow=False, font=dict(color='#22d3a7', size=11))

        # Node labels
        node_labels = {"Start": "🪙 Start", "H1": "Heads", "T1": "Tails",
                       "HH": "HH<br>P=0.25", "HT": "HT<br>P=0.25", "TH": "TH<br>P=0.25", "TT": "TT<br>P=0.25"}
        node_colors = {"Start": "#f5b731", "H1": "#7c6aff", "T1": "#7c6aff",
                       "HH": "#22d3a7", "HT": "#22d3a7", "TH": "#22d3a7", "TT": "#22d3a7"}

        for name, (x, y) in nodes.items():
            fig_tree.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                marker=dict(color=node_colors[name], size=30), text=[node_labels[name]],
                textposition="bottom center", textfont=dict(color='#e2e8f0', size=10),
                showlegend=False))

        fig_tree.update_layout(height=400, title="Probability Tree: Two Independent Coin Flips",
            xaxis=dict(visible=False, range=[-0.05, 1.05]),
            yaxis=dict(visible=False, range=[0.15, 1.1]),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'))
        st.plotly_chart(fig_tree, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""<div class="green-box">
        💡 <b>Key insight:</b> Each branch has P=0.5 regardless of what happened before.
        The second flip does not "know" what the first flip was.
        <br><b>P(HH) = 0.5 x 0.5 = 0.25</b> -- we multiply because the events are independent.
        <br>All four outcomes (HH, HT, TH, TT) have equal probability of 0.25.
        </div>""", unsafe_allow_html=True)

    elif "Dependent" in tree_type:
        st.markdown("""<div class="story-box">
        <b>Dependent events:</b> The outcome of one CHANGES the probability of the next.
        <br>Drawing cards without replacement -- after you draw one card, the deck has changed.
        </div>""", unsafe_allow_html=True)

        fig_tree2 = go.Figure()

        nodes2 = {
            "Start": (0.5, 1.0),
            "R1": (0.25, 0.65), "B1": (0.75, 0.65),
            "RR": (0.08, 0.3), "RB": (0.38, 0.3),
            "BR": (0.62, 0.3), "BB": (0.92, 0.3),
        }

        # 3 red, 2 blue cards
        edges2 = [
            ("Start", "R1", "Red (3/5)"), ("Start", "B1", "Blue (2/5)"),
            ("R1", "RR", "Red (2/4)"), ("R1", "RB", "Blue (2/4)"),
            ("B1", "BR", "Red (3/4)"), ("B1", "BB", "Blue (1/4)"),
        ]

        for src, dst, label in edges2:
            x0, y0 = nodes2[src]
            x1, y1 = nodes2[dst]
            color = '#f45d6d' if 'Red' in label else '#5eaeff'
            fig_tree2.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                line=dict(color=color, width=2), showlegend=False, hoverinfo='skip'))
            fig_tree2.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2 + 0.03, text=f"<b>{label}</b>",
                showarrow=False, font=dict(color='#e2e8f0', size=10))

        node_labels2 = {"Start": "🃏 Deck<br>3R, 2B", "R1": "Drew Red<br>(4 left)", "B1": "Drew Blue<br>(4 left)",
                        "RR": "RR<br>P=6/20", "RB": "RB<br>P=6/20", "BR": "BR<br>P=6/20", "BB": "BB<br>P=2/20"}

        for name, (x, y) in nodes2.items():
            fig_tree2.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                marker=dict(color='#f5b731' if name == 'Start' else '#7c6aff', size=30),
                text=[node_labels2[name]], textposition="bottom center",
                textfont=dict(color='#e2e8f0', size=9), showlegend=False))

        fig_tree2.update_layout(height=400, title="Probability Tree: Drawing 2 Cards (Without Replacement)",
            xaxis=dict(visible=False, range=[-0.05, 1.05]),
            yaxis=dict(visible=False, range=[0.15, 1.1]),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'))
        st.plotly_chart(fig_tree2, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""<div class="red-box">
        ⚠️ <b>Key insight:</b> The probabilities CHANGE after the first draw!
        <br>- First draw: P(Red) = 3/5. But if you drew Red, now only 2 Red remain out of 4 cards.
        <br>- Second draw: P(Red) = <b>2/4</b> (not 3/5 anymore!)
        <br><br><b>P(RR) = 3/5 x 2/4 = 6/20 = 0.30</b>
        <br><b>P(BB) = 2/5 x 1/4 = 2/20 = 0.10</b> -- much less likely!
        <br><br>This is why we cannot just multiply P(A) x P(B) for dependent events.
        We must use <b>P(A) x P(B|A)</b> -- the probability of B GIVEN that A happened.
        </div>""", unsafe_allow_html=True)

    else:  # Real-world churn path
        st.markdown("""<div class="story-box">
        <b>Real-world tree:</b> A telecom customer's journey through events that lead to churn.
        Each branch shows how the probability of churn changes based on what happened.
        </div>""", unsafe_allow_html=True)

        fig_tree3 = go.Figure()

        nodes3 = {
            "Start": (0.5, 1.0),
            "Complaint": (0.25, 0.7), "NoComplaint": (0.75, 0.7),
            "Resolved": (0.1, 0.4), "Unresolved": (0.35, 0.4),
            "Happy": (0.65, 0.4), "Meh": (0.85, 0.4),
            "Stay1": (0.05, 0.1), "Churn1": (0.18, 0.1),
            "Churn2": (0.28, 0.1), "Churn3": (0.42, 0.1),
            "Stay2": (0.6, 0.1), "Stay3": (0.72, 0.1),
            "Stay4": (0.82, 0.1), "Churn4": (0.95, 0.1),
        }

        edges3 = [
            ("Start", "Complaint", "Complains (40%)"), ("Start", "NoComplaint", "No complaint (60%)"),
            ("Complaint", "Resolved", "Resolved (60%)"), ("Complaint", "Unresolved", "Unresolved (40%)"),
            ("NoComplaint", "Happy", "Satisfied (70%)"), ("NoComplaint", "Meh", "Neutral (30%)"),
            ("Resolved", "Stay1", "Stay (80%)"), ("Resolved", "Churn1", "Churn (20%)"),
            ("Unresolved", "Churn2", "Churn (70%)"), ("Unresolved", "Churn3", "Stay (30%)"),
            ("Happy", "Stay2", "Stay (95%)"), ("Happy", "Stay3", "Churn (5%)"),
            ("Meh", "Stay4", "Stay (75%)"), ("Meh", "Churn4", "Churn (25%)"),
        ]

        for src, dst, label in edges3:
            x0, y0 = nodes3[src]
            x1, y1 = nodes3[dst]
            color = '#f45d6d' if 'Churn' in label else '#22d3a7'
            fig_tree3.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                line=dict(color=color, width=1.5), showlegend=False, hoverinfo='skip'))
            fig_tree3.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2 + 0.02,
                text=label, showarrow=False, font=dict(color='#8892b0', size=8))

        leaf_labels = {
            "Stay1": "STAY<br>P=19.2%", "Churn1": "CHURN<br>P=4.8%",
            "Churn2": "CHURN<br>P=11.2%", "Churn3": "STAY<br>P=4.8%",
            "Stay2": "STAY<br>P=39.9%", "Stay3": "CHURN<br>P=2.1%",
            "Stay4": "STAY<br>P=13.5%", "Churn4": "CHURN<br>P=4.5%",
        }
        mid_labels = {
            "Start": "Customer", "Complaint": "Complained", "NoComplaint": "No Complaint",
            "Resolved": "Resolved", "Unresolved": "Unresolved", "Happy": "Satisfied", "Meh": "Neutral",
        }

        for name, (x, y) in nodes3.items():
            label = leaf_labels.get(name, mid_labels.get(name, name))
            color = '#f45d6d' if 'CHURN' in label else '#22d3a7' if 'STAY' in label else '#f5b731'
            fig_tree3.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                marker=dict(color=color, size=18 if name in leaf_labels else 22),
                text=[label], textposition="bottom center",
                textfont=dict(color='#e2e8f0', size=8), showlegend=False))

        fig_tree3.update_layout(height=450, title="Customer Churn Probability Tree",
            xaxis=dict(visible=False, range=[-0.05, 1.05]),
            yaxis=dict(visible=False, range=[-0.05, 1.1]),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'))
        st.plotly_chart(fig_tree3, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""<div class="green-box">
        💡 <b>Reading this tree:</b> Follow any path from top to bottom. Multiply the probabilities along the way.
        <br><br><b>Worst path:</b> Customer complains (40%) -> Unresolved (40%) -> Churns (70%)
        <br>P = 0.40 x 0.40 x 0.70 = <b>11.2%</b> of all customers follow this path to churn.
        <br><br><b>Best path:</b> No complaint (60%) -> Satisfied (70%) -> Stays (95%)
        <br>P = 0.60 x 0.70 x 0.95 = <b>39.9%</b> of customers are happy and stay.
        <br><br><b>Total churn probability:</b> 4.8% + 11.2% + 2.1% + 4.5% = <b>22.6%</b>
        <br><br>This is how businesses model customer journeys -- each branch is a decision point where you can intervene.
        </div>""", unsafe_allow_html=True)

    # ── Bayes' Theorem ──
    st.markdown("### 📖 Bayes' Theorem")
    st.markdown("""<div class="story-box">
    Bayes' lets you <b>flip</b> conditional probability: if you know P(test positive | disease), you can calculate P(disease | test positive).
    <br><br><b>The classic trap:</b> A disease affects 1 in 1000 people. The test is 99% accurate. You test positive. What's the chance you're actually sick?
    <br><br><b>Intuition:</b> Out of 1000 people: ~1 truly sick (tests positive). ~50 healthy people also test positive (5% false positive rate). So only 1 out of 51 positives is real → <b>~2% chance</b> you're actually sick, despite a "99% accurate" test!
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Bayes' Calculator")
    c1, c2, c3 = st.columns(3)
    prevalence = c1.slider("Disease prevalence:", 0.001, 0.1, 0.001, 0.001, format="%.3f", key="m2_prev")
    sensitivity = c2.slider("Test sensitivity (true positive rate):", 0.5, 1.0, 0.99, 0.01, key="m2_sens")
    fpr = c3.slider("False positive rate:", 0.01, 0.2, 0.05, 0.01, key="m2_fpr")

    p_pos = sensitivity * prevalence + fpr * (1 - prevalence)
    p_disease_given_pos = (sensitivity * prevalence) / p_pos

    st.markdown(f"""<div class="{'green-box' if p_disease_given_pos > 0.5 else 'red-box'}">
    <b>P(actually sick | tested positive) = {p_disease_given_pos:.1%}</b>
    <br><br>{'✅ The test is reliable here — a positive result likely means disease.' if p_disease_given_pos > 0.5 else f'⚠️ Despite a {sensitivity:.0%} accurate test, a positive result only means a <b>{p_disease_given_pos:.1%}</b> chance of disease! This is because the disease is so rare that false positives overwhelm true positives.'}
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="green-box">
    👉 <b>Telecom example:</b> "Given that a customer called support 3 times this month (B), what's the probability they'll churn (A)?" Bayes' theorem lets you update your churn prediction based on new evidence (the support calls).
    </div>""", unsafe_allow_html=True)

    iq([
        {"q": "Explain Bayes' Theorem with a real example.", "d": "Hard", "c": ["Google", "Apple", "Amazon"],
         "a": "Bayes' updates beliefs with new evidence. <b>Example:</b> Disease affects 1/1000. Test is 99% accurate, 5% false positive. You test positive. Out of 1000 people: 1 true positive + 50 false positives = 51 positives. P(sick|positive) = 1/51 ≈ 2%. The 'base rate' (rarity of disease) dominates. This is the <b>base rate fallacy</b>.",
         "t": "Walk through the numbers step by step. Draw a 2×2 table if you can."},
        {"q": "What's the difference between independent and dependent events?", "d": "Easy", "c": ["Google", "Meta"],
         "a": "<b>Independent:</b> One event doesn't affect the other. P(A and B) = P(A) × P(B). Example: two coin flips. <b>Dependent:</b> One event changes the probability of the other. P(A and B) = P(A) × P(B|A). Example: drawing cards without replacement — the first draw changes what's left.",
         "t": "Use the card deck example — it's concrete and universally understood."},
        {"q": "A spam filter has 95% accuracy. 2% of emails are spam. An email is flagged as spam. What's the probability it's actually spam?", "d": "Hard", "c": ["Google", "Netflix"],
         "a": "This is Bayes': P(spam|flagged) = P(flagged|spam)×P(spam) / P(flagged). P(flagged) = 0.95×0.02 + 0.05×0.98 = 0.019 + 0.049 = 0.068. P(spam|flagged) = 0.019/0.068 ≈ <b>28%</b>. Despite 95% accuracy, only 28% of flagged emails are actually spam! The low base rate (2% spam) means false positives dominate.",
         "t": "This is the same structure as the disease testing question. Show you can apply Bayes' to any domain."},
        {"q": "You flip a fair coin 10 times and get 10 heads. What's the probability of heads on the 11th flip?", "d": "Easy", "c": ["Google", "Meta", "General"],
         "a": "<b>50%.</b> Each flip is independent. The coin has no memory. This is the <b>gambler's fallacy</b> — the belief that past results affect future independent events. The probability is always 0.5, regardless of what happened before.",
         "t": "Name the gambler's fallacy explicitly. Then mention: 'However, if I suspected the coin was unfair, I'd use Bayesian reasoning to update my belief about the coin's fairness.'"},
    ])


# ═══════════════════════════════════════
# MODULE 3: DISTRIBUTIONS
# ═══════════════════════════════════════
elif module == "📈 M3: Distributions":
    st.markdown("# 📈 Module 3: Distributions")
    st.caption("Week 2 · The shapes data takes — and why they matter.")

    st.markdown("""<div class="story-box">
    A <b>distribution</b> describes the <b>shape</b> of your data — how values are spread out.
    Is it a bell curve? Is it lopsided? Does it have two peaks? The shape tells you which tools to use
    and what to expect.
    </div>""", unsafe_allow_html=True)

    # ── Normal Distribution ──
    st.markdown("### 📖 Normal Distribution (The Bell Curve)")
    st.markdown("""<div class="story-box">
    The most important distribution. Written as <b>X ~ N(μ, σ²)</b> where μ = mean, σ = std dev.
    <br><br><b>Why it matters:</b> Heights, test scores, measurement errors, stock returns — countless real-world things follow this shape. And the Central Limit Theorem says that <b>averages of anything</b> become normal with enough samples.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Shape the Bell Curve")
    st.caption("The two sliders control the two parameters that define any normal distribution.")

    c1, c2 = st.columns(2)
    mu = c1.slider("μ (mu) — The average value:", -5.0, 5.0, 0.0, 0.5, key="m3_mu",
                   help="This is the MEAN. It controls where the peak of the bell sits on the number line. Slide it right = the whole curve shifts right.")
    sigma = c2.slider("σ (sigma) — How spread out the data is:", 0.3, 4.0, 1.0, 0.1, key="m3_sigma",
                      help="This is the STANDARD DEVIATION. Small σ = data is tightly clustered around the mean (tall narrow bell). Large σ = data is spread out (short wide bell).")

    st.markdown(f"""<div class="story-box">
    <b>What you are controlling:</b>
    <br>• <b>μ = {mu}</b> — the average. If this were exam scores, μ={mu} means the class average is {mu}. The peak of the bell sits here.
    <br>• <b>σ = {sigma}</b> — the standard deviation. It tells you how far a typical value is from the average.
    {'A small σ means almost everyone scored close to the average (tight bell).' if sigma < 1.5 else 'A large σ means scores are spread out — some very high, some very low (wide bell).'}
    </div>""", unsafe_allow_html=True)

    x = np.linspace(-10, 10, 500)
    y = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor='rgba(124,106,255,0.2)', line=dict(color='#7c6aff', width=2)))
    fig.add_vline(x=mu, line_dash="dash", line_color="#22d3a7", annotation_text=f"μ = {mu} (average)")
    fig.add_vrect(x0=mu-sigma, x1=mu+sigma, fillcolor="rgba(34,211,167,0.1)", line_width=0)
    # Mark sigma boundaries explicitly
    fig.add_vline(x=mu-sigma, line_dash="dot", line_color="#f5b731",
                  annotation_text=f"μ-σ = {mu-sigma:.1f}", annotation_position="bottom left")
    fig.add_vline(x=mu+sigma, line_dash="dot", line_color="#f5b731",
                  annotation_text=f"μ+σ = {mu+sigma:.1f}", annotation_position="bottom right")
    # Double-headed arrow showing sigma width
    mid_y = max(y) * 0.5
    fig.add_annotation(x=mu, y=mid_y, ax=mu-sigma, ay=mid_y, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor="#f5b731")
    fig.add_annotation(x=mu, y=mid_y, ax=mu+sigma, ay=mid_y, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor="#f5b731")
    fig.add_annotation(x=mu + sigma/2, y=mid_y + max(y)*0.08, text=f"<b>σ = {sigma}</b>",
                       showarrow=False, font=dict(color="#f5b731", size=13))
    fig.add_annotation(x=mu - sigma/2, y=mid_y + max(y)*0.08, text=f"<b>σ = {sigma}</b>",
                       showarrow=False, font=dict(color="#f5b731", size=13))
    # Label the 68% band
    fig.add_annotation(x=mu, y=max(y)*0.25, text="<b>68% of data</b>",
                       showarrow=False, font=dict(color="#22d3a7", size=12))
    fig.update_layout(height=350, title=f"Normal Distribution: N({mu}, {sigma}²)", xaxis_title="Value", yaxis_title="Probability Density", **DL)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">
    💡 <b>Reading this chart:</b>
    <br>• The <b>peak</b> is at μ = {mu} — this is where most values cluster.
    <br>• The <b>green shaded band</b> covers μ ± σ = [{mu-sigma:.1f} to {mu+sigma:.1f}] — <b>68%</b> of all data falls in here.
    <br>• <b>Try it:</b> Increase σ and watch the bell get wider and shorter. Decrease σ and it gets taller and narrower. The total area under the curve is always 100%.
    </div>""", unsafe_allow_html=True)

    # ── Skewed ──
    st.markdown("### 📖 Skewed Distributions")
    st.markdown("""<div class="story-box">
    Not all data is symmetric. <b>Right-skewed</b> (positive skew): long tail to the right. Example: income — most people earn moderate amounts, a few earn millions.
    <b>Left-skewed</b> (negative skew): long tail to the left. Example: age at retirement — most retire around 65, a few retire very young.
    <br><br><b>Key sign:</b> If mean > median → right-skewed. If mean < median → left-skewed.
    </div>""", unsafe_allow_html=True)

    skew_type = st.radio("Pick a shape:", ["Symmetric (Normal)", "Right-Skewed (Income-like)", "Left-Skewed"], horizontal=True, key="m3_skew")
    np.random.seed(42)
    if "Symmetric" in skew_type:
        sdata = np.random.normal(50, 10, 2000)
    elif "Right" in skew_type:
        sdata = np.random.exponential(20, 2000) + 20
    else:
        sdata = 100 - np.random.exponential(20, 2000)

    fig_s = go.Figure(go.Histogram(x=sdata, nbinsx=40, marker_color="#22d3a7", opacity=0.7))
    fig_s.add_vline(x=sdata.mean(), line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: {sdata.mean():.1f}")
    fig_s.add_vline(x=np.median(sdata), line_dash="dash", line_color="#7c6aff", annotation_text=f"Median: {np.median(sdata):.1f}")
    fig_s.update_layout(height=280, title=skew_type, **DL)
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    # ── Poisson ──
    st.markdown("### 📖 Poisson Distribution")
    st.markdown("""<div class="story-box">
    Counts of <b>rare events</b> in a fixed time/space. "How many support calls per hour?" "How many server errors per day?"
    <br>Defined by one parameter: <b>λ (lambda)</b> = average rate of events.
    </div>""", unsafe_allow_html=True)

    lam = st.slider("λ (average events per period):", 1, 20, 4, key="m3_lam")
    pois = np.random.poisson(lam, 5000)
    fig_p = go.Figure(go.Histogram(x=pois, nbinsx=int(max(pois)-min(pois)+1), marker_color="#f5b731", opacity=0.7))
    fig_p.add_vline(x=lam, line_dash="dash", line_color="#f45d6d", annotation_text=f"λ={lam}")
    fig_p.update_layout(height=250, title=f"Poisson(λ={lam}): Count of events per period", xaxis_title="Events", yaxis_title="Frequency", **DL)
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="green-box">👉 <b>Telecom:</b> Dropped calls per hour, support tickets per day, network errors per minute — all follow Poisson. If λ=3 errors/hour and you see 10 in one hour, that's a red flag.</div>""", unsafe_allow_html=True)

    # ── Binomial ──
    st.markdown("### 📖 Binomial Distribution")
    st.markdown("""<div class="story-box">
    Counts <b>successes in a fixed number of trials</b>, where each trial has the same probability of success.
    <br><br><b>Examples:</b> "Out of 100 emails sent, how many get opened?" "Out of 50 customers contacted, how many buy?"
    <br><br>Defined by two parameters: <b>n</b> (number of trials) and <b>p</b> (probability of success per trial).
    <br><br><b>Notation:</b> X ~ Binomial(n, p)
    </div>""", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    n_trials = bc1.slider("n (number of trials):", 5, 100, 20, key="m3_binom_n")
    p_success = bc2.slider("p (probability of success per trial):", 0.01, 0.99, 0.3, 0.01, key="m3_binom_p")

    np.random.seed(42)
    binom_data = np.random.binomial(n_trials, p_success, 5000)
    expected = n_trials * p_success

    fig_b = go.Figure(go.Histogram(x=binom_data, nbinsx=int(min(n_trials + 1, 50)), marker_color="#e879a8", opacity=0.7))
    fig_b.add_vline(x=expected, line_dash="dash", line_color="#22d3a7", annotation_text=f"Expected: {expected:.1f}")
    fig_b.update_layout(height=280, title=f"Binomial(n={n_trials}, p={p_success}): Successes out of {n_trials} trials",
                        xaxis_title="Number of Successes", yaxis_title="Frequency", **DL)
    st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">
    💡 With <b>n={n_trials}</b> trials and <b>p={p_success}</b> success rate, you'd expect about <b>{expected:.1f} successes</b> on average.
    <br>👉 <b>Telecom:</b> "Out of 1000 customers who received a retention offer, how many will accept?" If p=0.15, expect ~150 acceptances.
    </div>""", unsafe_allow_html=True)

    # ── Why Distributions Matter ──
    st.markdown("### 🧠 Why Distributions Matter for Data Science & ML")
    st.markdown("""<div class="story-box">
    Distributions aren't just academic — they're the <b>foundation of every model and every test</b> you'll use.
    </div>""", unsafe_allow_html=True)

    reasons = [
        ("🔍 Assumption checking", "#7c6aff",
         "Many statistical tests (t-test, ANOVA) and models (linear regression) <b>assume your data follows a specific distribution</b> — usually normal. If your data is heavily skewed, these tools give wrong answers. Knowing the distribution tells you which tools are safe to use."),
        ("🚨 Anomaly detection", "#f45d6d",
         "If you know data follows a Poisson distribution with λ=3 errors/hour, then seeing <b>15 errors in one hour</b> is extremely unlikely under normal conditions. That's how you detect anomalies — by knowing what 'normal' looks like and flagging what doesn't fit."),
        ("🤖 Model behavior", "#22d3a7",
         "Many ML models work better when features are <b>normally distributed</b>. Skewed features can dominate models unfairly. That's why data scientists often <b>transform</b> skewed data (log transform, Box-Cox) to make it more normal before modeling."),
        ("📊 Choosing the right metric", "#f5b731",
         "If your target variable follows a <b>Poisson distribution</b> (count data), you should use Poisson regression, not linear regression. If it's <b>binomial</b> (yes/no), use logistic regression. The distribution of your data determines which model family to use."),
    ]

    for title, color, desc in reasons:
        st.markdown(f"""<div class="story-box" style="border-left:4px solid {color}">
        <b>{title}</b><br>{desc}
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-box">
    <b>🎯 Key Takeaway:</b> Always look at the <b>shape</b> of your data before choosing a tool or model.
    A histogram takes 5 seconds and can save you from hours of wrong analysis.
    </div>""", unsafe_allow_html=True)

    iq([
        {"q": "What is the normal distribution and why is it important?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "Bell-shaped, symmetric, defined by mean (μ) and std dev (σ). <b>Important because:</b> (1) Many natural phenomena follow it. (2) The Central Limit Theorem says sample means become normal regardless of the original distribution. (3) Many statistical tests assume normality. (4) It's the foundation of confidence intervals and hypothesis testing.",
         "t": "Mention CLT — it's the bridge between distributions and inferential statistics."},
        {"q": "When would you use Poisson vs Binomial vs Normal?", "d": "Medium", "c": ["Google", "Netflix", "Amazon"],
         "a": "<b>Normal:</b> Continuous measurements (height, temperature, test scores). Can be any real number. <b>Binomial:</b> Count of successes in a fixed number of trials (emails opened out of 100 sent, defective items out of 50 inspected). Needs fixed n and constant p. <b>Poisson:</b> Count of events in a fixed time/space with no upper limit (support calls per hour, errors per day). No fixed n. <b>Key:</b> Binomial has a ceiling (n), Poisson doesn't. As n→∞ and p→0, Binomial → Poisson. As λ→large, Poisson → Normal.",
         "t": "Show you understand the relationships between distributions, not just the definitions."},
        {"q": "How do you check if data is normally distributed?", "d": "Medium", "c": ["Meta", "Apple"],
         "a": "<b>Visual:</b> Histogram (bell-shaped?), Q-Q plot (points on diagonal?). <b>Statistical tests:</b> Shapiro-Wilk (best for small samples), Kolmogorov-Smirnov, Anderson-Darling. <b>Quick checks:</b> Mean ≈ median? Skewness ≈ 0? Kurtosis ≈ 3? <b>Practical note:</b> Perfect normality is rare. Most tests work fine with 'approximately normal' data, especially with large samples (CLT).",
         "t": "Start with visual methods, then mention formal tests. Say 'in practice, approximate normality is usually sufficient.'"},
        {"q": "Your feature is heavily right-skewed. How does this affect your model and what do you do?", "d": "Medium", "c": ["Google", "Meta", "Netflix"],
         "a": "<b>Impact:</b> (1) Linear models assume roughly normal features — skewed data violates this. (2) The long tail means a few extreme values dominate the model. (3) Distance-based models (KNN, K-means) are distorted because the scale is uneven. <b>Fixes:</b> (1) <b>Log transform:</b> log(x) compresses the tail. Most common fix. (2) <b>Square root transform:</b> milder than log. (3) <b>Box-Cox transform:</b> finds the optimal power transform automatically. (4) <b>Binning:</b> convert to categories. (5) Use tree-based models (robust to skew). <b>Always:</b> Plot before and after to verify the transform helped.",
         "t": "Mention log transform first — it's the most common. Then say 'tree-based models are robust to skew' as an alternative."},
        {"q": "What is the Central Limit Theorem and how does it relate to distributions?", "d": "Medium", "c": ["Google", "Meta"],
         "a": "CLT says: take many samples from ANY distribution, compute the mean of each sample, and those means will form a <b>normal distribution</b> — regardless of the original shape. <b>Connection:</b> This is why the normal distribution is so important — even if your raw data is Poisson, exponential, or uniform, the <b>averages</b> of samples from it will be normal. This lets us use normal-based tools (z-tests, confidence intervals) on almost any data.",
         "t": "Draw the picture: any shape on the left → bell curve of means on the right."},
        {"q": "A dataset has a bimodal distribution (two peaks). What does this tell you?", "d": "Medium", "c": ["Amazon", "Apple"],
         "a": "Two peaks usually mean <b>two distinct groups are mixed together</b>. Example: heights of adults — one peak for women (~5'4\") and one for men (~5'9\"). <b>What to do:</b> (1) Investigate — is there a categorical variable that separates the groups? (2) Consider splitting the data and modeling each group separately. (3) Don't use the mean — it falls between the peaks and represents nobody. <b>In telecom:</b> Bimodal call duration might mean short calls (quick questions) and long calls (complex issues) — two different customer behaviors.",
         "t": "Always say 'investigate what's causing the two groups' — shows analytical thinking."},
    ])


# ═══════════════════════════════════════
# MODULE 4: INFERENTIAL STATISTICS
# ═══════════════════════════════════════
elif module == "🧪 M4: Inferential Statistics":
    st.markdown("# 🧪 Module 4: Inferential Statistics")
    st.caption("Week 3 · Is this result real or just random noise?")

    st.markdown("""<div class="story-box">
    You can't survey every customer. You can't test every product. You take a <b>sample</b> and use it
    to make conclusions about the whole population. Inferential statistics tells you <b>how confident</b>
    you can be in those conclusions.
    </div>""", unsafe_allow_html=True)

    # ── Sampling ──
    st.markdown("### 📖 Sampling & the Central Limit Theorem")
    st.markdown("""<div class="story-box">
    <b>CLT:</b> Take many random samples from ANY population. The means of those samples form a <b>normal distribution</b> — even if the original data isn't normal. Larger samples → tighter bell curve.
    <br><br>This is why we can use normal-distribution-based tools (confidence intervals, z-tests) on almost any data, as long as our sample is large enough (~30+).
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Watch CLT in Action")
    pop_shape = st.radio("Population shape:", ["Uniform (flat)", "Exponential (skewed)", "Bimodal (two peaks)"], horizontal=True, key="m4_pop")
    sample_size = st.slider("Sample size (n):", 5, 200, 30, 5, key="m4_n")

    np.random.seed(42)
    if "Uniform" in pop_shape:
        population = np.random.uniform(0, 100, 100000)
    elif "Exponential" in pop_shape:
        population = np.random.exponential(30, 100000)
    else:
        population = np.concatenate([np.random.normal(30, 8, 50000), np.random.normal(70, 8, 50000)])

    sample_means = [np.random.choice(population, sample_size).mean() for _ in range(2000)]

    c1, c2 = st.columns(2)
    with c1:
        fig_pop = go.Figure(go.Histogram(x=population, nbinsx=50, marker_color="#7c6aff", opacity=0.6))
        fig_pop.update_layout(height=250, title="Original Population (NOT normal)", **DL)
        st.plotly_chart(fig_pop, use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig_clt = go.Figure(go.Histogram(x=sample_means, nbinsx=40, marker_color="#22d3a7", opacity=0.7))
        fig_clt.update_layout(height=250, title=f"Distribution of Sample Means (n={sample_size}) → NORMAL!", **DL)
        st.plotly_chart(fig_clt, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="green-box">💡 <b>Magic of CLT:</b> The left chart can be ANY shape. But the right chart (sample means) always becomes a bell curve. Larger n → tighter curve → more precise estimates.</div>""", unsafe_allow_html=True)

    # ── Z-Score ──
    st.markdown("### 📖 Z-Score — How Unusual Is This Value?")
    st.markdown("""<div class="story-box">
    A Z-score answers: <b>"How many standard deviations is this value from the mean?"</b>
    <br><br><b>Formula:</b> z = (x - μ) / σ
    <br><br><b>Interpretation:</b>
    <br>• z = 0 → exactly at the mean (perfectly average)
    <br>• z = 1 → one std dev above the mean (top ~16%)
    <br>• z = 2 → two std devs above (top ~2.5%) — unusual
    <br>• z = 3 → three std devs above (top ~0.1%) — very rare, likely an outlier
    <br><br><b>Analogy:</b> If the average exam score is 75 with std dev 10, and you scored 95:
    <br>z = (95 - 75) / 10 = <b>2.0</b> → you're 2 standard deviations above average. Only ~2.5% of students scored higher.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Z-Score Calculator")
    st.caption("Enter a value and see where it falls on the bell curve.")
    zc1, zc2, zc3 = st.columns(3)
    z_mean = zc1.number_input("Population mean (μ):", value=75.0, key="m4_zmean")
    z_std = zc2.number_input("Std dev (σ):", value=10.0, min_value=0.1, key="m4_zstd")
    z_val = zc3.number_input("Your value (x):", value=95.0, key="m4_zval")

    z_score = (z_val - z_mean) / z_std
    from scipy.stats import norm
    percentile = norm.cdf(z_score) * 100

    mc = st.columns(3)
    mc[0].metric("Z-Score", f"{z_score:.2f}", help="How many std devs from the mean")
    mc[1].metric("Percentile", f"{percentile:.1f}%", help="% of values below this point")
    mc[2].metric("How unusual?",
                 "Normal" if abs(z_score) < 1.5 else "Unusual" if abs(z_score) < 2.5 else "Very rare!",
                 help="|z| < 1.5 = normal, 1.5-2.5 = unusual, > 2.5 = rare")

    # Bell curve with value marked
    x_bell = np.linspace(z_mean - 4*z_std, z_mean + 4*z_std, 500)
    y_bell = norm.pdf(x_bell, z_mean, z_std)
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=x_bell, y=y_bell, fill='tozeroy', fillcolor='rgba(124,106,255,0.15)',
                                line=dict(color='#7c6aff', width=2), name='Distribution'))
    # Shade area up to z_val
    x_shade = x_bell[x_bell <= z_val]
    y_shade = norm.pdf(x_shade, z_mean, z_std)
    fig_z.add_trace(go.Scatter(x=np.append(x_shade, [z_val, x_shade[0]]),
                                y=np.append(y_shade, [0, 0]),
                                fill='toself', fillcolor='rgba(34,211,167,0.25)',
                                line=dict(color='rgba(0,0,0,0)'), name=f'{percentile:.1f}% below'))
    fig_z.add_vline(x=z_val, line_dash="dash", line_color="#f45d6d",
                    annotation_text=f"x={z_val} (z={z_score:.2f})")
    fig_z.add_vline(x=z_mean, line_dash="dot", line_color="#8892b0", annotation_text=f"μ={z_mean}")
    fig_z.update_layout(height=300, title=f"Where does {z_val} fall? (z = {z_score:.2f}, top {100-percentile:.1f}%)",
                        xaxis_title="Value", yaxis_title="Probability Density", **DL)
    st.plotly_chart(fig_z, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">
    💡 <b>Reading this chart:</b> The green shaded area = <b>{percentile:.1f}%</b> of the population falls below your value.
    {'This value is within the normal range — nothing unusual.' if abs(z_score) < 1.5 else 'This value is in the tail — it stands out from the crowd.' if abs(z_score) < 2.5 else 'This is extremely rare (beyond 2.5σ) — investigate! Could be an outlier, error, or genuinely exceptional.'}
    <br><br>👉 <b>Telecom:</b> If average latency is 45ms (σ=15ms) and a cell tower shows 105ms → z = (105-45)/15 = <b>4.0</b> → extremely abnormal. Investigate immediately.
    </div>""", unsafe_allow_html=True)

    # ── Z-Test vs T-Test ──
    st.markdown("### 📖 Z-Test vs T-Test — Which One to Use?")
    st.markdown("""<div class="story-box">
    Both tests answer the same question: <b>"Is the difference between two groups real or just random noise?"</b>
    The difference is <b>when</b> you use each one.
    </div>""", unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("""<div class="story-box" style="border-left:4px solid #7c6aff">
        <b style="color:#7c6aff">Z-Test</b><br><br>
        <b>Use when:</b><br>
        • You <b>know</b> the population std dev (σ)<br>
        • Sample size is <b>large</b> (n > 30)<br><br>
        <b>Formula:</b> z = (x̄ - μ) / (σ / √n)<br><br>
        <b>In practice:</b> Rare — you almost never know the true σ. Used mainly in textbooks and when you have massive datasets (where t-test and z-test give the same result anyway).
        </div>""", unsafe_allow_html=True)
    with tc2:
        st.markdown("""<div class="story-box" style="border-left:4px solid #22d3a7">
        <b style="color:#22d3a7">T-Test</b><br><br>
        <b>Use when:</b><br>
        • You <b>don't know</b> the population σ (estimate from sample)<br>
        • Sample size is <b>small</b> (n < 30)<br><br>
        <b>Formula:</b> t = (x̄ - μ) / (s / √n)<br><br>
        <b>In practice:</b> This is what you'll use 99% of the time. A/B tests, comparing groups, checking if a metric changed — all t-tests. As n grows large, t-test → z-test.
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-box">
    <b>🎯 Rule of thumb:</b> When in doubt, use the <b>t-test</b>. It works for both small and large samples.
    The z-test is a special case of the t-test when n is very large.
    </div>""", unsafe_allow_html=True)

    # ── Types of T-Tests ──
    st.markdown("### 📖 Three Types of T-Tests")
    st.markdown("""<div class="story-box">
    <b>1. One-sample t-test:</b> "Is this group's average different from a known value?"
    <br><i>Example: "Is our average response time different from the 200ms SLA target?"</i>
    <br><br><b>2. Two-sample (independent) t-test:</b> "Are these two groups different from each other?"
    <br><i>Example: "Do customers on monthly contracts churn more than annual contract customers?"</i>
    <br><br><b>3. Paired t-test:</b> "Did the same group change after a treatment?"
    <br><i>Example: "Did the same cell towers improve after the firmware update?"</i>
    </div>""", unsafe_allow_html=True)

    # ── Interactive T-Test ──
    st.markdown("#### 🎮 Try It: Two-Sample T-Test")
    st.caption("Compare two groups and see if the difference is statistically significant.")

    tt_c1, tt_c2 = st.columns(2)
    grp_a_mean = tt_c1.slider("Group A mean:", 10, 100, 50, key="m4_tt_a")
    grp_b_mean = tt_c2.slider("Group B mean:", 10, 100, 55, key="m4_tt_b")
    tt_c3, tt_c4 = st.columns(2)
    grp_std = tt_c3.slider("Both groups std dev:", 5, 40, 15, key="m4_tt_std")
    grp_n = tt_c4.slider("Sample size per group:", 10, 500, 50, key="m4_tt_n")

    np.random.seed(42)
    samp_a = np.random.normal(grp_a_mean, grp_std, grp_n)
    samp_b = np.random.normal(grp_b_mean, grp_std, grp_n)

    from scipy.stats import ttest_ind as ttest_2
    t_stat_tt, p_val_tt = ttest_2(samp_a, samp_b)

    mc = st.columns(4)
    mc[0].metric("Group A Mean", f"{samp_a.mean():.1f}")
    mc[1].metric("Group B Mean", f"{samp_b.mean():.1f}", f"{samp_b.mean()-samp_a.mean():+.1f}")
    mc[2].metric("T-Statistic", f"{t_stat_tt:.2f}", help="How many standard errors apart the means are")
    mc[3].metric("P-Value", f"{p_val_tt:.4f}")

    fig_tt = go.Figure()
    fig_tt.add_trace(go.Histogram(x=samp_a, name="Group A", marker_color="#7c6aff", opacity=0.5, nbinsx=20))
    fig_tt.add_trace(go.Histogram(x=samp_b, name="Group B", marker_color="#22d3a7", opacity=0.5, nbinsx=20))
    fig_tt.update_layout(barmode="overlay", height=280, title="Group A vs Group B", **DL)
    st.plotly_chart(fig_tt, use_container_width=True, config={"displayModeBar": False})

    if p_val_tt < 0.05:
        st.markdown(f"""<div class="green-box">✅ <b>p = {p_val_tt:.4f}</b> — The difference IS statistically significant. These groups are likely truly different, not just random variation.</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="red-box">❌ <b>p = {p_val_tt:.4f}</b> — NOT significant. The difference could easily be random noise. Try increasing the sample size or the difference between means.</div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="green-box">
    💡 <b>What affects significance?</b> Try adjusting the sliders:
    <br>• <b>Bigger difference</b> between means → easier to detect (lower p-value)
    <br>• <b>Larger sample size</b> → more statistical power → easier to detect small differences
    <br>• <b>Higher std dev</b> → more noise → harder to detect differences
    <br><br>👉 <b>Telecom:</b> "Is average latency in Region A different from Region B?" → two-sample t-test. "Did latency improve after the upgrade?" → paired t-test.
    </div>""", unsafe_allow_html=True)

    # ── Hypothesis Testing ──
    st.markdown("### 📖 Hypothesis Testing & P-Values")
    st.markdown("""<div class="story-box">
    <b>The question:</b> "Is this result real, or could it have happened by chance?"
    <br><br><b>How it works:</b>
    <br>1. <b>Null hypothesis (H₀):</b> "Nothing interesting is happening" (no effect, no difference)
    <br>2. <b>Alternative hypothesis (H₁):</b> "Something real is happening"
    <br>3. Collect data, compute a <b>test statistic</b>
    <br>4. Calculate the <b>p-value:</b> "If H₀ is true, how likely is this result?"
    <br>5. If p < 0.05 → reject H₀ → "The result is statistically significant"
    </div>""", unsafe_allow_html=True)

    # ── P-Value Deep Dive ──
    st.markdown("### 📖 What Exactly Is a P-Value?")
    st.caption("This is one of the most misunderstood concepts in all of statistics. Let's get it right.")

    st.markdown("""<div class="story-box">
    Imagine you have a coin. You flip it <b>100 times</b> and get <b>60 heads</b>. Is the coin unfair, or did you just get lucky?
    <br><br>
    The p-value answers: <b>"If the coin IS fair (50/50), what's the probability of getting 60 or more heads just by random chance?"</b>
    <br><br>
    If that probability is very low (say 2%), you conclude: "It's very unlikely a fair coin would give me 60 heads. The coin is probably unfair."
    <br><br>
    That 2% IS the p-value.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: The Coin Flip P-Value")
    st.caption("You flip a coin 100 times. How many heads would make you suspicious?")

    n_heads = st.slider("Number of heads out of 100 flips:", 40, 75, 60, 1, key="m4_pval_heads")

    from scipy.stats import binom
    # P-value: probability of getting this many or more heads with a fair coin
    p_value_coin = 1 - binom.cdf(n_heads - 1, 100, 0.5)

    mc = st.columns(3)
    mc[0].metric("Heads", f"{n_heads} / 100")
    mc[1].metric("P-Value", f"{p_value_coin:.4f}",
                 help="Probability of getting this many (or more) heads with a FAIR coin")
    mc[2].metric("Conclusion",
                 "🟢 Looks fair" if p_value_coin > 0.05 else "🔴 Suspicious!" if p_value_coin > 0.01 else "🔴 Almost certainly unfair")

    # Visual: show the distribution with the observed value
    x_vals = np.arange(30, 71)
    y_vals = binom.pmf(x_vals, 100, 0.5)
    colors = ['#f45d6d' if x >= n_heads else '#7c6aff' for x in x_vals]

    fig_pv = go.Figure()
    fig_pv.add_trace(go.Bar(x=x_vals, y=y_vals, marker_color=colors, opacity=0.7))
    fig_pv.add_vline(x=n_heads, line_dash="dash", line_color="#f5b731",
                     annotation_text=f"Your result: {n_heads} heads")
    fig_pv.add_vline(x=50, line_dash="dot", line_color="#22d3a7",
                     annotation_text="Expected: 50")
    fig_pv.update_layout(height=320,
                         title=f"If the coin is fair, how likely is {n_heads}+ heads? (red area = p-value = {p_value_coin:.4f})",
                         xaxis_title="Number of Heads", yaxis_title="Probability", **DL)
    st.plotly_chart(fig_pv, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="{'green-box' if p_value_coin > 0.05 else 'red-box'}">
    📖 <b>Reading this chart:</b> The blue bars show all possible outcomes if the coin is fair.
    The <b style="color:#f45d6d">red bars</b> show outcomes as extreme as yours ({n_heads}+ heads).
    The total height of the red bars = <b>p-value = {p_value_coin:.4f}</b>.
    {'<br><br>The red area is large — getting ' + str(n_heads) + ' heads with a fair coin is not that unusual. No reason to suspect the coin.' if p_value_coin > 0.05 else '<br><br>The red area is tiny — getting ' + str(n_heads) + ' heads with a fair coin would be very unlikely. The coin is probably unfair (or something else is going on).'}
    </div>""", unsafe_allow_html=True)

    # ── What p-value does NOT mean ──
    st.markdown("### ⚠️ What P-Value Does NOT Mean")
    st.caption("This is where most people (even some scientists) get it wrong.")

    st.markdown("""<div class="red-box">
    <b>❌ WRONG:</b> "There's a 3% chance the null hypothesis is true."
    <br><b>✅ RIGHT:</b> "IF the null hypothesis is true, there's a 3% chance of seeing results this extreme."
    <br><br>
    <b>❌ WRONG:</b> "The effect is large / important."
    <br><b>✅ RIGHT:</b> P-value says nothing about <b>size</b>. A tiny, meaningless difference can have p = 0.001 with enough data.
    <br><br>
    <b>❌ WRONG:</b> "p < 0.05 means it's definitely real."
    <br><b>✅ RIGHT:</b> 0.05 is just a convention. It means "unlikely enough that we'll act on it." But 1 in 20 significant results is a false alarm by definition.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box">
    <b>Real-world analogy:</b> A smoke detector goes off. The p-value is like asking:
    "If there's NO fire, how likely would the detector go off?" If the answer is "very unlikely" (low p-value),
    you investigate. But the alarm doesn't tell you <b>how big</b> the fire is, or whether it's a real fire
    vs burnt toast. You still need to look.
    <br><br>
    <b>Statistical significance ≠ practical significance.</b> A drug that lowers blood pressure by 0.1 mmHg
    might be statistically significant (p = 0.001) with 100,000 patients, but clinically meaningless.
    Always ask: <b>"Is the effect big enough to matter?"</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: A/B Test Simulator")
    st.caption("You're testing a new website design. Does it improve conversion rate?")
    c1, c2 = st.columns(2)
    conv_a = c1.slider("Control group conversion %:", 1.0, 20.0, 10.0, 0.5, key="m4_ca")
    conv_b = c2.slider("Test group conversion %:", 1.0, 20.0, 11.0, 0.5, key="m4_cb")
    n_users = st.slider("Users per group:", 100, 50000, 5000, 500, key="m4_users")

    np.random.seed(42)
    group_a = np.random.binomial(1, conv_a/100, n_users)
    group_b = np.random.binomial(1, conv_b/100, n_users)

    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(group_a, group_b)
    lift = (group_b.mean() - group_a.mean()) / group_a.mean() * 100

    mc = st.columns(4)
    mc[0].metric("Control Rate", f"{group_a.mean()*100:.2f}%")
    mc[1].metric("Test Rate", f"{group_b.mean()*100:.2f}%", f"{lift:+.1f}% lift")
    mc[2].metric("P-Value", f"{p_val:.4f}")
    mc[3].metric("Significant?", "✅ Yes" if p_val < 0.05 else "❌ No")

    if p_val < 0.05:
        st.markdown(f"""<div class="green-box">✅ <b>p = {p_val:.4f}</b> — Statistically significant! There's less than a 5% chance this result is due to random luck. The new design likely works.</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="red-box">❌ <b>p = {p_val:.4f}</b> — Not significant. We can't rule out that this difference is just random noise. Need more data or a bigger effect.</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="green-box">👉 <b>Telecom:</b> "Did the new network configuration reduce dropped calls?" "Is the SLA breach rate different between regions?" — all hypothesis tests.</div>""", unsafe_allow_html=True)

    iq([
        {"q": "Explain p-value to a non-technical stakeholder.", "d": "Medium", "c": ["Meta", "Google", "Airbnb"],
         "a": "A p-value answers: <b>'If there's truly no effect, how likely would we see results this extreme by pure chance?'</b> Example: You flip a coin 100 times, get 60 heads. P-value = probability of 60+ heads with a fair coin. If it's very low (say 2%), you conclude the coin is probably unfair. <b>p < 0.05</b> = 'less than 5% chance this is luck.' <b>Critical nuance:</b> p-value does NOT tell you how big the effect is, or the probability your hypothesis is true.",
         "t": "Always mention what p-value does NOT mean. Interviewers specifically test this."},
        {"q": "What's the difference between Type I and Type II errors?", "d": "Medium", "c": ["Amazon", "Apple", "Google"],
         "a": "<b>Type I (False Positive):</b> You say there's an effect when there isn't. Convicting an innocent person. Launching a feature that doesn't actually work. <b>Type II (False Negative):</b> You miss a real effect. Acquitting a guilty person. Not launching a feature that would have worked. <b>Tradeoff:</b> Reducing one increases the other. <b>Which is worse?</b> Depends on context — in medicine, Type II (missing cancer) is worse. In A/B testing, Type I (launching bad features) is usually worse.",
         "t": "Always give a real-world example and say which error is worse in that context."},
        {"q": "Your A/B test shows p=0.04. Should you launch?", "d": "Hard", "c": ["Meta", "Google", "Netflix"],
         "a": "Not automatically. I'd check: <b>(1) Effect size</b> — is the improvement practically meaningful? (2) <b>Multiple comparisons</b> — did we test many metrics? (3) <b>Sample size</b> — was the test properly powered? (4) <b>Guardrail metrics</b> — did anything important get worse? (5) <b>Novelty effect</b> — is engagement just because it's new? I'd recommend: 'Promising, but let's validate with a longer test before full launch.'",
         "t": "This is a CLASSIC Meta/Google question. Show you don't blindly trust p-values."},
        {"q": "What is the Central Limit Theorem and why does it matter?", "d": "Medium", "c": ["Google", "Meta", "Netflix"],
         "a": "CLT says: sample means from ANY population form a normal distribution as sample size increases. <b>Why it matters:</b> (1) Foundation of A/B testing — we compare sample means assuming normality. (2) Enables confidence intervals even for non-normal data. (3) Works with n ≥ 30 for most distributions. <b>Analogy:</b> Roll a die (uniform). Average of 100 rolls ≈ 3.5 every time. Those averages form a bell curve.",
         "t": "Draw the picture: any shape on the left → bell curve of means on the right."},
        {"q": "What is a confidence interval? How do you explain it to a PM?", "d": "Medium", "c": ["Google", "Amazon"],
         "a": "A 95% confidence interval means: if we repeated this experiment 100 times, about 95 of those intervals would contain the true value. <b>To a PM:</b> 'We estimate the conversion rate is 12%, and we're 95% confident the true rate is between 11.2% and 12.8%.' <b>Common misconception:</b> It does NOT mean 'there's a 95% probability the true value is in this interval.' The true value is fixed — the interval is what varies across experiments.",
         "t": "The misconception is what interviewers test. Nail the correct interpretation."},
        {"q": "What is a Z-score and when would you use it?", "d": "Easy", "c": ["Google", "Amazon", "General"],
         "a": "Z-score = (value - mean) / std dev. It tells you <b>how many standard deviations</b> a value is from the mean. <b>Use cases:</b> (1) Detecting outliers (|z| > 3 = very unusual). (2) Comparing values from different scales (a score of 80 on a hard test vs 90 on an easy test — z-scores make them comparable). (3) Standardizing features before ML models that are distance-sensitive (KNN, SVM).",
         "t": "Give the outlier detection use case — it's the most practical."},
        {"q": "When do you use a t-test vs a z-test?", "d": "Medium", "c": ["Google", "Meta", "Netflix"],
         "a": "<b>Z-test:</b> When you know the population σ AND n > 30. Rare in practice. <b>T-test:</b> When you estimate σ from the sample (almost always). Works for any sample size. <b>Key:</b> As n → large, t-distribution → normal distribution, so t-test and z-test give the same result. <b>Rule:</b> When in doubt, use t-test. It's always safe.",
         "t": "Say 'I use t-test 99% of the time' — shows practical experience."},
        {"q": "Explain the three types of t-tests with examples.", "d": "Medium", "c": ["Amazon", "Apple"],
         "a": "<b>One-sample:</b> Compare a group mean to a known value. 'Is our avg response time different from the 200ms SLA?' <b>Two-sample (independent):</b> Compare two separate groups. 'Do monthly vs annual customers have different churn rates?' <b>Paired:</b> Compare the same group before/after. 'Did the same servers improve after the patch?' <b>Key:</b> Paired tests are more powerful because they control for individual differences.",
         "t": "Always give a real-world example for each type."},
        {"q": "What assumptions does a t-test make?", "d": "Hard", "c": ["Google", "Netflix"],
         "a": "<b>1. Independence:</b> Observations are independent (no repeated measures in unpaired test). <b>2. Normality:</b> Data is approximately normal — but t-test is robust to this with n > 30 (CLT). <b>3. Equal variance (for two-sample):</b> Both groups have similar spread. If not, use Welch's t-test (doesn't assume equal variance). <b>Practical:</b> With large samples, only independence really matters. Always use Welch's t-test as default — it's safe regardless of variance equality.",
         "t": "Mention Welch's t-test — it shows you know the modern best practice."},
    ])


# ═══════════════════════════════════════
# MODULE 5: CORRELATION
# ═══════════════════════════════════════
elif module == "🔗 M5: Correlation":
    st.markdown("# 🔗 Module 5: Correlation & Relationships")
    st.caption("Week 3 · Which factors move together?")

    st.markdown("""<div class="story-box">
    <b>Correlation</b> measures how two variables move together. It's a number from <b>-1 to +1</b>.
    <br>• <b>+1:</b> Perfect positive — both go up together
    <br>• <b>0:</b> No relationship
    <br>• <b>-1:</b> Perfect negative — one goes up, the other goes down
    <br><br>The golden rule: <b style="color:#f45d6d">Correlation does NOT mean causation.</b>
    </div>""", unsafe_allow_html=True)

    # ── Covariance first ──
    st.markdown("### 📖 Covariance — The Building Block")
    st.markdown("""<div class="story-box">
    Before correlation, there's <b>covariance</b>. It answers the same question — "do X and Y move together?" — but in a <b>raw, unscaled</b> way.
    <br><br><b>Cov(X, Y) > 0:</b> When X goes up, Y tends to go up too
    <br><b>Cov(X, Y) < 0:</b> When X goes up, Y tends to go down
    <br><b>Cov(X, Y) ≈ 0:</b> No consistent pattern
    <br><br><b>The problem with covariance:</b> Its value depends on the <b>scale</b> of your data. Cov(height in cm, weight in kg) gives a completely different number than Cov(height in inches, weight in pounds) — even though the relationship is the same. That makes it hard to interpret.
    </div>""", unsafe_allow_html=True)

    # ── Pearson Correlation ──
    st.markdown("### 📖 Pearson Correlation — Covariance, Normalized")
    st.markdown("""<div class="story-box">
    <b>Pearson correlation (r)</b> fixes covariance's scale problem by dividing by the standard deviations:
    <br><br>
    <div style="text-align:center;font-size:1.3rem;padding:0.8rem;background:rgba(124,106,255,0.08);border-radius:10px;margin:0.5rem 0">
    <b>r = Cov(X, Y) / (σ<sub>X</sub> × σ<sub>Y</sub>)</b>
    </div>
    <br>
    This <b>normalizes</b> the covariance to always fall between <b>-1 and +1</b>, regardless of the units or scale of X and Y.
    <br><br><b>In plain English:</b> "Take the covariance (how much X and Y move together) and divide by how much each one varies on its own. The result is a clean -1 to +1 score."
    <br><br><b>Why this matters for feature selection:</b> You can compare correlations across features directly. "Feature A has r=0.8 with the target, Feature B has r=0.3" — A is more useful, regardless of whether A is measured in dollars and B in milliseconds.
    </div>""", unsafe_allow_html=True)

    # ── Interactive demo ──
    st.markdown("### 🎮 Try It: See Covariance vs Correlation")
    st.caption("Drag the slider to change the relationship strength. Notice how correlation stays between -1 and +1 while covariance changes with scale.")

    target_r = st.slider("Target correlation (r):", -1.0, 1.0, 0.7, 0.05, key="m5_r")
    scale_x = st.slider("Scale of X (doesn't change the relationship, only the units):", 1, 100, 1, key="m5_scale")

    np.random.seed(42)
    x = np.random.normal(0, 1, 200) * scale_x
    noise = np.random.normal(0, 1, 200) * scale_x
    y = target_r * x + np.sqrt(max(0.01, 1 - target_r**2)) * noise
    actual_r = np.corrcoef(x, y)[0, 1]
    actual_cov = np.cov(x, y)[0, 1]

    mc = st.columns(3)
    mc[0].metric("Covariance", f"{actual_cov:.1f}", help="Raw measure — changes with scale")
    mc[1].metric("Correlation (r)", f"{actual_r:.3f}", help="Normalized — always between -1 and +1")
    mc[2].metric("Scale of X", f"×{scale_x}", help="Try changing this — covariance changes, correlation doesn't!")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#22d3a7', size=5, opacity=0.6)))
    z = np.polyfit(x, y, 1); x_l = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(x=x_l, y=np.polyval(z, x_l), mode='lines', line=dict(color='#f45d6d', width=2, dash='dash'), name='Trend'))
    fig.update_layout(height=350, title=f"r = {actual_r:.3f} (Cov = {actual_cov:.1f})", **DL)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    abs_r = abs(actual_r)
    strength = "Very strong" if abs_r > 0.8 else "Strong" if abs_r > 0.6 else "Moderate" if abs_r > 0.4 else "Weak" if abs_r > 0.2 else "None"
    st.markdown(f"""<div class="green-box">💡 <b>{strength} {'positive' if actual_r > 0.05 else 'negative' if actual_r < -0.05 else 'no'} correlation.</b> r = {actual_r:.3f}</div>""", unsafe_allow_html=True)

    # ── Correlation ≠ Causation ──
    st.markdown("### ⚠️ Correlation ≠ Causation")
    st.markdown("""<div class="red-box">
    <b>Ice cream sales ↔ drowning deaths</b> (r ≈ 0.8). Does ice cream cause drowning? No — <b>summer heat</b> causes both.
    <br><br>The hidden third variable is called a <b>confounding variable</b>. Always ask: "Could something else be causing both?"
    <br><br><b>Only controlled experiments (A/B tests)</b> can prove causation.
    </div>""", unsafe_allow_html=True)

    # ── Heatmap ──
    st.markdown("### 🗺️ Correlation Heatmap")
    np.random.seed(42); n = 300
    age = np.random.normal(35, 10, n)
    hm = pd.DataFrame({"Age": age, "Income": age*1200+np.random.normal(0,8000,n), "Spending": age*800+np.random.normal(0,5000,n), "Satisfaction": np.random.normal(7,2,n)})
    corr = hm.corr()
    fig_hm = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale=[[0,'#f45d6d'],[0.5,'#1a1d2e'],[1,'#22d3a7']], zmin=-1, zmax=1, text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=12)))
    fig_hm.update_layout(height=380, title="Which variables are related?", **DL)
    st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="green-box">👉 <b>Telecom:</b> "Which KPIs correlate with churn?" "Does network latency correlate with support tickets?" Heatmaps are your first tool for feature selection.</div>""", unsafe_allow_html=True)

    iq([
        {"q": "Explain correlation to a non-technical PM.", "d": "Easy", "c": ["Google", "Meta"],
         "a": "Correlation measures how two things move together, from -1 to +1. Example: 'Customers who call support more are more likely to churn (r=0.65). But support calls don't CAUSE churn — the underlying service problems cause both.'",
         "t": "Always end with 'correlation ≠ causation.'"},
        {"q": "Pearson vs Spearman — when do you use each?", "d": "Medium", "c": ["Google", "Netflix"],
         "a": "<b>Pearson:</b> Linear relationships, continuous data, roughly normal. <b>Spearman:</b> Monotonic (consistently up/down but not necessarily linear), ordinal data, robust to outliers. Use Spearman when data is ranked or has outliers.",
         "t": "Mention that tree-based models don't care about linearity, so Spearman is often more useful for feature selection."},
        {"q": "Two features have r=0.95. Should you keep both in your model?", "d": "Medium", "c": ["Amazon", "Google"],
         "a": "Probably not — they're <b>collinear</b> (carry the same info). Problems: unstable coefficients in linear models, no prediction improvement, potential overfitting. <b>Fix:</b> Keep the one more correlated with the target, or combine them (PCA), or use tree-based models (robust to collinearity).",
         "t": "Mention VIF (Variance Inflation Factor) > 5-10 as a formal collinearity check."},
        {"q": "How would you determine if a correlation implies causation?", "d": "Hard", "c": ["Google", "Apple"],
         "a": "<b>Gold standard:</b> Randomized controlled experiment (A/B test). <b>Alternatives:</b> Natural experiments, instrumental variables, Granger causality (time series), difference-in-differences. <b>Checklist:</b> (1) Temporal ordering (cause before effect), (2) Plausible mechanism, (3) No confounders, (4) Dose-response relationship.",
         "t": "Lead with 'run an A/B test' — that's what FAANG does."},
        {"q": "What is a confounding variable? Give examples.", "d": "Easy", "c": ["Amazon", "Meta"],
         "a": "A hidden variable that influences BOTH variables you're studying. <b>Examples:</b> Shoe size ↔ reading ability (confounder: age). Chocolate consumption ↔ Nobel Prizes (confounder: wealth). Hospital treatment ↔ mortality (confounder: disease severity).",
         "t": "Have 2-3 examples memorized. The shoe size one is universally understood."},
        {"q": "What's the difference between covariance and correlation?", "d": "Medium", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>Covariance:</b> Measures how two variables move together, but in raw units. Cov(height_cm, weight_kg) gives a different number than Cov(height_inches, weight_lbs) — same relationship, different scale. Hard to interpret. <b>Correlation:</b> Covariance normalized by the std devs of both variables. Always between -1 and +1 regardless of units. <b>Formula:</b> r = Cov(X,Y) / (σX × σY). <b>Key:</b> Correlation is what you report. Covariance is the building block behind it.",
         "t": "Say 'correlation is covariance made interpretable' — clean and memorable."},
        {"q": "How do you use correlation for feature selection?", "d": "Medium", "c": ["Amazon", "Netflix", "Google"],
         "a": "<b>Step 1:</b> Compute correlation of every feature with the target variable. Rank by |r|. Features with near-zero correlation are likely useless. <b>Step 2:</b> Check feature-to-feature correlations. If two features have r > 0.9, they're redundant — drop one. <b>Step 3:</b> Use a heatmap to visualize all pairwise correlations at once. <b>Caveat:</b> Pearson only catches linear relationships. Use mutual information or Spearman for non-linear patterns. <b>Also:</b> Low correlation doesn't mean useless — the feature might interact with others.",
         "t": "Mention the caveat about non-linear relationships — shows depth."},
    ])


# ═══════════════════════════════════════
# MODULE 6: REGRESSION INTUITION
# ═══════════════════════════════════════
elif module == "📉 M6: Regression Intuition":
    st.markdown("# 📉 Module 6: Regression Intuition")
    st.caption("Week 4 · Predict output from input — the foundation of ML.")

    st.markdown("""<div class="story-box">
    Regression answers: <b>"Given X, what's the best guess for Y?"</b>
    <br><br>It draws the <b>best-fit line</b> through your data: <b>y = mx + b</b>
    <br>• <b>m (slope):</b> For every 1-unit increase in X, Y changes by m
    <br>• <b>b (intercept):</b> The value of Y when X = 0
    <br><br>This is the simplest ML model — and understanding it deeply makes everything else click.
    </div>""", unsafe_allow_html=True)

    # ── Interactive regression ──
    st.markdown("### 🎮 Try It: Fit a Line Through Data")
    st.caption("Adjust the slope and intercept to fit the line to the data. Then see the 'best fit' answer.")

    np.random.seed(42)
    x_data = np.linspace(0, 10, 30)
    y_true = 3 * x_data + 10 + np.random.normal(0, 5, 30)

    c1, c2 = st.columns(2)
    user_m = c1.slider("Your slope (m):", 0.0, 6.0, 1.0, 0.1, key="m6_m")
    user_b = c2.slider("Your intercept (b):", 0.0, 20.0, 5.0, 0.5, key="m6_b")

    # Best fit
    best_m, best_b = np.polyfit(x_data, y_true, 1)
    user_pred = user_m * x_data + user_b
    best_pred = best_m * x_data + best_b
    user_mse = np.mean((y_true - user_pred)**2)
    best_mse = np.mean((y_true - best_pred)**2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', marker=dict(color='#22d3a7', size=7), name='Data'))
    fig.add_trace(go.Scatter(x=x_data, y=user_pred, mode='lines', line=dict(color='#f5b731', width=2), name=f'Your line (m={user_m}, b={user_b})'))
    fig.add_trace(go.Scatter(x=x_data, y=best_pred, mode='lines', line=dict(color='#7c6aff', width=2, dash='dash'), name=f'Best fit (m={best_m:.1f}, b={best_b:.1f})'))
    fig.update_layout(height=380, title="Fit the Line!", xaxis_title="X", yaxis_title="Y", **DL)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    mc = st.columns(3)
    mc[0].metric("Your Error (MSE)", f"{user_mse:.1f}", help="Mean Squared Error — lower is better")
    mc[1].metric("Best Fit Error", f"{best_mse:.1f}")
    mc[2].metric("How close?", f"{user_mse/best_mse:.1f}x" if best_mse > 0 else "∞", help="1.0x = perfect, higher = worse")

    if user_mse < best_mse * 1.2:
        st.markdown("""<div class="green-box">🎯 <b>Great fit!</b> Your line is very close to the optimal one.</div>""", unsafe_allow_html=True)

    # ── Residuals ──
    st.markdown("### 📖 Residuals — How Wrong Is Each Prediction?")
    st.markdown("""<div class="story-box">
    A <b>residual</b> = actual value - predicted value. It's the <b>error</b> for each data point.
    <br><br>Good model → residuals are small and random (no pattern).
    <br>Bad model → residuals show a pattern (the model is missing something).
    </div>""", unsafe_allow_html=True)

    residuals = y_true - best_pred
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=x_data, y=residuals, mode='markers', marker=dict(color='#f5b731', size=7)))
    fig_r.add_hline(y=0, line_dash="dash", line_color="#7c6aff")
    fig_r.update_layout(height=250, title="Residuals (should be random, centered at 0)", xaxis_title="X", yaxis_title="Residual", **DL)
    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

    # ── Overfitting ──
    st.markdown("### 📖 Overfitting vs Underfitting")
    st.markdown("""<div class="story-box">
    <b>Underfitting:</b> Model is too simple — misses the real pattern. Like studying only chapter titles.
    <br><b>Overfitting:</b> Model is too complex — memorizes noise. Like memorizing every word including typos.
    <br><b>Just right:</b> Captures the real pattern without chasing noise.
    </div>""", unsafe_allow_html=True)

    fit_type = st.radio("Model complexity:", ["📉 Underfit (flat line)", "✅ Good fit (straight line)", "⚠️ Overfit (wiggly line)"], horizontal=True, key="m6_fit")

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', marker=dict(color='#22d3a7', size=7), name='Data'))
    x_s = np.linspace(0, 10, 200)
    if "Underfit" in fit_type:
        fig_f.add_trace(go.Scatter(x=x_s, y=np.full_like(x_s, y_true.mean()), mode='lines', line=dict(color='#f5b731', width=3), name='Model'))
    elif "Good" in fit_type:
        fig_f.add_trace(go.Scatter(x=x_s, y=np.polyval(np.polyfit(x_data, y_true, 1), x_s), mode='lines', line=dict(color='#7c6aff', width=3), name='Model'))
    else:
        fig_f.add_trace(go.Scatter(x=x_s, y=np.polyval(np.polyfit(x_data, y_true, 15), x_s).clip(-10, 60), mode='lines', line=dict(color='#f45d6d', width=3), name='Model'))
    fig_f.update_layout(height=320, title="Underfitting → Good Fit → Overfitting", **DL)
    st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="green-box">👉 <b>Telecom:</b> "Predict monthly revenue from subscriber count" — that's linear regression. "Predict churn probability from tenure + charges" — that's logistic regression (next phase). Regression is the gateway to ML.</div>""", unsafe_allow_html=True)

    iq([
        {"q": "Explain linear regression to a non-technical person.", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "Linear regression draws the <b>best straight line</b> through data to make predictions. Example: 'For every extra square foot, house price goes up by $150.' The line captures the relationship: y = 150×sqft + 50,000. It's the simplest prediction model — and often surprisingly effective.",
         "t": "Use the house price example — everyone understands it."},
        {"q": "What are residuals and why do they matter?", "d": "Medium", "c": ["Google", "Meta"],
         "a": "Residual = actual - predicted. They tell you <b>how wrong</b> each prediction is. <b>Why they matter:</b> (1) If residuals show a pattern (e.g., curve), your model is missing something — try adding features or a non-linear model. (2) If residuals are random and centered at 0, your model captured the real pattern. (3) Outliers in residuals = data points your model struggles with.",
         "t": "Always mention 'check residual plots' — it shows you know how to diagnose models."},
        {"q": "What is overfitting? How do you detect and prevent it?", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "<b>Overfitting:</b> Model performs great on training data but poorly on new data. It memorized noise instead of learning patterns. <b>Detection:</b> Big gap between training accuracy and test accuracy. <b>Prevention:</b> (1) Train/test split or cross-validation, (2) Regularization (L1/L2), (3) Simpler model, (4) More training data, (5) Early stopping, (6) Dropout (neural nets).",
         "t": "Use the student analogy: memorizing answers vs understanding concepts."},
        {"q": "What's the difference between R² and MSE?", "d": "Medium", "c": ["Amazon", "Netflix"],
         "a": "<b>MSE (Mean Squared Error):</b> Average of squared residuals. Lower = better. In the units of your target squared (hard to interpret). <b>R² (R-squared):</b> Proportion of variance explained by the model. Ranges 0–1. R²=0.8 means the model explains 80% of the variation. <b>Key:</b> R² is relative (good for comparing models), MSE is absolute (good for understanding error magnitude).",
         "t": "Mention that R² can be misleading — a model can have high R² but still make bad predictions if the data has low variance."},
        {"q": "What assumptions does linear regression make?", "d": "Hard", "c": ["Google", "Apple", "Netflix"],
         "a": "<b>1. Linearity:</b> Relationship between X and Y is linear. <b>2. Independence:</b> Observations are independent. <b>3. Homoscedasticity:</b> Residuals have constant variance (no fan shape). <b>4. Normality of residuals:</b> Residuals are normally distributed (for inference, not prediction). <b>5. No multicollinearity:</b> Features aren't highly correlated with each other. <b>Practical note:</b> Mild violations are usually fine. Check residual plots to verify.",
         "t": "Don't just list them — say which ones matter most in practice (linearity and homoscedasticity) and which are less critical (normality with large samples)."},
        {"q": "When would you use linear regression vs a more complex model?", "d": "Medium", "c": ["Amazon", "Google"],
         "a": "<b>Use linear regression when:</b> (1) Relationship is roughly linear, (2) Interpretability matters (coefficients tell a story), (3) Small dataset, (4) Baseline model. <b>Use complex models when:</b> (1) Non-linear relationships, (2) Many features with interactions, (3) Large dataset, (4) Prediction accuracy matters more than interpretability. <b>My approach:</b> Always start with linear regression as a baseline. If it's not good enough, try tree-based models.",
         "t": "Saying 'I always start with a simple baseline' shows maturity and practical experience."},
    ])
