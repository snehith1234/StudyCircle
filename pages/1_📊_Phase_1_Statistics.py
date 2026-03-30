# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="📊 Phase 1: Statistics", page_icon="📊", layout="wide")

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
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'), title_font=dict(color='#8892b0')),
    yaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'), title_font=dict(color='#8892b0')),
    font=dict(color='#e2e8f0'), legend=dict(font=dict(color='#8892b0')),
    margin=dict(t=40, b=40, l=40, r=40),
)
DIFF_C = {"Easy": "#22d3a7", "Medium": "#f5b731", "Hard": "#f45d6d"}

@st.cache_data
def load_pizza_data():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "store_id": [f"Store_{i+1}" for i in range(n)],
        "daily_sales": np.concatenate([[1200, 80], np.random.normal(500, 100, n-2)]).clip(80, 1200).round(2),
        "avg_order_value": np.random.normal(22, 5, n).clip(10, 40).round(2),
        "customer_rating": np.random.uniform(3.5, 5.0, n).round(2),
        "delivery_time_min": np.concatenate([np.random.normal(30, 8, n-5), [55, 60, 65, 70, 75]]).round(1),
        "employees": np.random.randint(3, 12, n),
        "ad_spend": np.random.normal(200, 50, n).clip(50, 400).round(2),
        "region": np.random.choice(["East", "West"], n, p=[0.35, 0.65]),
    })

df = load_pizza_data()

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
    st.markdown("## 📊 Phase 1: Statistics")
    st.caption("Concept | Code + Output")
    st.divider()
    module = st.radio("Module:", [
        "🏠 Overview",
        "🧩 M1: Descriptive Statistics",
        "🎲 M2: Probability",
        "📈 M3: Distributions",
        "🧪 M4: Inferential Statistics",
        "🔗 M5: Correlation",
        "📉 M6: Regression",
    ], label_visibility="collapsed")


# ═══════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════
if module == "🏠 Overview":
    st.markdown("# 📊 Phase 1: Statistics Foundation")
    st.caption("Concept on Left · Code + Output on Right")

    st.markdown("""<div class="concept-card">
    <b>🎬 Welcome to Statistics!</b> You just got hired at PizzaChain with 50 stores.
    Your boss asks: "Are we doing well? Which stores need help?"
    <br><br>You need <b>statistics</b> — the language that lets you summarize data and find patterns.
    </div>""", unsafe_allow_html=True)

    modules = [
        ("🧩", "Descriptive Statistics", "Week 1", "#7c6aff", "Summarize data: mean, median, spread, outliers"),
        ("🎲", "Probability", "Week 1–2", "#22d3a7", "How likely is something? Bayes' theorem"),
        ("📈", "Distributions", "Week 2", "#f5b731", "What shape is your data? Normal, Poisson, Binomial"),
        ("🧪", "Inferential Statistics", "Week 3", "#f45d6d", "Is this result real? Z-scores, T-tests, P-values"),
        ("🔗", "Correlation", "Week 3", "#e879a8", "Do two things move together? Pearson correlation"),
        ("📉", "Regression", "Week 4", "#5eaeff", "Predict values with a line. The bridge to ML."),
    ]

    for icon, title, week, color, desc in modules:
        st.markdown(f"""<div class="concept-card" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span>
        <span class="iq-tag" style="background:{color}22;color:{color}">{week}</span>
        <b style="margin-left:6px">{title}</b>
        <br><span style="color:#8892b0">{desc}</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# M1: DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════
elif module == "🧩 M1: Descriptive Statistics":
    st.markdown("# 🧩 Module 1: Descriptive Statistics")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Descriptive Statistics?</b> It's the art of <b>summarizing data</b> so you can understand it at a glance.
    Instead of looking at 1000 numbers, you get a few key metrics that tell the story.
    <br><br>🎯 <b>Three key questions:</b> What's typical (center)? How spread out (variability)? Any weird values (outliers)?
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Load Data
    def show_dataset():
        st.dataframe(df.head(8), use_container_width=True, height=180)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    split_row(
        concept_html="""<div class="concept-card">
        <b>🎬 The Story:</b> You manage 50 pizza stores. Your boss asks "How are we doing?"
        <br><br>Descriptive stats answers three questions:
        <br>🔹 <b>Center:</b> What's typical? (Mean, Median)
        <br>🔹 <b>Spread:</b> How different are values? (Std Dev, IQR)
        <br>🔹 <b>Outliers:</b> Any weird values?
        </div>
        <div class="insight-box">💡 <code>df.head()</code> and <code>df.describe()</code> are your first steps with any dataset.</div>""",
        code_str='''import pandas as pd

# Load data
df = pd.read_csv("pizza_stores.csv")

# Quick look
df.head()
df.shape
df.describe()''',
        output_func=show_dataset,
        concept_title="📋 Load & Explore Data",
        output_title="Pizza Store Dataset"
    )

    # Row 2: Mean, Median, Mode
    def show_center():
        col = "daily_sales"
        data = df[col]
        mean_val, median_val = data.mean(), data.median()
        
        mc = st.columns(3)
        mc[0].metric("Mean", f"${mean_val:.0f}")
        mc[1].metric("Median", f"${median_val:.0f}")
        mc[2].metric("Gap", f"${mean_val - median_val:.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=20, marker_color="#7c6aff", opacity=0.7))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: ${mean_val:.0f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="#22d3a7", annotation_text=f"Median: ${median_val:.0f}")
        fig.update_layout(height=220, title="Daily Sales Distribution", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why do we need "center" measures?</b>
        <br><br>Imagine your boss asks: "How much does a typical store make?" You can't list all 50 numbers. You need ONE number that represents the whole group.
        <br><br><b>Mean (Average):</b> Add everything up, divide by count.
        <br>• <b>Pros:</b> Uses all data points
        <br>• <b>Cons:</b> One extreme value can drag it way up or down
        <br>• <b>Analogy:</b> If Bill Gates walks into a bar, the "average" wealth skyrockets — but nobody actually got richer!
        <br><br><b>Median (Middle):</b> Sort all values, pick the one in the middle.
        <br>• <b>Pros:</b> Ignores extremes — robust to outliers
        <br>• <b>Cons:</b> Doesn't use all data
        <br>• <b>Analogy:</b> In a race, the median runner is literally in the middle of the pack
        <br><br><b>Mode:</b> The most common value. Best for categories (e.g., "most popular pizza topping").
        </div>
        <div class="math-box">
        <b>📐 Mean — Step by Step:</b>
        <br><br><b>Data:</b> Sales = [400, 450, 500, 550, 1200]
        <br><br><b>Mean:</b> (400+450+500+550+1200) / 5
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= 3100 / 5 = <b>$620</b>
        <br><br><b>Median:</b> Sort → [400, 450, <b>500</b>, 550, 1200]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Middle value = <b>$500</b>
        <br><br>🧠 Mean ($620) > Median ($500) → right-skewed!
        <br>&nbsp;&nbsp;&nbsp;&nbsp;The $1200 outlier pulls mean up by $120!
        </div>
        <div class="warn-box">⚠️ <b>Interview tip:</b> "When would you use median over mean?" → When data has outliers or is skewed (income, house prices, response times).</div>""",
        code_str='''# Calculate center measures
mean_val = df["daily_sales"].mean()
median_val = df["daily_sales"].median()
mode_val = df["daily_sales"].mode()[0]

print(f"Mean: ${mean_val:.0f}")
print(f"Median: ${median_val:.0f}")
print(f"Gap: ${mean_val - median_val:.0f}")''',
        output_func=show_center,
        concept_title="📊 Mean, Median, Mode",
        output_title="Center Measures"
    )

    # Row 3: Standard Deviation
    def show_spread():
        col = "daily_sales"
        data = df[col]
        std_val, var_val = data.std(), data.var()
        iqr_val = data.quantile(0.75) - data.quantile(0.25)
        
        mc = st.columns(3)
        mc[0].metric("Std Dev", f"${std_val:.0f}")
        mc[1].metric("Variance", f"${var_val:,.0f}")
        mc[2].metric("IQR", f"${iqr_val:.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Box(x=data, marker_color="#22d3a7", boxmean=True, name="Daily Sales"))
        fig.update_layout(height=120, **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why do we need "spread" measures?</b>
        <br><br>Two classes can have the same average grade (75%), but one class might have everyone at 70-80%, while another has scores from 30% to 100%. The <b>spread</b> tells you how consistent or variable the data is.
        <br><br><b>Standard Deviation (σ):</b> "On average, how far is each value from the mean?"
        <br>• <b>Small σ:</b> Values are tightly clustered (consistent)
        <br>• <b>Large σ:</b> Values are spread out (variable)
        <br>• <b>Analogy:</b> A reliable employee (low σ) arrives at 9:00 ± 5 min. An unreliable one (high σ) arrives at 9:00 ± 45 min.
        <br><br><b>Variance:</b> Just σ² (squared). Mathematically useful but harder to interpret.
        <br><br><b>IQR (Interquartile Range):</b> The range of the middle 50% of data.
        <br>• Q1 = 25th percentile, Q3 = 75th percentile
        <br>• IQR = Q3 - Q1
        <br>• <b>Why use it?</b> Like median, it ignores outliers!
        </div>
        <div class="math-box">
        <b>📐 Std Dev — Step by Step:</b>
        <br><br><b>Data:</b> [400, 500, 600], Mean = 500
        <br><br><b>Step 1:</b> Deviations from mean
        <br>&nbsp;&nbsp;&nbsp;&nbsp;400-500 = -100, 500-500 = 0, 600-500 = 100
        <br><br><b>Step 2:</b> Square them (removes negatives)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;(-100)² + 0² + 100² = 10000 + 0 + 10000 = 20000
        <br><br><b>Step 3:</b> Average & sqrt
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Variance = 20000/3 = 6667
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Std Dev = √6667 = <b>$82</b>
        <br><br>🧠 "A typical store is about $82 away from the $500 average"
        </div>""",
        code_str='''# Calculate spread measures
std_dev = df["daily_sales"].std()
variance = df["daily_sales"].var()

Q1 = df["daily_sales"].quantile(0.25)
Q3 = df["daily_sales"].quantile(0.75)
IQR = Q3 - Q1

print(f"Std Dev: ${std_dev:.0f}")
print(f"IQR: ${IQR:.0f}")''',
        output_func=show_spread,
        concept_title="📏 Spread: Std Dev & IQR",
        output_title="Variability Measures"
    )

    # Row 4: Outliers
    def show_outliers():
        col = "daily_sales"
        data = df[col]
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = df[(data < lower) | (data > upper)]
        
        mc = st.columns(4)
        mc[0].metric("Q1", f"${Q1:.0f}")
        mc[1].metric("Q3", f"${Q3:.0f}")
        mc[2].metric("Upper Fence", f"${upper:.0f}")
        mc[3].metric("Outliers", len(outliers))
        
        st.dataframe(outliers[["store_id", col]], use_container_width=True, height=100)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What are outliers and why do they matter?</b>
        <br><br>An outlier is a data point that's <b>far away from the rest</b>. It could be:
        <br>• <b>A data entry error:</b> Someone typed $12000 instead of $1200
        <br>• <b>A measurement error:</b> Faulty sensor reading
        <br>• <b>A genuine extreme:</b> Your flagship store really does make 3× more
        <br>• <b>Fraud:</b> Suspicious transaction that's actually the signal you're looking for!
        <br><br><b>The key question:</b> WHY is this value so different?
        <br>• If it's an error → fix or remove it
        <br>• If it's real but rare → study it separately
        <br>• If it's fraud → that's your finding!
        <br><br><b>IQR Method:</b> The most common way to detect outliers.
        <br>• Calculate Q1 (25th percentile) and Q3 (75th percentile)
        <br>• IQR = Q3 - Q1 (the middle 50% range)
        <br>• Anything below Q1 - 1.5×IQR or above Q3 + 1.5×IQR is an outlier
        <br>• <b>Why 1.5?</b> It's a convention that works well for normal-ish data
        </div>
        <div class="math-box">
        <b>📐 IQR Outlier Rule — Step by Step:</b>
        <br><br><b>Given:</b> Q1 = $440, Q3 = $560
        <br><br><b>Step 1:</b> Calculate IQR
        <br>&nbsp;&nbsp;&nbsp;&nbsp;IQR = Q3 - Q1 = 560 - 440 = <b>$120</b>
        <br><br><b>Step 2:</b> Calculate fences
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Lower = 440 - 1.5×120 = 440 - 180 = <b>$260</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Upper = 560 + 1.5×120 = 560 + 180 = <b>$740</b>
        <br><br><b>Step 3:</b> Flag outliers
        <br>&nbsp;&nbsp;&nbsp;&nbsp;$1200 > $740 → <b>Outlier!</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;$80 < $260 → <b>Outlier!</b>
        </div>
        <div class="warn-box">⚠️ <b>Never auto-delete outliers!</b> Always investigate first. In fraud detection, outliers ARE the signal.</div>""",
        code_str='''# IQR method for outlier detection
Q1 = df["daily_sales"].quantile(0.25)
Q3 = df["daily_sales"].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# Find outliers
outliers = df[
    (df["daily_sales"] < lower_fence) | 
    (df["daily_sales"] > upper_fence)
]
print(f"Found {len(outliers)} outliers")''',
        output_func=show_outliers,
        concept_title="🔔 Outlier Detection",
        output_title="IQR Method Results"
    )

    iq([
        {"q": "Explain mean, median, and mode. When would you use each?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Mean:</b> Use when data is symmetric. <b>Median:</b> Use when data is skewed or has outliers. <b>Mode:</b> Use for categorical data.",
         "t": "Always mention outlier sensitivity of the mean."},
    ])


# ═══════════════════════════════════════
# M2: PROBABILITY
# ═══════════════════════════════════════
elif module == "🎲 M2: Probability":
    st.markdown("# 🎲 Module 2: Probability")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Probability?</b> It's the math of <b>uncertainty</b> — quantifying how likely something is to happen.
    From "Will it rain?" to "Will this customer churn?" — probability gives you a number between 0 and 1.
    <br><br>🎯 <b>Key concepts:</b> Basic probability, conditional probability (given X, what's P(Y)?), and Bayes' theorem (flipping conditionals).
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Generate order data
    np.random.seed(42)
    n_orders = 2000
    orders = pd.DataFrame({
        "order_id": range(1, n_orders+1),
        "delivery_time": np.random.normal(35, 12, n_orders).clip(15, 80),
        "complained": np.random.choice([0, 1], n_orders, p=[0.85, 0.15]),
        "churned": np.random.choice([0, 1], n_orders, p=[0.87, 0.13]),
    })
    orders.loc[orders["complained"] == 1, "churned"] = np.random.choice([0, 1], orders["complained"].sum(), p=[0.6, 0.4])

    # Row 1: Basic Probability
    def show_basic_prob():
        late_threshold = 45
        late_count = (orders["delivery_time"] > late_threshold).sum()
        total = len(orders)
        prob = late_count / total
        
        mc = st.columns(3)
        mc[0].metric("Total Orders", f"{total:,}")
        mc[1].metric("Late (>45min)", late_count)
        mc[2].metric("P(Late)", f"{prob:.1%}")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=orders["delivery_time"], nbinsx=30, marker_color="#7c6aff", opacity=0.7))
        fig.add_vline(x=late_threshold, line_dash="dash", line_color="#f45d6d", annotation_text="Late: 45min")
        fig.update_layout(height=200, title="Delivery Time Distribution", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is probability?</b>
        <br><br>Probability is a number between 0 and 1 that tells you <b>how likely something is to happen</b>.
        <br>• P = 0 means "impossible" (sun rising in the west)
        <br>• P = 1 means "certain" (sun rising tomorrow)
        <br>• P = 0.5 means "50-50 chance" (fair coin flip)
        <br><br><b>The formula is simple:</b>
        <br>P(event) = (# of ways event can happen) / (total # of possible outcomes)
        <br><br><b>Real-world example:</b> You're a delivery manager. Out of 2000 orders, 495 arrived late (>45 min). What's the probability a random order is late?
        <br><br><b>Why it matters:</b> If P(late) = 25%, and you have 100 orders today, expect ~25 complaints. Now you can staff accordingly!
        </div>
        <div class="math-box">
        <b>📐 Basic Probability — Step by Step:</b>
        <br><br><b>Given:</b> 2000 total orders, 495 were late
        <br><br><b>Calculate P(late):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(late) = late orders / total orders
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(late) = 495 / 2000 = <b>0.2475 = 24.75%</b>
        <br><br>🧠 <b>Interpretation:</b> "If you pick a random order, there's about a 1-in-4 chance it was late."
        </div>""",
        code_str='''# Basic probability calculation
late_threshold = 45  # minutes

late_orders = orders[orders["delivery_time"] > late_threshold]
late_count = len(late_orders)
total = len(orders)

p_late = late_count / total
print(f"P(late) = {late_count}/{total} = {p_late:.1%}")''',
        output_func=show_basic_prob,
        concept_title="📊 Basic Probability",
        output_title="P(Late Delivery)"
    )

    # Row 2: Conditional Probability
    def show_conditional():
        complainers = orders[orders["complained"] == 1]
        non_complainers = orders[orders["complained"] == 0]
        
        p_churn_complained = complainers["churned"].mean()
        p_churn_no_complaint = non_complainers["churned"].mean()
        
        mc = st.columns(2)
        mc[0].metric("P(Churn | Complained)", f"{p_churn_complained:.1%}")
        mc[1].metric("P(Churn | No Complaint)", f"{p_churn_no_complaint:.1%}")
        
        fig = go.Figure(data=[
            go.Bar(name='Complained', x=['Churn Rate'], y=[p_churn_complained], marker_color='#f45d6d'),
            go.Bar(name='No Complaint', x=['Churn Rate'], y=[p_churn_no_complaint], marker_color='#22d3a7')
        ])
        fig.update_layout(height=200, barmode='group', **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is conditional probability?</b>
        <br><br>Sometimes you want to know the probability of something <b>given that something else already happened</b>. The "|" symbol means "given that."
        <br><br><b>P(A|B)</b> = "Probability of A, given that B happened"
        <br><br><b>The key insight:</b> You're <b>filtering your universe</b>. Instead of looking at ALL customers, you only look at customers who complained.
        <br><br><b>Real-world example:</b>
        <br>• Overall churn rate: 13% of all customers leave
        <br>• But what about customers who complained? 40% of them leave!
        <br>• This tells you: complaints are a <b>warning signal</b> for churn
        <br><br><b>Why it matters:</b> If you can identify high-risk groups (complainers), you can intervene before they leave!
        </div>
        <div class="math-box">
        <b>📐 Conditional Probability — Step by Step:</b>
        <br><br><b>Question:</b> What's P(churn | complained)?
        <br><br><b>Step 1:</b> Filter to only complainers
        <br>&nbsp;&nbsp;&nbsp;&nbsp;300 customers complained
        <br><br><b>Step 2:</b> Count how many churned
        <br>&nbsp;&nbsp;&nbsp;&nbsp;120 of those 300 churned
        <br><br><b>Step 3:</b> Calculate
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(churn | complained) = 120/300 = <b>40%</b>
        <br><br><b>Compare to baseline:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(churn | no complaint) = 136/1700 = <b>8%</b>
        <br><br>🧠 Complainers churn at <b>5× the rate</b>! This is actionable insight.
        </div>
        <div class="warn-box">⚠️ <b>Business insight:</b> Prioritize reaching out to customers who complained — they're 5× more likely to leave!</div>""",
        code_str='''# Conditional probability: P(A|B)
complainers = orders[orders["complained"] == 1]
non_complainers = orders[orders["complained"] == 0]

p_churn_given_complained = complainers["churned"].mean()
p_churn_given_no_complaint = non_complainers["churned"].mean()

print(f"P(churn | complained) = {p_churn_given_complained:.1%}")
print(f"P(churn | no complaint) = {p_churn_given_no_complaint:.1%}")''',
        output_func=show_conditional,
        concept_title="🔀 Conditional Probability",
        output_title="P(Churn | Complained)"
    )

    # Row 3: Bayes' Theorem
    def show_bayes():
        p_complained = orders["complained"].mean()
        p_churned = orders["churned"].mean()
        p_churn_given_complained = orders[orders["complained"] == 1]["churned"].mean()
        p_complained_given_churned = (p_churn_given_complained * p_complained) / p_churned
        
        mc = st.columns(4)
        mc[0].metric("P(Complained)", f"{p_complained:.1%}")
        mc[1].metric("P(Churned)", f"{p_churned:.1%}")
        mc[2].metric("P(Churn|Complained)", f"{p_churn_given_complained:.1%}")
        mc[3].metric("P(Complained|Churned)", f"{p_complained_given_churned:.1%}")

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Bayes' Theorem and why is it powerful?</b>
        <br><br>Bayes' Theorem lets you <b>flip a conditional probability</b>. You know P(B|A), but you need P(A|B).
        <br><br><b>Real-world example:</b>
        <br>• You know: "40% of complainers churn" — P(churn | complained)
        <br>• You want: "What % of churned customers had complained?" — P(complained | churned)
        <br><br><b>Why does this matter?</b>
        <br>• If 46% of churned customers complained first, complaints are a <b>leading indicator</b>
        <br>• You can build an early warning system!
        <br><br><b>The formula:</b> P(A|B) = P(B|A) × P(A) / P(B)
        <br><br><b>Analogy:</b> A medical test is 99% accurate. You test positive. What's the chance you're actually sick? Bayes tells you — and the answer might surprise you (it depends on how rare the disease is)!
        </div>
        <div class="math-box">
        <b>📐 Bayes' Theorem — Step by Step:</b>
        <br><br><b>Question:</b> What % of churned customers had complained?
        <br><br><b>Given:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(complained) = 15% (overall complaint rate)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(churned) = 13% (overall churn rate)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(churn | complained) = 40%
        <br><br><b>Apply Bayes:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(complained | churned) = P(churn|complained) × P(complained) / P(churned)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= (0.40 × 0.15) / 0.13
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= 0.06 / 0.13 = <b>46%</b>
        <br><br>🧠 <b>Insight:</b> Nearly half of churned customers complained first! Complaints are a strong warning signal.
        </div>""",
        code_str='''# Bayes' Theorem: P(A|B) = P(B|A) × P(A) / P(B)
p_complained = orders["complained"].mean()
p_churned = orders["churned"].mean()
p_churn_given_complained = orders[orders["complained"] == 1]["churned"].mean()

# Apply Bayes
p_complained_given_churned = (
    p_churn_given_complained * p_complained / p_churned
)
print(f"P(complained | churned) = {p_complained_given_churned:.1%}")''',
        output_func=show_bayes,
        concept_title="🔄 Bayes' Theorem",
        output_title="Flipping Conditionals"
    )


# ═══════════════════════════════════════
# M3: DISTRIBUTIONS
# ═══════════════════════════════════════
elif module == "📈 M3: Distributions":
    st.markdown("# 📈 Module 3: Distributions")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What are Distributions?</b> They describe the <b>shape of your data</b> — how values are spread across a range.
    Is it bell-shaped? Skewed? Understanding distributions helps you pick the right statistical tests and make better predictions.
    <br><br>🎯 <b>Key distributions:</b> Normal (bell curve), Binomial (success/failure trials), Poisson (event counts).
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Normal Distribution
    def show_normal():
        from scipy.stats import norm
        data = df["avg_order_value"]
        mean_val, std_val = data.mean(), data.std()
        
        mc = st.columns(3)
        mc[0].metric("Mean (μ)", f"${mean_val:.2f}")
        mc[1].metric("Std Dev (σ)", f"${std_val:.2f}")
        mc[2].metric("68% Range", f"${mean_val-std_val:.0f}-${mean_val+std_val:.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=20, marker_color="#7c6aff", opacity=0.7, name="Data"))
        x_range = np.linspace(data.min(), data.max(), 100)
        y_norm = norm.pdf(x_range, mean_val, std_val) * len(data) * (data.max() - data.min()) / 20
        fig.add_trace(go.Scatter(x=x_range, y=y_norm, mode='lines', line=dict(color='#f5b731', width=2, dash='dash'), name='Normal'))
        fig.update_layout(height=220, title="Order Values vs Normal Curve", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is the Normal Distribution?</b>
        <br><br>The normal distribution (bell curve) is <b>nature's favorite shape</b>. When you measure almost anything in large quantities — heights, test scores, manufacturing errors — you get this symmetric, bell-shaped pattern.
        <br><br><b>Why does it happen?</b> When many small, independent factors add up, the result tends toward normal. Your height = genetics + nutrition + sleep + exercise + ... = bell curve!
        <br><br><b>The two magic numbers:</b>
        <br>• <b>μ (mean):</b> Where the peak is — the "center" of the bell
        <br>• <b>σ (std dev):</b> How wide the bell is — the "spread"
        <br><br><b>Why it matters:</b> If you know data is normal, you can make powerful predictions. "68% of customers spend between $17-$27" — that's actionable!
        <br><br><b>The 68-95-99.7 Rule:</b> This is the most useful thing about normal distributions:
        <br>• 68% of data falls within 1σ of the mean
        <br>• 95% falls within 2σ
        <br>• 99.7% falls within 3σ
        <br>• Beyond 3σ? That's a 0.3% event — investigate it!
        </div>
        <div class="math-box">
        <b>📐 68-95-99.7 Rule — Step by Step:</b>
        <br><br><b>Given:</b> μ = $22, σ = $5
        <br><br><b>68% range (μ ± 1σ):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;$22 - $5 to $22 + $5 = <b>$17 to $27</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ "Most customers spend $17-$27"
        <br><br><b>95% range (μ ± 2σ):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;$22 - $10 to $22 + $10 = <b>$12 to $32</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ "Almost all customers spend $12-$32"
        <br><br><b>99.7% range (μ ± 3σ):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;$22 - $15 to $22 + $15 = <b>$7 to $37</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ "Virtually everyone is in this range"
        <br><br>🧠 Order > $37 is a 0.15% event — either a big spender or a data error!
        </div>
        <div class="insight-box">💡 <b>Real-world use:</b> Quality control uses 3σ limits. If a measurement is beyond 3σ, something is wrong with the process!</div>""",
        code_str='''from scipy.stats import norm

data = df["avg_order_value"]
mean = data.mean()
std = data.std()

# 68-95-99.7 rule
print(f"68% of data: {mean-std:.0f} to {mean+std:.0f}")
print(f"95% of data: {mean-2*std:.0f} to {mean+2*std:.0f}")''',
        output_func=show_normal,
        concept_title="🔔 Normal Distribution",
        output_title="Order Values"
    )

    # Row 2: Binomial Distribution
    def show_binomial():
        from scipy.stats import binom
        n, p = 100, 0.10
        x = np.arange(0, 25)
        pmf = binom.pmf(x, n, p)
        expected = n * p
        
        mc = st.columns(3)
        mc[0].metric("Expected", f"{expected:.0f} conversions")
        mc[1].metric("Std Dev", f"{np.sqrt(n*p*(1-p)):.1f}")
        mc[2].metric("P(exactly 10)", f"{binom.pmf(10, n, p):.1%}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=pmf, marker_color="#22d3a7", opacity=0.7))
        fig.add_vline(x=expected, line_dash="dash", line_color="#f45d6d", annotation_text=f"Expected: {expected}")
        fig.update_layout(height=220, title="Binomial: 100 emails, 10% conversion", xaxis_title="Number of Conversions (k)", yaxis_title="Probability P(X=k)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        # Graph interpretation
        st.markdown("""<div class="insight-box">
        <b>📊 Reading the Graph:</b>
        <br>• <b>X-axis:</b> Number of conversions (0, 1, 2, ... 24)
        <br>• <b>Y-axis:</b> Probability of getting exactly that many conversions
        <br>• <b>Bar height:</b> Taller bar = more likely outcome
        <br>• <b>Peak at 10:</b> Most likely outcome (but only ~13% chance!)
        <br>• <b>Bell shape:</b> Values near expected (10) are common; extremes (0 or 20+) are rare
        <br>• <b>Red line:</b> Expected value (mean) = n × p = 10
        </div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is the Binomial Distribution?</b>
        <br><br>The binomial distribution answers: <b>"If I flip a coin n times, how many heads will I get?"</b> More generally, it counts successes in a fixed number of yes/no trials.
        <br><br><b>When to use it:</b>
        <br>• Fixed number of trials (n) — "I'm sending exactly 100 emails"
        <br>• Each trial is independent — one email's result doesn't affect others
        <br>• Same probability each time (p) — 10% conversion rate for all
        <br>• Only two outcomes — convert or don't convert
        <br><br><b>Real-world examples:</b>
        <br>• Email campaigns: How many of 100 emails will convert?
        <br>• Quality control: How many of 50 products are defective?
        <br>• A/B testing: How many of 1000 visitors will click?
        <br><br><b>The key insight:</b> Even with a 10% conversion rate, you won't always get exactly 10 conversions from 100 emails. Sometimes 7, sometimes 13. The binomial tells you the probability of each outcome.
        </div>
        <div class="math-box">
        <b>📐 How to Read the Binomial Graph:</b>
        <br><br><b>Each bar answers:</b> "What's the probability of getting exactly k successes?"
        <br><br><b>Example from graph:</b>
        <br>• P(X = 10) ≈ 13% → tallest bar, most likely
        <br>• P(X = 5) ≈ 2% → short bar, unlikely
        <br>• P(X = 20) ≈ 0.01% → almost invisible, very rare
        <br><br><b>The shape tells you:</b>
        <br>• <b>Center:</b> Where the peak is (expected value = n×p)
        <br>• <b>Spread:</b> How wide the "hump" is (std dev = √(np(1-p)))
        <br>• <b>Skew:</b> If p is small, skewed right; if p is large, skewed left
        <br><br><b>Sum of all bars = 1</b> (100% — something must happen!)
        </div>
        <div class="warn-box">⚠️ <b>Key insight:</b> The tallest bar (most likely outcome) still has only ~13% probability! There's 87% chance of getting something OTHER than exactly 10 conversions.</div>""",
        code_str='''from scipy.stats import binom

n = 100  # emails sent
p = 0.10  # conversion rate

# PMF: Probability Mass Function
# P(X = k) = probability of exactly k successes
x = np.arange(0, 25)  # possible outcomes
pmf = binom.pmf(x, n, p)  # probability of each

# Key values
expected = n * p  # mean = 10
std = np.sqrt(n * p * (1-p))  # std dev = 3

print(f"Expected: {expected:.0f}")
print(f"P(exactly 10) = {binom.pmf(10, n, p):.1%}")
print(f"P(7 to 13) = {binom.cdf(13, n, p) - binom.cdf(6, n, p):.1%}")''',
        output_func=show_binomial,
        concept_title="🎯 Binomial Distribution",
        output_title="Email Conversions"
    )

    # Row 3: Poisson Distribution
    def show_poisson():
        from scipy.stats import poisson
        lam = 5
        x = np.arange(0, 15)
        pmf = poisson.pmf(x, lam)
        
        mc = st.columns(3)
        mc[0].metric("λ (avg rate)", f"{lam} calls/hour")
        mc[1].metric("P(exactly 3)", f"{poisson.pmf(3, lam):.1%}")
        mc[2].metric("P(0 calls)", f"{poisson.pmf(0, lam):.2%}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=pmf, marker_color="#f5b731", opacity=0.7))
        fig.add_vline(x=lam, line_dash="dash", line_color="#f45d6d", annotation_text=f"λ = {lam}")
        fig.update_layout(height=220, title="Poisson: Support calls per hour", xaxis_title="Calls", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is the Poisson Distribution?</b>
        <br><br>Poisson answers: <b>"How many events will happen in a given time period?"</b> Unlike binomial (fixed trials), Poisson counts events that could happen any number of times.
        <br><br><b>When to use it:</b>
        <br>• Events happen at a known average rate (λ)
        <br>• Events are independent — one doesn't trigger another
        <br>• Events can happen any number of times (0, 1, 2, 3...)
        <br>• You're counting over a fixed interval (per hour, per day, per page)
        <br><br><b>Real-world examples:</b>
        <br>• Support calls per hour (λ = 5 calls/hour)
        <br>• Website crashes per month (λ = 2 crashes/month)
        <br>• Typos per page (λ = 0.5 typos/page)
        <br>• Customers arriving at a store per minute
        <br><br><b>The key insight:</b> λ is both the mean AND the variance! If λ = 5, you expect 5 events on average, with std dev = √5 ≈ 2.2.
        <br><br><b>Binomial vs Poisson:</b>
        <br>• Binomial: "Out of 100 emails, how many convert?" (fixed trials)
        <br>• Poisson: "How many calls will we get this hour?" (events over time)
        </div>
        <div class="math-box">
        <b>📐 Poisson — Step by Step:</b>
        <br><br><b>Given:</b> λ = 5 calls/hour
        <br><br><b>P(exactly 3 calls):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(X=3) = (λ³ × e⁻λ) / 3!
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= (5³ × e⁻⁵) / 6
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= (125 × 0.0067) / 6
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= 0.84 / 6 = <b>14%</b>
        <br><br><b>P(0 calls):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(X=0) = e⁻⁵ = <b>0.67%</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Less than 1% chance of a quiet hour!
        <br><br><b>Staffing decision:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;P(≥8 calls) = 1 - P(0 to 7) ≈ 13%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Staff for 8+ calls 13% of the time
        </div>
        <div class="insight-box">💡 <b>Practical use:</b> If your support team can handle 7 calls/hour, and λ=5, you'll be overwhelmed about 13% of the time. Hire more staff or reduce call volume!</div>""",
        code_str='''from scipy.stats import poisson

lam = 5  # average 5 calls per hour

p_exactly_3 = poisson.pmf(3, lam)
p_zero = poisson.pmf(0, lam)

print(f"P(exactly 3 calls) = {p_exactly_3:.1%}")
print(f"P(0 calls) = {p_zero:.2%}")''',
        output_func=show_poisson,
        concept_title="📞 Poisson Distribution",
        output_title="Support Calls"
    )


# ═══════════════════════════════════════
# M4: INFERENTIAL STATISTICS
# ═══════════════════════════════════════
elif module == "🧪 M4: Inferential Statistics":
    st.markdown("# 🧪 Module 4: Inferential Statistics")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Inferential Statistics?</b> It's about making <b>conclusions from samples</b> — using data from a few to understand the many.
    You see a difference between two groups. But is it real, or just random noise? Inferential stats answers that.
    <br><br>🎯 <b>Key concepts:</b> Z-scores (how unusual?), T-tests (are groups different?), P-values (is it significant?).
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Z-Score
    def show_zscore():
        from scipy.stats import norm
        data = df["daily_sales"]
        mean_val, std_val = data.mean(), data.std()
        flagship = 1200
        z = (flagship - mean_val) / std_val
        percentile = norm.cdf(z) * 100
        
        mc = st.columns(4)
        mc[0].metric("Flagship Sales", f"${flagship}")
        mc[1].metric("Chain Mean", f"${mean_val:.0f}")
        mc[2].metric("Z-Score", f"{z:.2f}")
        mc[3].metric("Percentile", f"{percentile:.1f}%")
        
        x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 200)
        y = norm.pdf(x, mean_val, std_val)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor='rgba(124,106,255,0.2)', line=dict(color='#7c6aff')))
        fig.add_vline(x=flagship, line_dash="dash", line_color="#f45d6d", annotation_text=f"Flagship: z={z:.2f}")
        fig.update_layout(height=200, title="Where does the flagship fall?", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is a Z-Score?</b>
        <br><br>A z-score answers: <b>"How unusual is this value?"</b> It converts any measurement to a universal scale where 0 = average and each unit = one standard deviation.
        <br><br><b>Why do we need it?</b> Imagine comparing:
        <br>• A store making $1200/day (when average is $500)
        <br>• A student scoring 85 on a test (when average is 70)
        <br>Which is more impressive? Z-scores let you compare apples to oranges!
        <br><br><b>The formula:</b> z = (value - mean) / std dev
        <br><br><b>Interpretation:</b>
        <br>• z = 0 → exactly average
        <br>• z = 1 → one std dev above average (top 16%)
        <br>• z = 2 → two std devs above (top 2.5%)
        <br>• z = 3 → three std devs above (top 0.15% — very rare!)
        <br>• Negative z → below average
        <br><br><b>Rule of thumb:</b> |z| > 3 is extremely unusual — either exceptional performance or a data error!
        </div>
        <div class="math-box">
        <b>📐 Z-Score — Step by Step:</b>
        <br><br><b>Given:</b> Flagship = $1200, μ = $500, σ = $150
        <br><br><b>Calculate z:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;z = (x - μ) / σ
        <br>&nbsp;&nbsp;&nbsp;&nbsp;z = (1200 - 500) / 150
        <br>&nbsp;&nbsp;&nbsp;&nbsp;z = 700 / 150 = <b>4.67</b>
        <br><br><b>Interpretation:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;4.67 std devs above mean!
        <br>&nbsp;&nbsp;&nbsp;&nbsp;|z| > 3 → extremely rare (<0.1%)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Percentile: 99.9998%
        <br><br>🧠 This store is either exceptional OR there's a data error. Investigate!
        </div>
        <div class="warn-box">⚠️ <b>Action item:</b> When you see z > 3, don't just celebrate — verify the data first! Could be a typo ($12000 → $1200).</div>""",
        code_str='''from scipy.stats import norm

data = df["daily_sales"]
mean = data.mean()
std = data.std()

flagship_sales = 1200
z_score = (flagship_sales - mean) / std
percentile = norm.cdf(z_score) * 100

print(f"Z-score: {z_score:.2f}")
print(f"Percentile: {percentile:.1f}%")''',
        output_func=show_zscore,
        concept_title="📊 Z-Score",
        output_title="Is This Store Abnormal?"
    )

    # Row 2: T-Test
    def show_ttest():
        from scipy.stats import ttest_ind
        east = df[df["region"] == "East"]["daily_sales"]
        west = df[df["region"] == "West"]["daily_sales"]
        t_stat, p_val = ttest_ind(east, west)
        
        mc = st.columns(4)
        mc[0].metric("East Mean", f"${east.mean():.0f}")
        mc[1].metric("West Mean", f"${west.mean():.0f}")
        mc[2].metric("T-Statistic", f"{t_stat:.2f}")
        mc[3].metric("P-Value", f"{p_val:.4f}")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=east, name="East", marker_color="#7c6aff", opacity=0.5, nbinsx=15))
        fig.add_trace(go.Histogram(x=west, name="West", marker_color="#22d3a7", opacity=0.5, nbinsx=15))
        fig.update_layout(barmode="overlay", height=200, title="East vs West Sales", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is a T-Test?</b>
        <br><br>A t-test answers: <b>"Are these two groups actually different, or is the difference just random noise?"</b>
        <br><br><b>The problem:</b> You see East stores average $520 and West stores average $480. That's a $40 difference. But is it REAL or just luck?
        <br><br><b>Why we need it:</b> If you flip a coin 10 times, you might get 6 heads and 4 tails. That doesn't mean the coin is biased — it's just random variation. Same with business data!
        <br><br><b>How it works:</b>
        <br>• Calculate the difference between group means
        <br>• Divide by the "noise" (standard error)
        <br>• t = signal / noise
        <br>• Big t → difference is probably real
        <br>• Small t → difference might be random
        <br><br><b>The t-statistic:</b> Think of it as a "signal-to-noise ratio"
        <br>• |t| > 2 → probably significant
        <br>• |t| < 2 → probably just noise
        </div>
        <div class="math-box">
        <b>📐 T-Test — Step by Step:</b>
        <br><br><b>Given:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;East: mean = $520, n = 18, std = $100
        <br>&nbsp;&nbsp;&nbsp;&nbsp;West: mean = $480, n = 32, std = $90
        <br><br><b>Step 1:</b> Calculate difference
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Difference = 520 - 480 = <b>$40</b>
        <br><br><b>Step 2:</b> Calculate standard error
        <br>&nbsp;&nbsp;&nbsp;&nbsp;SE = √(s₁²/n₁ + s₂²/n₂)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;SE = √(100²/18 + 90²/32)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;SE = √(556 + 253) = √809 = <b>$28</b>
        <br><br><b>Step 3:</b> Calculate t
        <br>&nbsp;&nbsp;&nbsp;&nbsp;t = difference / SE = 40 / 28 = <b>1.43</b>
        <br><br>🧠 t = 1.43 → signal is only 1.4× the noise. Not convincing!
        </div>
        <div class="insight-box">💡 <b>Business translation:</b> "The $40 difference between East and West could easily be random variation. Don't make strategic decisions based on this!"</div>""",
        code_str='''from scipy.stats import ttest_ind

east = df[df["region"] == "East"]["daily_sales"]
west = df[df["region"] == "West"]["daily_sales"]

t_statistic, p_value = ttest_ind(east, west)

print(f"East mean: ${east.mean():.0f}")
print(f"West mean: ${west.mean():.0f}")
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.4f}")''',
        output_func=show_ttest,
        concept_title="⚖️ T-Test",
        output_title="East vs West Regions"
    )

    # Row 3: P-Value
    def show_pvalue():
        from scipy.stats import ttest_ind
        east = df[df["region"] == "East"]["daily_sales"]
        west = df[df["region"] == "West"]["daily_sales"]
        t_stat, p_val = ttest_ind(east, west)
        
        significant = p_val < 0.05
        if significant:
            st.markdown(f"""<div class="insight-box">✅ <b>p = {p_val:.4f} < 0.05 → Significant!</b>
            <br>The difference is real, not random noise.</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="warn-box">❌ <b>p = {p_val:.4f} ≥ 0.05 → Not significant.</b>
            <br>The difference could be random variation.</div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is a P-Value?</b>
        <br><br>The p-value answers: <b>"If there's NO real difference, how likely would I see this result by pure chance?"</b>
        <br><br><b>The logic (proof by contradiction):</b>
        <br>1. Assume there's NO real difference (null hypothesis)
        <br>2. Calculate: "How likely is this data under that assumption?"
        <br>3. If very unlikely (p < 0.05) → reject the assumption → difference is real!
        <br><br><b>Analogy:</b> You flip a coin 100 times and get 90 heads. 
        <br>• Assume the coin is fair
        <br>• P(90+ heads | fair coin) = 0.000000001%
        <br>• That's so unlikely → the coin must be biased!
        <br><br><b>The 0.05 threshold:</b>
        <br>• p < 0.05 → "Statistically significant" (less than 5% chance by luck)
        <br>• p ≥ 0.05 → "Not significant" (could easily be random)
        <br><br><b>Common misconception:</b> p = 0.03 does NOT mean "3% chance the result is wrong." It means "3% chance of seeing this if there's no real effect."
        </div>
        <div class="math-box">
        <b>📐 P-Value — Step by Step:</b>
        <br><br><b>Given:</b> t = 1.43, df = 48
        <br><br><b>Question:</b> If East and West are truly equal, what's the chance of seeing t ≥ 1.43?
        <br><br><b>Lookup p-value:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;p = 0.16 (from t-distribution table)
        <br><br><b>Decision (α = 0.05):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;p = 0.16 > 0.05
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ <b>Fail to reject null</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Difference not significant
        <br><br><b>Plain English:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;"There's a 16% chance we'd see this $40 difference even if East and West are identical. That's too high to claim a real difference."
        </div>
        <div class="warn-box">⚠️ <b>p > 0.05 doesn't mean "no difference exists"</b> — it means "we don't have enough evidence." Maybe collect more data!</div>""",
        code_str='''# P-value interpretation
if p_value < 0.05:
    print("Significant! Reject null hypothesis.")
    print("The difference is real.")
else:
    print("Not significant. Fail to reject null.")
    print("Could be random variation.")''',
        output_func=show_pvalue,
        concept_title="🎯 P-Value",
        output_title="Is It Real or Luck?"
    )

    # Row 4: Central Limit Theorem
    def show_clt():
        np.random.seed(42)
        # Create skewed population (exponential)
        population = np.random.exponential(scale=20, size=10000)
        
        sample_sizes = [5, 30, 100]
        n_samples = 500
        
        fig = go.Figure()
        colors = ['#f45d6d', '#f5b731', '#22d3a7']
        
        for idx, n in enumerate(sample_sizes):
            sample_means = [np.random.choice(population, size=n).mean() for _ in range(n_samples)]
            fig.add_trace(go.Histogram(x=sample_means, name=f'n={n}', marker_color=colors[idx], opacity=0.6, nbinsx=30))
        
        fig.add_vline(x=population.mean(), line_dash="dash", line_color="white", annotation_text=f"μ={population.mean():.1f}")
        fig.update_layout(barmode='overlay', height=250, title="Distribution of Sample Means (from skewed population)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        mc = st.columns(3)
        mc[0].metric("Population μ", f"{population.mean():.1f}")
        for idx, n in enumerate(sample_sizes):
            sample_means = [np.random.choice(population, size=n).mean() for _ in range(n_samples)]
            if idx < 2:
                mc[idx+1].metric(f"SE (n={n})", f"{np.std(sample_means):.2f}")

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is the Central Limit Theorem (CLT)?</b>
        <br><br>The CLT is the <b>most important theorem in statistics</b>. It says:
        <br><br><b>"When you take many samples and calculate their means, those means will be normally distributed — even if the original data is NOT normal!"</b>
        <br><br><b>Why does this matter?</b>
        <br>• Real-world data is often skewed (income, delivery times, etc.)
        <br>• But sample MEANS are always normal (if n is large enough)
        <br>• This is WHY t-tests and confidence intervals work!
        <br><br><b>Key implications:</b>
        <br>• Mean of sample means = Population mean (μ)
        <br>• Standard Error = σ/√n (gets smaller as n increases)
        <br>• n ≥ 30 is usually "large enough" for CLT to kick in
        <br><br><b>Analogy:</b> Individual pizza delivery times are all over the place (15-60 min). But the AVERAGE delivery time across 30 orders? That's very predictable and bell-shaped!
        </div>
        <div class="math-box">
        <b>📐 CLT — Step by Step:</b>
        <br><br><b>Given:</b> Population is exponential (skewed right)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;μ = 20, σ = 20
        <br><br><b>Take 500 samples of size n, calculate means:</b>
        <br><br><b>n = 5:</b> Sample means are still somewhat skewed
        <br>&nbsp;&nbsp;&nbsp;&nbsp;SE = σ/√5 = 20/2.24 = <b>8.9</b>
        <br><br><b>n = 30:</b> Sample means look normal!
        <br>&nbsp;&nbsp;&nbsp;&nbsp;SE = σ/√30 = 20/5.48 = <b>3.7</b>
        <br><br><b>n = 100:</b> Sample means are very normal, tight
        <br>&nbsp;&nbsp;&nbsp;&nbsp;SE = σ/√100 = 20/10 = <b>2.0</b>
        <br><br>🧠 Larger n → narrower distribution → more precise estimates!
        </div>
        <div class="insight-box">💡 <b>This is why we can use normal-based tests (t-tests, z-tests) even when data isn't normal — we're testing the MEAN, not individual values!</b></div>""",
        code_str='''import numpy as np

# Skewed population (exponential)
population = np.random.exponential(scale=20, size=10000)

# Take many samples, calculate means
n_samples = 500
sample_size = 30

sample_means = [
    np.random.choice(population, size=sample_size).mean() 
    for _ in range(n_samples)
]

# Sample means are normally distributed!
print(f"Population mean: {population.mean():.1f}")
print(f"Mean of sample means: {np.mean(sample_means):.1f}")
print(f"Standard Error: {np.std(sample_means):.2f}")
print(f"Theoretical SE: {population.std()/np.sqrt(sample_size):.2f}")''',
        output_func=show_clt,
        concept_title="🔔 Central Limit Theorem",
        output_title="Sample Means Become Normal!"
    )

    # Row 5: Confidence Intervals
    def show_ci():
        from scipy.stats import t
        data = df["daily_sales"]
        n = len(data)
        mean = data.mean()
        std = data.std()
        se = std / np.sqrt(n)
        
        confidence = 0.95
        t_crit = t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_crit * se
        ci_lower, ci_upper = mean - margin, mean + margin
        
        mc = st.columns(4)
        mc[0].metric("Sample Mean", f"${mean:.0f}")
        mc[1].metric("Std Error", f"${se:.1f}")
        mc[2].metric("95% CI Lower", f"${ci_lower:.0f}")
        mc[3].metric("95% CI Upper", f"${ci_upper:.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[ci_lower, ci_upper], y=[1, 1], mode='lines', line=dict(color='#22d3a7', width=8), name='95% CI'))
        fig.add_trace(go.Scatter(x=[mean], y=[1], mode='markers', marker=dict(color='#f45d6d', size=15), name='Sample Mean'))
        fig.update_layout(
            height=120, showlegend=True, 
            xaxis_title="Daily Sales ($)",
            yaxis=dict(visible=False, gridcolor='#2d3148'),
            xaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'), title_font=dict(color='#8892b0')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'), legend=dict(font=dict(color='#8892b0')),
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is a Confidence Interval?</b>
        <br><br>A confidence interval gives you a <b>range of plausible values</b> for the true population parameter, not just a single point estimate.
        <br><br><b>The problem with point estimates:</b>
        <br>• Sample mean = $500. But how confident are you?
        <br>• Could the true mean be $450? $550? $600?
        <br><br><b>95% Confidence Interval:</b>
        <br>• "We're 95% confident the true mean is between $470 and $530"
        <br>• If we repeated this study 100 times, ~95 of those intervals would contain the true mean
        <br><br><b>The formula:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;CI = x̄ ± (t × SE)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;where SE = s/√n
        <br><br><b>Width depends on:</b>
        <br>• <b>Sample size (n):</b> Larger n → narrower CI (more precise)
        <br>• <b>Variability (s):</b> More spread → wider CI (less certain)
        <br>• <b>Confidence level:</b> 99% CI is wider than 95% CI
        </div>
        <div class="math-box">
        <b>📐 Confidence Interval — Step by Step:</b>
        <br><br><b>Given:</b> n=50, x̄=$500, s=$100
        <br><br><b>Step 1:</b> Calculate Standard Error
        <br>&nbsp;&nbsp;&nbsp;&nbsp;SE = s/√n = 100/√50 = 100/7.07 = <b>$14.14</b>
        <br><br><b>Step 2:</b> Find t-critical (95%, df=49)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;t = <b>2.01</b> (from t-table)
        <br><br><b>Step 3:</b> Calculate margin of error
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Margin = t × SE = 2.01 × 14.14 = <b>$28.42</b>
        <br><br><b>Step 4:</b> Build interval
        <br>&nbsp;&nbsp;&nbsp;&nbsp;CI = 500 ± 28.42
        <br>&nbsp;&nbsp;&nbsp;&nbsp;CI = <b>[$472, $528]</b>
        <br><br>🧠 "We're 95% confident the true average daily sales is between $472 and $528"
        </div>
        <div class="warn-box">⚠️ <b>Common misconception:</b> "95% CI" does NOT mean "95% chance the true mean is in this interval." The true mean is fixed — either it's in there or it's not. The 95% refers to the method's long-run success rate.</div>""",
        code_str='''from scipy.stats import t

data = df["daily_sales"]
n = len(data)
mean = data.mean()
std = data.std()
se = std / np.sqrt(n)

# 95% confidence interval
confidence = 0.95
t_critical = t.ppf((1 + confidence) / 2, df=n-1)
margin = t_critical * se

ci_lower = mean - margin
ci_upper = mean + margin

print(f"Sample mean: ${mean:.0f}")
print(f"95% CI: [${ci_lower:.0f}, ${ci_upper:.0f}]")''',
        output_func=show_ci,
        concept_title="📏 Confidence Intervals",
        output_title="95% CI for Mean Sales"
    )

    iq([
        {"q": "Explain the Central Limit Theorem and why it's important.", "d": "Medium", "c": ["Google", "Meta"],
         "a": "<b>CLT:</b> Sample means are normally distributed regardless of population shape (if n is large enough). <b>Why important:</b> It's the foundation for hypothesis testing and confidence intervals — we can use normal-based methods even with non-normal data.",
         "t": "Mention that n≥30 is the common rule of thumb."},
        {"q": "What's the difference between standard deviation and standard error?", "d": "Easy", "c": ["Amazon", "Netflix"],
         "a": "<b>Std Dev (σ):</b> Measures spread of individual data points. <b>Std Error (SE):</b> Measures precision of the sample mean estimate. SE = σ/√n. SE gets smaller as sample size increases.",
         "t": "SE tells you how much the sample mean would vary if you repeated the study."},
        {"q": "Interpret a 95% confidence interval of [$450, $550] for mean sales.", "d": "Medium", "c": ["Google", "Apple"],
         "a": "We're 95% confident the true population mean falls between $450 and $550. If we repeated this sampling 100 times, about 95 of those intervals would contain the true mean.",
         "t": "Avoid saying '95% probability the true mean is in this range' — that's a common misconception."},
    ])


# ═══════════════════════════════════════
# M5: CORRELATION
# ═══════════════════════════════════════
elif module == "🔗 M5: Correlation":
    st.markdown("# 🔗 Module 5: Correlation")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Correlation?</b> It measures how <b>two variables move together</b>. When one goes up, does the other go up too?
    Correlation ranges from -1 (perfect opposite) to +1 (perfect together), with 0 meaning no relationship.
    <br><br>🎯 <b>Key insight:</b> Correlation ≠ Causation! Just because two things move together doesn't mean one causes the other.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Pearson Correlation
    def show_correlation():
        x, y = df["ad_spend"], df["daily_sales"]
        r = np.corrcoef(x, y)[0, 1]
        
        mc = st.columns(3)
        mc[0].metric("Correlation (r)", f"{r:.3f}")
        mc[1].metric("Strength", "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak")
        mc[2].metric("Direction", "Positive" if r > 0 else "Negative")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#22d3a7', size=8, opacity=0.6)))
        z = np.polyfit(x, y, 1)
        fig.add_trace(go.Scatter(x=x, y=np.polyval(z, x), mode='lines', line=dict(color='#f45d6d', dash='dash')))
        fig.update_layout(height=250, title=f"Ad Spend vs Sales (r = {r:.3f})", xaxis_title="Ad Spend ($)", yaxis_title="Daily Sales ($)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Correlation?</b>
        <br><br>Correlation measures <b>how two things move together</b>. When one goes up, does the other go up too? Down? Or no pattern at all?
        <br><br><b>The correlation coefficient (r):</b>
        <br>• r = +1 → Perfect positive: both go up together
        <br>• r = 0 → No relationship: knowing one tells you nothing about the other
        <br>• r = -1 → Perfect negative: one goes up, other goes down
        <br><br><b>Strength interpretation:</b>
        <br>• |r| < 0.3 → Weak (barely related)
        <br>• 0.3 ≤ |r| < 0.7 → Moderate (some relationship)
        <br>• |r| ≥ 0.7 → Strong (closely related)
        <br><br><b>Real-world examples:</b>
        <br>• Height & weight: r ≈ +0.7 (taller people tend to weigh more)
        <br>• Study time & grades: r ≈ +0.5 (more study → better grades, usually)
        <br>• Price & demand: r ≈ -0.6 (higher price → less demand)
        <br><br><b>Why it matters:</b> If ad spend and sales have r = 0.78, you know ads are working! But be careful...
        </div>
        <div class="math-box">
        <b>📐 Correlation — Step by Step:</b>
        <br><br><b>Given:</b> Ad spend & Sales for 5 stores
        <br>&nbsp;&nbsp;&nbsp;&nbsp;X = [100, 150, 200, 250, 300]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Y = [400, 450, 520, 580, 650]
        <br><br><b>Step 1:</b> Calculate means
        <br>&nbsp;&nbsp;&nbsp;&nbsp;x̄ = 200, ȳ = 520
        <br><br><b>Step 2:</b> Calculate deviations
        <br>&nbsp;&nbsp;&nbsp;&nbsp;(100-200)(400-520) = (-100)(-120) = 12000
        <br>&nbsp;&nbsp;&nbsp;&nbsp;(150-200)(450-520) = (-50)(-70) = 3500
        <br>&nbsp;&nbsp;&nbsp;&nbsp;... and so on
        <br><br><b>Step 3:</b> Covariance = Σ(deviations) / n = <b>5000</b>
        <br><br><b>Step 4:</b> Std devs: σx = 71, σy = 90
        <br><br><b>Step 5:</b> r = Cov / (σx × σy) = 5000 / (71×90) = <b>0.78</b>
        <br><br>🧠 Strong positive correlation! More ad spend → more sales
        </div>
        <div class="warn-box">⚠️ <b>CRITICAL:</b> Correlation ≠ Causation! Ice cream sales and drowning deaths are correlated (both increase in summer). Ice cream doesn't cause drowning — summer does!</div>""",
        code_str='''import numpy as np

x = df["ad_spend"]
y = df["daily_sales"]

# Method 1: numpy
r = np.corrcoef(x, y)[0, 1]

# Method 2: pandas
r = df["ad_spend"].corr(df["daily_sales"])

print(f"Correlation: {r:.3f}")''',
        output_func=show_correlation,
        concept_title="📈 Pearson Correlation",
        output_title="Ad Spend vs Sales"
    )

    # Row 2: Correlation Heatmap
    def show_heatmap():
        num_cols = ["daily_sales", "avg_order_value", "delivery_time_min", "employees", "ad_spend"]
        corr = df[num_cols].corr()
        
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,'#f45d6d'],[0.5,'#1a1d2e'],[1,'#22d3a7']], zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=10)
        ))
        fig.update_layout(height=300, **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why use a Correlation Heatmap?</b>
        <br><br>When you have many variables, checking correlations one by one is tedious. A heatmap shows ALL relationships at once!
        <br><br><b>How to read it:</b>
        <br>• 🟢 Green = positive correlation (both increase together)
        <br>• 🔴 Red = negative correlation (one up, other down)
        <br>• ⚫ Dark/Black = no correlation (independent)
        <br>• Diagonal is always 1.0 (variable correlates perfectly with itself)
        <br><br><b>What to look for:</b>
        <br>• <b>Strong greens:</b> Variables that move together — potential predictors!
        <br>• <b>Strong reds:</b> Variables that move opposite — interesting relationships
        <br>• <b>Multicollinearity:</b> If two predictors are highly correlated (r > 0.8), you might only need one
        <br><br><b>Business insight:</b> If "ad_spend" and "daily_sales" are green, ads might be working. If "delivery_time" and "customer_rating" are red, slow delivery hurts ratings.
        </div>
        <div class="math-box">
        <b>📐 Reading the Heatmap:</b>
        <br><br><b>Example findings:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;ad_spend ↔ daily_sales: r = 0.78 (strong positive)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ More ad spend → more sales
        <br><br>&nbsp;&nbsp;&nbsp;&nbsp;delivery_time ↔ rating: r = -0.45 (moderate negative)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Longer delivery → lower ratings
        <br><br>&nbsp;&nbsp;&nbsp;&nbsp;employees ↔ sales: r = 0.12 (weak)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ More staff doesn't guarantee more sales
        <br><br>🧠 <b>Action:</b> Focus on ad spend (strong effect) rather than hiring (weak effect)
        </div>
        <div class="warn-box">⚠️ <b>Correlation ≠ Causation!</b>
        <br>Ice cream sales and drowning deaths are correlated — but ice cream doesn't cause drowning! (Summer causes both)
        <br><br>Always ask: "Is there a hidden third variable?"</div>""",
        code_str='''# Correlation matrix
num_cols = ["daily_sales", "avg_order_value", 
            "delivery_time_min", "employees", "ad_spend"]

corr_matrix = df[num_cols].corr()
print(corr_matrix.round(2))

# Visualize as heatmap
import seaborn as sns
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', 
            center=0, vmin=-1, vmax=1)''',
        output_func=show_heatmap,
        concept_title="🗺️ Correlation Heatmap",
        output_title="All Variables"
    )


# ═══════════════════════════════════════
# M6: REGRESSION
# ═══════════════════════════════════════
elif module == "📉 M6: Regression":
    st.markdown("# 📉 Module 6: Regression")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Regression?</b> It's about <b>predicting one variable from another</b> by finding the best-fit line through your data.
    "If I spend $X on ads, how much sales can I expect?" — regression answers this with a formula.
    <br><br>🎯 <b>Key concepts:</b> Slope (rate of change), intercept (starting point), R² (how good is the fit?).
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Linear Regression
    def show_regression():
        x = df["ad_spend"].values
        y = df["daily_sales"].values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        mc = st.columns(3)
        mc[0].metric("Slope", f"${slope:.2f}/$ ad spend")
        mc[1].metric("Intercept", f"${intercept:.0f}")
        mc[2].metric("R²", f"{r_squared:.3f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#22d3a7', size=8, opacity=0.6), name='Stores'))
        fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', line=dict(color='#f45d6d', width=2), name='Best Fit'))
        fig.update_layout(height=250, title=f"Sales = {slope:.2f} × Ad Spend + {intercept:.0f}", xaxis_title="Ad Spend ($)", yaxis_title="Daily Sales ($)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Linear Regression?</b>
        <br><br>Linear regression finds the <b>best straight line through your data</b> so you can make predictions. It's the bridge from "these things are correlated" to "I can predict one from the other."
        <br><br><b>The equation:</b> y = mx + b
        <br>• <b>m (slope):</b> "For every $1 more in ad spend, sales change by $m"
        <br>• <b>b (intercept):</b> "If ad spend is $0, sales would be $b"
        <br><br><b>Why "best" line?</b> The line that minimizes the total squared distance from all points. This is called "least squares" — it finds the line where prediction errors are as small as possible.
        <br><br><b>Real-world use:</b>
        <br>• "If I spend $300 on ads, how much sales can I expect?"
        <br>• "How much should I budget for ads to hit $600 in sales?"
        <br><br><b>The power:</b> Regression turns correlation into a formula you can use for planning and forecasting!
        </div>
        <div class="math-box">
        <b>📐 Linear Regression — Step by Step:</b>
        <br><br><b>Given:</b> x̄ = $200, ȳ = $520
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Cov(X,Y) = 5000, Var(X) = 5000
        <br><br><b>Step 1:</b> Calculate slope
        <br>&nbsp;&nbsp;&nbsp;&nbsp;m = Cov(X,Y) / Var(X)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;m = 5000 / 5000 = <b>1.0</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ "Every $1 ad spend → $1 more sales"
        <br><br><b>Step 2:</b> Calculate intercept
        <br>&nbsp;&nbsp;&nbsp;&nbsp;b = ȳ - m × x̄
        <br>&nbsp;&nbsp;&nbsp;&nbsp;b = 520 - 1.0 × 200 = <b>320</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ "Base sales without ads = $320"
        <br><br><b>Equation:</b> Sales = 1.0 × AdSpend + 320
        <br><br><b>Prediction:</b> If ad spend = $250:
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Sales = 1.0 × 250 + 320 = <b>$570</b>
        </div>
        <div class="insight-box">💡 <b>Business insight:</b> With slope = 1.0, every dollar in ads returns a dollar in sales. That's break-even! You need slope > 1 for ads to be profitable (after costs).</div>""",
        code_str='''from sklearn.linear_model import LinearRegression

X = df[["ad_spend"]]
y = df["daily_sales"]

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)

print(f"Equation: y = {slope:.2f}x + {intercept:.0f}")
print(f"R² = {r_squared:.3f}")''',
        output_func=show_regression,
        concept_title="📏 Linear Regression",
        output_title="Ad Spend → Sales"
    )

    # Row 2: R-Squared & Predictions
    def show_r_squared():
        x = df["ad_spend"].values
        y = df["daily_sales"].values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        st.markdown(f"""<div class="insight-box">
        <b>R² = {r_squared:.3f}</b> means ad spend explains <b>{r_squared*100:.1f}%</b> of the variation in sales.
        </div>""", unsafe_allow_html=True)
        
        new_ad_spend = st.slider("Predict sales for ad spend:", 100, 400, 250)
        predicted = slope * new_ad_spend + intercept
        st.metric(f"Predicted Sales (${new_ad_spend} ad spend)", f"${predicted:.0f}")

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is R² (R-Squared)?</b>
        <br><br>R² answers: <b>"How much of the variation in Y does my model explain?"</b>
        <br><br><b>The intuition:</b>
        <br>• Sales vary from store to store (some make $400, others $600)
        <br>• Part of that variation is explained by ad spend
        <br>• Part is unexplained (other factors, randomness)
        <br>• R² = explained variation / total variation
        <br><br><b>Interpretation:</b>
        <br>• R² = 0 → Model explains nothing (useless)
        <br>• R² = 0.5 → Model explains 50% of variation (decent)
        <br>• R² = 0.8 → Model explains 80% of variation (good!)
        <br>• R² = 1.0 → Model explains everything (perfect, but suspicious)
        <br><br><b>What about the unexplained part?</b>
        <br>If R² = 0.6, then 40% of sales variation comes from other factors: location, staff quality, competition, weather, etc.
        <br><br><b>Warning:</b> High R² doesn't mean the model is correct! You can get high R² with overfitting or spurious correlations.
        </div>
        <div class="math-box">
        <b>📐 R² — Step by Step:</b>
        <br><br><b>Given:</b> Actual Y = [400, 450, 520, 580, 650]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Predicted Ŷ = [420, 470, 520, 570, 620]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Mean ȳ = 520
        <br><br><b>Step 1:</b> SS_residual (prediction errors²)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= (400-420)² + (450-470)² + (520-520)² + (580-570)² + (650-620)²
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= 400 + 400 + 0 + 100 + 900 = <b>1800</b>
        <br><br><b>Step 2:</b> SS_total (variance from mean)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= (400-520)² + (450-520)² + ... + (650-520)²
        <br>&nbsp;&nbsp;&nbsp;&nbsp;= 14400 + 4900 + 0 + 3600 + 16900 = <b>39800</b>
        <br><br><b>Step 3:</b> R² = 1 - SS_res/SS_tot
        <br>&nbsp;&nbsp;&nbsp;&nbsp;R² = 1 - 1800/39800 = 1 - 0.045 = <b>0.955</b>
        <br><br>🧠 Model explains 95.5% of sales variation!
        </div>
        <div class="warn-box">⚠️ <b>Don't chase R² blindly!</b> A model with R² = 0.99 might be overfitting. A model with R² = 0.4 might still be useful for business decisions.</div>""",
        code_str='''# R-squared
r_squared = model.score(X, y)
print(f"R² = {r_squared:.3f}")
print(f"Model explains {r_squared*100:.1f}% of variance")

# Make predictions
new_ad_spend = 250
predicted_sales = model.predict([[new_ad_spend]])[0]
print(f"Predicted: ${predicted_sales:.0f}")''',
        output_func=show_r_squared,
        concept_title="📊 R-Squared & Predictions",
        output_title="Model Quality"
    )

    iq([
        {"q": "Explain linear regression to a non-technical person.", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "Linear regression draws the <b>best straight line</b> through data to make predictions. Example: 'For every extra $1 in ad spend, sales go up by $1.16.'",
         "t": "Use a concrete example everyone understands."},
    ])
