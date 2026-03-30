# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="📊 Phase 2: DS Core", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; background-color: #000000; }
[data-testid="stSidebar"] { background-color: #000000; }
.main .block-container { background-color: #000000; }
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

@st.cache_data
def get_telecom_data(n=500):
    np.random.seed(42)
    tenure = np.random.randint(1, 72, n)
    monthly = np.random.normal(65, 20, n).clip(20, 120).round(2)
    total = (tenure * monthly + np.random.normal(0, 500, n)).round(2)
    support = np.random.poisson(2, n)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2])
    churn_score = -0.03*tenure + 0.02*monthly + 0.15*support + np.random.normal(0, 0.5, n)
    churned = (churn_score > np.percentile(churn_score, 70)).astype(int)
    age = np.random.randint(18, 70, n).astype(float)
    age[np.random.choice(n, 30, replace=False)] = np.nan
    return pd.DataFrame({"customer_id": range(1, n+1), "age": age, "tenure_months": tenure,
        "monthly_charges": monthly, "total_charges": total, "support_tickets": support,
        "contract": contract, "churned": churned})

df = get_telecom_data(500)

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
    st.markdown("## 📊 Phase 2: Data Science")
    st.caption("Concept | Code + Output")
    st.divider()
    module = st.radio("Module:", [
        "🏠 Overview",
        "🔄 M1: Data Lifecycle",
        "🧹 M2: Data Cleaning",
        "🏗️ M3: Feature Engineering",
        "📊 M4: Data Visualization",
    ], label_visibility="collapsed")


# ═══════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════
if module == "🏠 Overview":
    st.markdown("# 📊 Phase 2: Data Science Core")
    st.caption("Concept on Left · Code + Output on Right")

    st.markdown("""<div class="concept-card">
    <b>🎬 Welcome to Data Science!</b> You're now a data scientist at TelecomCo with 500 customers.
    Your boss asks: "Why are customers leaving? Can we predict who will churn?"
    <br><br>This phase teaches you the <b>data science workflow</b> — from raw data to insights.
    </div>""", unsafe_allow_html=True)

    modules = [
        ("🔄", "Data Lifecycle", "Week 1", "#7c6aff", "The journey from raw data to insights"),
        ("🧹", "Data Cleaning", "Week 1-2", "#22d3a7", "Handle missing values, outliers, duplicates"),
        ("🏗️", "Feature Engineering", "Week 2-3", "#f5b731", "Create new features that boost model performance"),
        ("📊", "Data Visualization", "Week 3-4", "#f45d6d", "Tell stories with charts and graphs"),
    ]

    for icon, title, week, color, desc in modules:
        st.markdown(f"""<div class="concept-card" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span>
        <span class="iq-tag" style="background:{color}22;color:{color}">{week}</span>
        <b style="margin-left:6px">{title}</b>
        <br><span style="color:#8892b0">{desc}</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# M1: DATA LIFECYCLE
# ═══════════════════════════════════════
elif module == "🔄 M1: Data Lifecycle":
    st.markdown("# 🔄 Module 1: Data Lifecycle")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is the Data Lifecycle?</b> It's the <b>journey data takes</b> from raw collection to actionable insights.
    Every data science project follows a similar path: understand the problem, explore data, clean it, model it, evaluate, deploy.
    <br><br>🎯 <b>Key insight:</b> Data preparation takes 60-80% of a data scientist's time! The "glamorous" modeling is only 10-20%.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Load & Explore
    def show_data_explore():
        st.dataframe(df.head(8), use_container_width=True, height=180)
        mc = st.columns(3)
        mc[0].metric("Rows", f"{df.shape[0]:,}")
        mc[1].metric("Columns", f"{df.shape[1]}")
        mc[2].metric("Missing", f"{df.isna().sum().sum()}")

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why do we explore data first?</b>
        <br><br>Imagine trying to cook a meal without checking what ingredients you have. Data exploration is the same — you need to know what you're working with before you can do anything useful.
        <br><br><b>The first questions to ask:</b>
        <br>• <b>How big is it?</b> 500 rows? 5 million? This affects your approach.
        <br>• <b>What columns do I have?</b> What information is available?
        <br>• <b>What types are they?</b> Numbers? Text? Dates?
        <br>• <b>Is anything missing?</b> Missing data = potential problems
        <br><br><b>Your exploration toolkit:</b>
        <br>• <code>df.head()</code> — See first few rows (sanity check)
        <br>• <code>df.shape</code> — How many rows and columns?
        <br>• <code>df.info()</code> — Data types and missing values
        <br>• <code>df.describe()</code> — Summary statistics for numbers
        <br><br><b>Real-world tip:</b> Spend 10-15 minutes exploring before diving into analysis. It saves hours of debugging later!
        </div>
        <div class="insight-box">💡 <b>Pro tip:</b> Always check <code>df.head()</code> AND <code>df.tail()</code> — sometimes data quality degrades at the end of a file!</div>""",
        code_str='''import pandas as pd

# Load data
df = pd.read_csv("telecom_customers.csv")

# Quick exploration
df.head()
df.shape
df.info()
df.describe()''',
        output_func=show_data_explore,
        concept_title="📋 Load & Explore",
        output_title="Telecom Customer Data"
    )

    # Row 2: Data Types
    def show_dtypes():
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count().values,
            "Sample": [str(df[c].iloc[0])[:20] for c in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True, height=200)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why do data types matter?</b>
        <br><br>Data types tell Python how to treat your data. Get them wrong, and things break in confusing ways.
        <br><br><b>The main types:</b>
        <br>• <b>int64/float64:</b> Numbers you can do math with
        <br>• <b>object:</b> Text (strings) — often categories in disguise
        <br>• <b>datetime64:</b> Dates and times — enables time-based analysis
        <br>• <b>category:</b> Efficient storage for repeated text values
        <br>• <b>bool:</b> True/False values
        <br><br><b>Common problems:</b>
        <br>• <b>Numbers as strings:</b> "123" instead of 123 — can't do math!
        <br>• <b>Dates as strings:</b> "2024-01-15" — can't sort chronologically
        <br>• <b>Categories as numbers:</b> Region coded as 1, 2, 3 — misleading correlations
        <br><br><b>Why it matters:</b>
        <br>• Wrong types → wrong analysis
        <br>• Wrong types → memory waste (storing "Male"/"Female" as strings vs category)
        <br>• Wrong types → ML models fail
        </div>
        <div class="math-box">
        <b>📐 Memory Impact Example:</b>
        <br><br><b>Column "contract" with 500 rows:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;As object (string): ~40 KB
        <br>&nbsp;&nbsp;&nbsp;&nbsp;As category: ~1 KB
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ <b>40× memory savings!</b>
        <br><br><b>With 5 million rows:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Object: 400 MB
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Category: 10 MB
        <br><br>🧠 At scale, data types = the difference between "runs" and "crashes"
        </div>
        <div class="warn-box">⚠️ <b>Red flag:</b> If a numeric column shows as "object", it probably has text mixed in (like "N/A" or "$100"). Clean it first!</div>""",
        code_str='''# Check data types
df.dtypes

# Convert types
df["age"] = df["age"].astype(float)
df["contract"] = df["contract"].astype("category")

# Check for issues
df.info()''',
        output_func=show_dtypes,
        concept_title="🔢 Data Types",
        output_title="Column Information"
    )

    # Row 3: CRISP-DM Lifecycle
    def show_lifecycle():
        stages = ["Business\nUnderstanding", "Data\nUnderstanding", "Data\nPreparation", 
                  "Modeling", "Evaluation", "Deployment"]
        colors = ["#7c6aff", "#22d3a7", "#f5b731", "#f45d6d", "#e879a8", "#5eaeff"]
        
        fig = go.Figure()
        for i, (stage, color) in enumerate(zip(stages, colors)):
            angle = i * 60 * np.pi / 180
            x, y = 2 * np.cos(angle), 2 * np.sin(angle)
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', 
                marker=dict(size=50, color=color), text=[stage], textposition="bottom center",
                textfont=dict(size=10, color='#e2e8f0'), showlegend=False))
        fig.update_layout(height=280, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=40, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is CRISP-DM?</b>
        <br><br>CRISP-DM (Cross-Industry Standard Process for Data Mining) is the <b>standard workflow for data science projects</b>. It's been around since 1996 and is still the most widely used framework.
        <br><br><b>The 6 stages:</b>
        <br><br>1. <b>Business Understanding:</b> What problem are we solving? What does success look like? (Often skipped, always regretted)
        <br><br>2. <b>Data Understanding:</b> What data do we have? Is it enough? Is it clean?
        <br><br>3. <b>Data Preparation:</b> Clean, transform, engineer features. <b>This is 60-80% of the work!</b>
        <br><br>4. <b>Modeling:</b> Train ML models. The "glamorous" part that's actually only 10-20% of the work.
        <br><br>5. <b>Evaluation:</b> Does the model actually solve the business problem? Not just "is accuracy high?"
        <br><br>6. <b>Deployment:</b> Put it in production where it creates value.
        <br><br><b>The dirty secret:</b> Most projects fail at stages 1 or 6, not 4. Building a model is easy; solving the right problem and deploying it is hard.
        </div>
        <div class="math-box">
        <b>📐 Time Allocation (Reality vs Expectation):</b>
        <br><br><b>What beginners expect:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Data prep: 20%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Modeling: 60%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Other: 20%
        <br><br><b>What actually happens:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Business understanding: 10%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Data understanding: 15%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Data preparation: <b>50-60%</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Modeling: <b>10-15%</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Evaluation: 10%
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Deployment: 10%
        <br><br>🧠 The best data scientists are great at data prep, not just modeling!
        </div>
        <div class="warn-box">⚠️ <b>Career tip:</b> If you only learn modeling, you'll be frustrated. Learn to love data cleaning — it's where the real work happens!</div>""",
        code_str='''# CRISP-DM in practice
# 1. Business: "Predict customer churn"
# 2. Data: Load and explore
# 3. Prep: Clean, engineer features
# 4. Model: Train classifier
# 5. Evaluate: Check accuracy, precision
# 6. Deploy: API or batch predictions''',
        output_func=show_lifecycle,
        concept_title="🔄 CRISP-DM Lifecycle",
        output_title="The 6 Stages"
    )

    iq([
        {"q": "Walk me through a typical data science project lifecycle.", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>CRISP-DM:</b> (1) Business Understanding — define the problem. (2) Data Understanding — explore data. (3) Data Preparation — clean & transform (60-80% of time!). (4) Modeling — train models. (5) Evaluation — test performance. (6) Deployment — put in production.",
         "t": "Emphasize that data prep takes most of the time."},
    ])


# ═══════════════════════════════════════
# M2: DATA CLEANING
# ═══════════════════════════════════════
elif module == "🧹 M2: Data Cleaning":
    st.markdown("# 🧹 Module 2: Data Cleaning")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Data Cleaning?</b> It's the process of <b>fixing messy data</b> — handling missing values, removing duplicates, and dealing with outliers.
    Real-world data is never perfect. Garbage in = garbage out, so cleaning is critical for reliable analysis.
    <br><br>🎯 <b>Key tasks:</b> Find & fill missing values, detect & remove duplicates, identify & handle outliers.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Missing Values
    def show_missing():
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        missing_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values, "Percent": missing_pct.values})
        missing_df = missing_df[missing_df["Missing"] > 0]
        
        mc = st.columns(3)
        mc[0].metric("Total Missing", f"{df.isna().sum().sum()}")
        mc[1].metric("Columns Affected", f"{(missing > 0).sum()}")
        mc[2].metric("Age Missing %", f"{missing_pct['age']:.1f}%")
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, height=100)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why are missing values a problem?</b>
        <br><br>Missing values are like holes in your data. Most ML algorithms can't handle them — they'll either crash or give wrong results.
        <br><br><b>Types of missing data:</b>
        <br>• <b>MCAR (Missing Completely At Random):</b> Random sensor failures. Safe to drop.
        <br>• <b>MAR (Missing At Random):</b> Young people skip "income" question. Can impute.
        <br>• <b>MNAR (Missing Not At Random):</b> High earners hide income. Dangerous — the missingness IS information!
        <br><br><b>Strategies:</b>
        <br>• <b>Drop rows:</b> If < 5% missing AND MCAR. Simple but loses data.
        <br>• <b>Fill with mean:</b> For normally distributed numeric data.
        <br>• <b>Fill with median:</b> For skewed numeric data (robust to outliers).
        <br>• <b>Fill with mode:</b> For categorical data.
        <br>• <b>Flag it:</b> Create "is_missing" column — sometimes missingness is a feature!
        <br><br><b>The key question:</b> WHY is this data missing? The answer determines your strategy.
        </div>
        <div class="math-box">
        <b>📐 Imputation Impact — Example:</b>
        <br><br><b>Original ages:</b> [25, 30, NaN, 35, NaN, 40]
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Mean of known = (25+30+35+40)/4 = <b>32.5</b>
        <br><br><b>After mean imputation:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;[25, 30, <b>32.5</b>, 35, <b>32.5</b>, 40]
        <br><br><b>Effect on statistics:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Original std dev: 6.5
        <br>&nbsp;&nbsp;&nbsp;&nbsp;After imputation: 5.2
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Imputation <b>reduces variance</b>!
        <br><br>🧠 This is why imputation can be dangerous — it makes data look more consistent than it really is.
        </div>
        <div class="warn-box">⚠️ <b>Never blindly drop or fill!</b> First understand WHY data is missing. If high earners hide income, filling with mean will underestimate their income.</div>""",
        code_str='''# Check missing values
df.isna().sum()
df.isna().sum() / len(df) * 100  # percentage

# Fill with median (robust to outliers)
df["age"].fillna(df["age"].median(), inplace=True)

# Fill with mode (for categorical)
df["contract"].fillna(df["contract"].mode()[0], inplace=True)

# Drop rows with any missing
df.dropna(inplace=True)''',
        output_func=show_missing,
        concept_title="🕳️ Missing Values",
        output_title="What's Missing?"
    )

    # Row 2: Duplicates
    def show_duplicates():
        n_dupes = df.duplicated().sum()
        n_dupes_subset = df.duplicated(subset=["customer_id"]).sum()
        
        mc = st.columns(3)
        mc[0].metric("Exact Duplicates", n_dupes)
        mc[1].metric("Duplicate IDs", n_dupes_subset)
        mc[2].metric("Unique Customers", df["customer_id"].nunique())
        
        st.markdown("""<div class="insight-box">💡 Always check for duplicates before analysis — they can skew your results!</div>""", unsafe_allow_html=True)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why are duplicates dangerous?</b>
        <br><br>Duplicates make your data lie to you. If one customer appears twice, they count double in your analysis — skewing averages, inflating counts, and corrupting ML models.
        <br><br><b>Types of duplicates:</b>
        <br>• <b>Exact duplicates:</b> Every column is identical. Usually data loading errors.
        <br>• <b>Partial duplicates:</b> Same customer_id but different values. Could be updates or errors.
        <br>• <b>Semantic duplicates:</b> "John Smith" and "J. Smith" — same person, different records.
        <br><br><b>How they sneak in:</b>
        <br>• Data loaded multiple times
        <br>• Merging tables without proper keys
        <br>• Form submitted twice
        <br>• ETL pipeline bugs
        <br><br><b>Detection strategy:</b>
        <br>1. Check exact duplicates: <code>df.duplicated().sum()</code>
        <br>2. Check key columns: <code>df.duplicated(subset=['customer_id'])</code>
        <br>3. Compare unique counts: <code>df['id'].nunique()</code> vs <code>len(df)</code>
        </div>
        <div class="math-box">
        <b>📐 Impact of Duplicates — Example:</b>
        <br><br><b>Scenario:</b> Customer #42 appears 3 times
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Customer #42 churned = True
        <br><br><b>Without deduplication:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Churn rate = 103/500 = <b>20.6%</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;(Customer #42 counted 3× in numerator)
        <br><br><b>After deduplication:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Churn rate = 101/498 = <b>20.3%</b>
        <br><br>🧠 Small difference here, but with more duplicates, your metrics become meaningless!
        </div>
        <div class="warn-box">⚠️ <b>Before removing:</b> Ask "which record is correct?" For partial duplicates, you might want the most recent one (<code>keep='last'</code>).</div>""",
        code_str='''# Check for duplicates
df.duplicated().sum()  # exact duplicates

# Check specific columns
df.duplicated(subset=["customer_id"]).sum()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Keep last occurrence
df.drop_duplicates(subset=["customer_id"], keep="last")''',
        output_func=show_duplicates,
        concept_title="👯 Duplicates",
        output_title="Finding Copies"
    )

    # Row 3: Outliers
    def show_outliers_cleaning():
        col = "monthly_charges"
        data = df[col]
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = df[(data < lower) | (data > upper)]
        
        mc = st.columns(4)
        mc[0].metric("Q1", f"${Q1:.0f}")
        mc[1].metric("Q3", f"${Q3:.0f}")
        mc[2].metric("IQR", f"${IQR:.0f}")
        mc[3].metric("Outliers", len(outliers))
        
        fig = go.Figure()
        fig.add_trace(go.Box(x=data, marker_color="#22d3a7", boxmean=True, name="Monthly Charges"))
        fig.update_layout(height=120, **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What are outliers and should you remove them?</b>
        <br><br>Outliers are data points that are <b>far from the rest</b>. But "far" is subjective — and removal is often wrong!
        <br><br><b>Outliers can be:</b>
        <br>• <b>Data errors:</b> Typo ($12000 instead of $120) → Fix or remove
        <br>• <b>Measurement errors:</b> Faulty sensor → Remove
        <br>• <b>Legitimate extremes:</b> Your best customer really does spend 10× average → Keep!
        <br>• <b>The signal you're looking for:</b> In fraud detection, outliers ARE the fraud!
        <br><br><b>Detection methods:</b>
        <br>• <b>IQR method:</b> Outside Q1-1.5×IQR to Q3+1.5×IQR
        <br>• <b>Z-score:</b> |z| > 3 (more than 3 std devs from mean)
        <br>• <b>Visual:</b> Box plots, scatter plots
        <br><br><b>Handling options:</b>
        <br>• <b>Remove:</b> Only if you're SURE it's an error
        <br>• <b>Cap (Winsorize):</b> Replace with fence value
        <br>• <b>Transform:</b> Log transform reduces outlier impact
        <br>• <b>Keep:</b> If it's real, it's data!
        </div>
        <div class="math-box">
        <b>📐 IQR Method — Step by Step:</b>
        <br><br><b>Given:</b> Q1 = $51, Q3 = $79
        <br><br><b>Step 1:</b> Calculate IQR
        <br>&nbsp;&nbsp;&nbsp;&nbsp;IQR = Q3 - Q1 = 79 - 51 = <b>$28</b>
        <br><br><b>Step 2:</b> Calculate fences
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Lower = Q1 - 1.5×IQR = 51 - 42 = <b>$9</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Upper = Q3 + 1.5×IQR = 79 + 42 = <b>$121</b>
        <br><br><b>Step 3:</b> Flag outliers
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Values < $9 or > $121 = outliers
        <br><br><b>Why 1.5?</b> For normal data, this captures 99.3% of values. The 0.7% outside are "unusual."
        </div>
        <div class="warn-box">⚠️ <b>Golden rule:</b> Never auto-delete outliers! Always investigate first. In fraud detection, outliers ARE the signal you're looking for!</div>""",
        code_str='''# IQR method
Q1 = df["monthly_charges"].quantile(0.25)
Q3 = df["monthly_charges"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df["monthly_charges"] < lower) | 
              (df["monthly_charges"] > upper)]

# Cap outliers (winsorization)
df["monthly_charges"] = df["monthly_charges"].clip(lower, upper)''',
        output_func=show_outliers_cleaning,
        concept_title="🔔 Outlier Detection",
        output_title="Monthly Charges"
    )

    iq([
        {"q": "How do you handle missing values?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Strategies:</b> (1) <b>Drop</b> if < 5% and MCAR. (2) <b>Impute</b> with mean (normal), median (skewed), or mode (categorical). (3) <b>Flag</b> with is_missing column. (4) <b>Model-based</b> imputation for complex cases.",
         "t": "Always understand WHY data is missing before choosing a strategy."},
    ])


# ═══════════════════════════════════════
# M3: FEATURE ENGINEERING
# ═══════════════════════════════════════
elif module == "🏗️ M3: Feature Engineering":
    st.markdown("# 🏗️ Module 3: Feature Engineering")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Feature Engineering?</b> It's the art of <b>creating new columns</b> from existing data that help ML models learn better.
    Raw data rarely has the perfect features. You transform, combine, and encode data to extract more signal.
    <br><br>🎯 <b>Key techniques:</b> Create ratios, bin continuous values, encode categories, scale features.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Creating Features
    def show_new_features():
        df_fe = df.copy()
        df_fe["avg_monthly_spend"] = df_fe["total_charges"] / df_fe["tenure_months"].clip(1)
        df_fe["tickets_per_month"] = df_fe["support_tickets"] / df_fe["tenure_months"].clip(1)
        df_fe["is_new_customer"] = (df_fe["tenure_months"] <= 6).astype(int)
        
        st.dataframe(df_fe[["customer_id", "tenure_months", "total_charges", "avg_monthly_spend", 
                           "support_tickets", "tickets_per_month", "is_new_customer"]].head(6), 
                    use_container_width=True, height=180)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 What is Feature Engineering?</b>
        <br><br>Feature engineering is the art of <b>creating new columns that help ML models learn better</b>. Raw data rarely has the perfect features — you need to transform, combine, and extract information.
        <br><br><b>Why it matters:</b> A mediocre algorithm with great features beats a great algorithm with mediocre features. Feature engineering is often the difference between a model that works and one that doesn't.
        <br><br><b>Common techniques:</b>
        <br>• <b>Ratios:</b> spend_per_month = total_charges / tenure
        <br>• <b>Differences:</b> price_change = current_price - last_price
        <br>• <b>Bins:</b> age_group = "young" / "middle" / "senior"
        <br>• <b>Flags:</b> is_new_customer = tenure < 6 months
        <br>• <b>Aggregations:</b> avg_order_value per customer
        <br>• <b>Time features:</b> day_of_week, is_weekend, month
        <br><br><b>The key insight:</b> Think about what information would help YOU predict the outcome. If you'd want to know "how long has this customer been with us?" — create a tenure feature!
        </div>
        <div class="math-box">
        <b>📐 Feature Engineering — Examples:</b>
        <br><br><b>Ratio feature:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;total_charges = $720, tenure = 12 months
        <br>&nbsp;&nbsp;&nbsp;&nbsp;avg_monthly_spend = 720/12 = <b>$60/month</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ More interpretable than raw total!
        <br><br><b>Interaction feature:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;tickets = 5, tenure = 12
        <br>&nbsp;&nbsp;&nbsp;&nbsp;tickets_per_month = 5/12 = <b>0.42</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ Captures "complaint intensity"
        <br><br><b>Flag feature:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;tenure = 3 months
        <br>&nbsp;&nbsp;&nbsp;&nbsp;is_new_customer = (3 ≤ 6) = <b>True (1)</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ New customers behave differently!
        </div>
        <div class="insight-box">💡 <b>Pro tip:</b> Talk to domain experts! They know which features matter. A telecom expert might say "customers who call support in their first month are 3× more likely to churn."</div>""",
        code_str='''# Create ratio features
df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"]
df["tickets_per_month"] = df["support_tickets"] / df["tenure_months"]

# Create flag features
df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
df["high_spender"] = (df["monthly_charges"] > 80).astype(int)

# Create bins
df["tenure_group"] = pd.cut(df["tenure_months"], 
    bins=[0, 12, 36, 72], labels=["New", "Mid", "Long"])''',
        output_func=show_new_features,
        concept_title="🔧 Creating Features",
        output_title="New Columns"
    )

    # Row 2: Encoding Categorical
    def show_encoding():
        # One-hot encoding example
        contract_dummies = pd.get_dummies(df["contract"], prefix="contract")
        encoded_df = pd.concat([df[["customer_id", "contract"]], contract_dummies], axis=1).head(6)
        st.dataframe(encoded_df, use_container_width=True, height=180)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why do we need to encode categories?</b>
        <br><br>ML algorithms work with numbers, not text. "Month-to-month" means nothing to a computer. Encoding converts categories to numbers in a way that preserves meaning.
        <br><br><b>Encoding methods:</b>
        <br><br><b>One-Hot Encoding:</b>
        <br>• Creates a binary column for each category
        <br>• "Month-to-month" → [1, 0, 0]
        <br>• "One year" → [0, 1, 0]
        <br>• <b>Use when:</b> Categories have no natural order
        <br>• <b>Warning:</b> Explodes dimensions with many categories!
        <br><br><b>Label Encoding:</b>
        <br>• Assigns numbers: 0, 1, 2, 3...
        <br>• "Month-to-month" → 0, "One year" → 1, "Two year" → 2
        <br>• <b>Use when:</b> Categories have natural order (ordinal)
        <br>• <b>Warning:</b> Implies 2 > 1 > 0, which may be wrong!
        <br><br><b>Target Encoding:</b>
        <br>• Replace category with mean of target variable
        <br>• "Month-to-month" → 0.42 (42% churn rate)
        <br>• <b>Use when:</b> Many categories, want to capture relationship with target
        <br>• <b>Warning:</b> Can cause data leakage if not done carefully!
        </div>
        <div class="math-box">
        <b>📐 Encoding — Step by Step:</b>
        <br><br><b>One-Hot Example:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Original: contract = "Month-to-month"
        <br>&nbsp;&nbsp;&nbsp;&nbsp;After:
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;contract_Month-to-month = <b>1</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;contract_One_year = <b>0</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;contract_Two_year = <b>0</b>
        <br><br><b>Target Encoding Example:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Month-to-month: 250 customers, 105 churned
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Churn rate = 105/250 = 0.42
        <br>&nbsp;&nbsp;&nbsp;&nbsp;contract_target = <b>0.42</b>
        <br><br>🧠 Target encoding captures "Month-to-month is risky" in one number!
        </div>
        <div class="warn-box">⚠️ <b>Cardinality trap:</b> If a column has 1000 unique values, one-hot creates 1000 columns! Use target encoding or embeddings instead.</div>""",
        code_str='''# One-hot encoding
contract_dummies = pd.get_dummies(df["contract"], prefix="contract")
df = pd.concat([df, contract_dummies], axis=1)

# Label encoding (for ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["contract_encoded"] = le.fit_transform(df["contract"])

# Target encoding
df["contract_target"] = df.groupby("contract")["churned"].transform("mean")''',
        output_func=show_encoding,
        concept_title="🔤 Encoding Categorical",
        output_title="One-Hot Encoding"
    )

    # Row 3: Scaling
    def show_scaling():
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        original = df[["monthly_charges", "tenure_months"]].head(6)
        
        scaler_std = StandardScaler()
        scaler_mm = MinMaxScaler()
        
        scaled_std = scaler_std.fit_transform(df[["monthly_charges", "tenure_months"]])
        scaled_mm = scaler_mm.fit_transform(df[["monthly_charges", "tenure_months"]])
        
        comparison = pd.DataFrame({
            "Original_Charges": original["monthly_charges"].values,
            "StandardScaled": scaled_std[:6, 0].round(2),
            "MinMaxScaled": scaled_mm[:6, 0].round(2)
        })
        st.dataframe(comparison, use_container_width=True, height=180)

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why do we need to scale features?</b>
        <br><br>Imagine two features: age (18-80) and income ($20,000-$500,000). Without scaling, income dominates because its numbers are bigger — even though age might be more important!
        <br><br><b>Algorithms that NEED scaling:</b>
        <br>• KNN (uses distances)
        <br>• SVM (uses distances)
        <br>• Neural Networks (gradient descent)
        <br>• PCA (variance-based)
        <br>• Regularized regression (L1/L2)
        <br><br><b>Algorithms that DON'T need scaling:</b>
        <br>• Decision Trees (splits don't care about scale)
        <br>• Random Forest
        <br>• XGBoost/LightGBM
        <br><br><b>Scaling methods:</b>
        <br>• <b>StandardScaler:</b> Mean=0, Std=1. Good for normally distributed data.
        <br>• <b>MinMaxScaler:</b> Range [0,1]. Good when you need bounded values.
        <br>• <b>RobustScaler:</b> Uses median/IQR. Good when you have outliers.
        </div>
        <div class="math-box">
        <b>📐 Scaling — Step by Step:</b>
        <br><br><b>Given:</b> Monthly charges = $85
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Mean = $65, Std = $20
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Min = $20, Max = $120
        <br><br><b>Standard Scaling (z-score):</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;z = (x - μ) / σ
        <br>&nbsp;&nbsp;&nbsp;&nbsp;z = (85 - 65) / 20 = 20/20 = <b>1.0</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ "1 std dev above mean"
        <br><br><b>MinMax Scaling:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;x' = (x - min) / (max - min)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;x' = (85 - 20) / (120 - 20) = 65/100 = <b>0.65</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;→ "65% of the way from min to max"
        <br><br>🧠 Both put features on comparable scales!
        </div>
        <div class="insight-box">💡 <b>Important:</b> Fit the scaler on training data only, then transform both train and test. Otherwise you leak test information!</div>""",
        code_str='''from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaling (z-score)
scaler = StandardScaler()
df[["charges_scaled", "tenure_scaled"]] = scaler.fit_transform(
    df[["monthly_charges", "tenure_months"]]
)

# Min-Max scaling (0 to 1)
mm_scaler = MinMaxScaler()
df[["charges_mm", "tenure_mm"]] = mm_scaler.fit_transform(
    df[["monthly_charges", "tenure_months"]]
)''',
        output_func=show_scaling,
        concept_title="📏 Feature Scaling",
        output_title="Comparison"
    )

    iq([
        {"q": "What is feature engineering? Give examples.", "d": "Medium", "c": ["Google", "Meta"],
         "a": "<b>Feature engineering</b> is creating new features from existing data to improve model performance. <b>Examples:</b> (1) Ratios: spend_per_month. (2) Bins: age_group. (3) Flags: is_new_customer. (4) Aggregations: avg_order_value. (5) Time features: day_of_week, is_weekend.",
         "t": "Mention that good features often matter more than algorithm choice."},
    ])


# ═══════════════════════════════════════
# M4: DATA VISUALIZATION
# ═══════════════════════════════════════
elif module == "📊 M4: Data Visualization":
    st.markdown("# 📊 Module 4: Data Visualization")
    st.caption("Concept on Left · Code + Output on Right")
    st.markdown("""<div class="concept-card">
    <b>What is Data Visualization?</b> It's <b>telling stories with pictures</b> — turning numbers into charts that reveal patterns instantly.
    A good visualization can communicate insights in seconds that would take paragraphs to explain in text.
    <br><br>🎯 <b>Key charts:</b> Histograms (distributions), bar charts (comparisons), scatter plots (relationships), heatmaps (correlations).
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Row 1: Distribution Plots
    def show_distribution():
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["monthly_charges"], nbinsx=25, marker_color="#7c6aff", opacity=0.7))
        fig.add_vline(x=df["monthly_charges"].mean(), line_dash="dash", line_color="#f45d6d", 
                     annotation_text=f"Mean: ${df['monthly_charges'].mean():.0f}")
        fig.add_vline(x=df["monthly_charges"].median(), line_dash="dash", line_color="#22d3a7",
                     annotation_text=f"Median: ${df['monthly_charges'].median():.0f}")
        fig.update_layout(height=250, title="Monthly Charges Distribution", xaxis_title="Monthly Charges ($)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why visualize distributions?</b>
        <br><br>A distribution shows you the <b>shape of your data</b> — where values cluster, how spread out they are, and whether there are unusual patterns.
        <br><br><b>What to look for:</b>
        <br>• <b>Shape:</b> Is it bell-shaped (normal)? Skewed? Bimodal (two peaks)?
        <br>• <b>Center:</b> Where's the peak? Is mean ≈ median?
        <br>• <b>Spread:</b> Tight cluster or wide range?
        <br>• <b>Outliers:</b> Any bars far from the rest?
        <br><br><b>Skewness clues:</b>
        <br>• Mean > Median → Right-skewed (tail on right)
        <br>• Mean < Median → Left-skewed (tail on left)
        <br>• Mean ≈ Median → Symmetric
        <br><br><b>Why it matters:</b>
        <br>• Skewed data → use median, not mean
        <br>• Bimodal data → maybe two different populations?
        <br>• Outliers → investigate before modeling
        </div>
        <div class="math-box">
        <b>📐 Reading the Histogram:</b>
        <br><br><b>Example findings:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Mean = $65, Median = $63
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Mean > Median → slight right skew
        <br><br><b>What this tells us:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Most customers pay $50-$80/month
        <br>&nbsp;&nbsp;&nbsp;&nbsp;A few high spenders pull the mean up
        <br>&nbsp;&nbsp;&nbsp;&nbsp;For "typical" customer, use median ($63)
        <br><br>🧠 <b>Business insight:</b> If you're setting a "standard" plan price, $63 is more representative than $65!
        </div>
        <div class="insight-box">💡 <b>Pro tip:</b> Always plot your target variable first! If churn is 90% "No", you have class imbalance — standard accuracy metrics will be misleading.</div>""",
        code_str='''import plotly.express as px

# Histogram
fig = px.histogram(df, x="monthly_charges", nbins=25)
fig.show()

# With mean/median lines
import plotly.graph_objects as go
fig.add_vline(x=df["monthly_charges"].mean(), 
              line_dash="dash", annotation_text="Mean")
fig.add_vline(x=df["monthly_charges"].median(), 
              line_dash="dash", annotation_text="Median")''',
        output_func=show_distribution,
        concept_title="📊 Distribution Plots",
        output_title="Monthly Charges"
    )

    # Row 2: Categorical Plots
    def show_categorical():
        churn_by_contract = df.groupby("contract")["churned"].mean().reset_index()
        churn_by_contract["churned"] = churn_by_contract["churned"] * 100
        
        fig = go.Figure()
        colors = ["#f45d6d", "#f5b731", "#22d3a7"]
        fig.add_trace(go.Bar(x=churn_by_contract["contract"], y=churn_by_contract["churned"],
                            marker_color=colors, text=churn_by_contract["churned"].round(1).astype(str) + "%",
                            textposition="outside"))
        fig.update_layout(height=250, title="Churn Rate by Contract Type", yaxis_title="Churn Rate (%)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why use bar charts for categories?</b>
        <br><br>Bar charts are the <b>best way to compare groups</b>. They make differences obvious at a glance — no mental math required.
        <br><br><b>When to use bar charts:</b>
        <br>• Comparing counts across categories
        <br>• Comparing rates/percentages across groups
        <br>• Showing rankings
        <br><br><b>Design tips:</b>
        <br>• Sort bars by value (not alphabetically) for easy comparison
        <br>• Use color to highlight the key insight
        <br>• Add data labels for precise values
        <br>• Start y-axis at 0 (don't truncate!)
        <br><br><b>Variations:</b>
        <br>• <b>Grouped bars:</b> Compare across 2 dimensions
        <br>• <b>Stacked bars:</b> Show composition (parts of whole)
        <br>• <b>Horizontal bars:</b> Better for long category names
        </div>
        <div class="math-box">
        <b>📐 Churn Rate Calculation:</b>
        <br><br><b>Month-to-month:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;250 customers, 105 churned
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Rate = 105/250 = <b>42%</b>
        <br><br><b>One year:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;150 customers, 30 churned
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Rate = 30/150 = <b>20%</b>
        <br><br><b>Two year:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;100 customers, 14 churned
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Rate = 14/100 = <b>14%</b>
        <br><br>🧠 Month-to-month churns at <b>3× the rate</b> of two-year contracts!
        </div>
        <div class="insight-box">💡 <b>Business action:</b> Incentivize customers to sign longer contracts — it dramatically reduces churn!</div>""",
        code_str='''# Churn rate by contract type
churn_by_contract = df.groupby("contract")["churned"].mean()
print(churn_by_contract)

# Bar chart
fig = px.bar(churn_by_contract.reset_index(), 
             x="contract", y="churned",
             title="Churn Rate by Contract")
fig.show()''',
        output_func=show_categorical,
        concept_title="📊 Categorical Plots",
        output_title="Churn by Contract"
    )

    # Row 3: Correlation Heatmap
    def show_corr_heatmap():
        num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "churned"]
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
        <b>Correlation Heatmap:</b> See ALL relationships at once.
        <br><br>🟢 Green = positive correlation
        <br>🔴 Red = negative correlation
        <br>⚫ Dark = no relationship
        </div>
        <div class="math-box">
        <b>📐 Correlation Calculation — Example:</b>
        <br><br><b>Tenure vs Churn:</b> r = -0.35
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Negative → longer tenure = less churn
        <br><br><b>Support Tickets vs Churn:</b> r = +0.28
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Positive → more tickets = more churn
        <br><br><b>Interpretation:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;|r| < 0.3 → Weak
        <br>&nbsp;&nbsp;&nbsp;&nbsp;0.3 ≤ |r| < 0.7 → Moderate
        <br>&nbsp;&nbsp;&nbsp;&nbsp;|r| ≥ 0.7 → Strong
        </div>
        <div class="warn-box">⚠️ Correlation ≠ Causation! Always investigate further.</div>""",
        code_str='''# Correlation matrix
num_cols = ["tenure_months", "monthly_charges", 
            "total_charges", "support_tickets", "churned"]
corr = df[num_cols].corr()

# Heatmap
import seaborn as sns
sns.heatmap(corr, annot=True, cmap="RdYlGn", 
            center=0, vmin=-1, vmax=1)''',
        output_func=show_corr_heatmap,
        concept_title="🗺️ Correlation Heatmap",
        output_title="All Relationships"
    )

    # Row 4: Scatter Plot
    def show_scatter():
        fig = px.scatter(df, x="tenure_months", y="monthly_charges", color="churned",
                        color_discrete_map={0: "#22d3a7", 1: "#f45d6d"},
                        labels={"churned": "Churned"}, opacity=0.6)
        fig.update_layout(height=280, title="Tenure vs Monthly Charges (by Churn)", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    split_row(
        concept_html="""<div class="concept-card">
        <b>🤔 Why use scatter plots?</b>
        <br><br>Scatter plots show <b>relationships between two numeric variables</b>. Each point is one observation, and patterns emerge from the cloud of points.
        <br><br><b>What to look for:</b>
        <br>• <b>Direction:</b> Upward slope (positive), downward (negative), or flat (none)
        <br>• <b>Strength:</b> Tight cluster (strong) or scattered (weak)
        <br>• <b>Shape:</b> Linear or curved?
        <br>• <b>Outliers:</b> Points far from the pattern
        <br>• <b>Clusters:</b> Distinct groups?
        <br><br><b>Adding dimensions:</b>
        <br>• <b>Color:</b> Add a third variable (categorical)
        <br>• <b>Size:</b> Add a fourth variable (numeric)
        <br>• <b>Facets:</b> Split into subplots
        <br><br><b>In this example:</b> We color by churn status to see if churners cluster in certain regions of the tenure-charges space.
        </div>
        <div class="math-box">
        <b>📐 Reading the Scatter Plot:</b>
        <br><br><b>Pattern observed:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Red dots (churned) cluster in:
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Low tenure (left side)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• High charges (top)
        <br><br><b>Interpretation:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;New customers (low tenure) with high bills
        <br>&nbsp;&nbsp;&nbsp;&nbsp;are most likely to churn.
        <br><br><b>Business insight:</b>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;Target retention efforts at:
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Customers < 12 months tenure
        <br>&nbsp;&nbsp;&nbsp;&nbsp;• Who pay > $70/month
        <br><br>🧠 This is a high-risk segment!
        </div>
        <div class="insight-box">💡 <b>Action:</b> New customers with high charges are at risk. Consider offering them a discount or loyalty bonus in their first year!</div>""",
        code_str='''import plotly.express as px

# Scatter with color by churn
fig = px.scatter(df, 
    x="tenure_months", 
    y="monthly_charges",
    color="churned",
    opacity=0.6,
    title="Tenure vs Charges by Churn"
)
fig.show()''',
        output_func=show_scatter,
        concept_title="📈 Scatter Plots",
        output_title="Tenure vs Charges"
    )

    iq([
        {"q": "What visualizations would you use to explore a new dataset?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Numeric:</b> Histograms (distribution), box plots (outliers), scatter plots (relationships). <b>Categorical:</b> Bar charts (counts), grouped bars (comparisons). <b>Relationships:</b> Correlation heatmap, pair plots. <b>Time:</b> Line charts.",
         "t": "Match the visualization to the data type and question you're answering."},
    ])
