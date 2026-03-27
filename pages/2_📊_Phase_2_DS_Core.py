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
    background: #252840; border-left: 4px solid #22d3a7;
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

with st.sidebar:
    st.markdown("## 📊 Phase 2: Data Science")
    st.caption("2–3 Weeks · Turn raw data into insights")
    st.divider()
    module = st.radio("Module:", [
        "🏠 Overview",
        "🧠 M0: What is DS, ML & AI?",
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
    st.caption("2–3 Weeks · Turn raw data → insights. This is where statistics meets the real world.")

    st.markdown("""<div class="story-box">
    <b>Phase 1</b> gave you the math toolkit. <b>Phase 2</b> teaches you to use it on messy, real-world data.
    You'll learn to clean data, create features, visualize patterns, and ask the right questions.
    <br><br><b>The key skill:</b> Looking at raw data and asking <i>"What story is this data trying to tell me?"</i>
    </div>""", unsafe_allow_html=True)

    modules = [
        ("🔄", "Data Lifecycle", "Week 1", "#22d3a7", "Where data comes from, how it flows, what can go wrong at each stage."),
        ("🧹", "Data Cleaning", "Week 1–2", "#7c6aff", "Missing values, outliers, duplicates, type errors — making messy data usable."),
        ("🏗️", "Feature Engineering", "Week 2", "#f5b731", "Creating new columns that reveal hidden patterns — the art of data science."),
        ("📊", "Data Visualization", "Week 2–3", "#f45d6d", "Telling stories with charts — histograms, scatter plots, heatmaps, dashboards."),
    ]
    for icon, title, week, color, desc in modules:
        st.markdown(f"""<div class="story-box" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span>
        <span class="iq-tag" style="background:{color}22;color:{color}">{week}</span>
        <b style="margin-left:6px">{title}</b>
        <br><span style="color:#8892b0">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-box">
    <b>✅ After Phase 2 you will:</b><br>
    • Identify patterns and anomalies in any dataset<br>
    • Clean and prepare data for modeling<br>
    • Create features that improve predictions<br>
    • Build visualizations that tell a clear story<br>
    • Ask the right questions: "Why is throughput low in this cell?" "Which KPI impacts churn most?"
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# M0: WHAT IS DS, ML & AI?
# ═══════════════════════════════════════
elif module == "🧠 M0: What is DS, ML & AI?":
    st.markdown("# 🧠 What is Data Science, Machine Learning & AI?")
    st.caption("Before we dive into tools and techniques — let's understand the big picture.")

    st.markdown("""<div class="story-box">
    These three terms get thrown around interchangeably, but they mean very different things.
    Think of them as <b>nested circles</b> — AI is the biggest, ML is inside it, and Data Science overlaps with both.
    </div>""", unsafe_allow_html=True)

    # ── Visual: The Nested Circles ──
    import plotly.graph_objects as go
    fig_circles = go.Figure()
    # AI circle (biggest)
    theta = np.linspace(0, 2*np.pi, 100)
    fig_circles.add_trace(go.Scatter(x=3.5*np.cos(theta), y=3.5*np.sin(theta)+0.5, mode='lines',
        line=dict(color='#f45d6d', width=3), fill='toself', fillcolor='rgba(244,93,109,0.06)', name='AI'))
    # ML circle (medium)
    fig_circles.add_trace(go.Scatter(x=2.3*np.cos(theta), y=2.3*np.sin(theta)+0.3, mode='lines',
        line=dict(color='#f5b731', width=3), fill='toself', fillcolor='rgba(245,183,49,0.08)', name='ML'))
    # DL circle (small)
    fig_circles.add_trace(go.Scatter(x=1.2*np.cos(theta), y=1.2*np.sin(theta)+0.1, mode='lines',
        line=dict(color='#7c6aff', width=3), fill='toself', fillcolor='rgba(124,106,255,0.1)', name='Deep Learning'))
    # DS circle (overlapping)
    fig_circles.add_trace(go.Scatter(x=2.0*np.cos(theta)+2.5, y=2.0*np.sin(theta)-0.5, mode='lines',
        line=dict(color='#22d3a7', width=3, dash='dash'), fill='toself', fillcolor='rgba(34,211,167,0.06)', name='Data Science'))
    # Labels
    fig_circles.add_annotation(x=0, y=3.5, text="<b>Artificial Intelligence</b>", showarrow=False, font=dict(color='#f45d6d', size=13))
    fig_circles.add_annotation(x=0, y=2.2, text="<b>Machine Learning</b>", showarrow=False, font=dict(color='#f5b731', size=12))
    fig_circles.add_annotation(x=0, y=0.8, text="<b>Deep Learning</b>", showarrow=False, font=dict(color='#7c6aff', size=11))
    fig_circles.add_annotation(x=3.8, y=-0.5, text="<b>Data Science</b>", showarrow=False, font=dict(color='#22d3a7', size=12))
    fig_circles.update_layout(height=400, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, scaleanchor='y'), yaxis=dict(visible=False),
        margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_circles, use_container_width=True, config={"displayModeBar": False})

    # ── AI ──
    st.markdown("### 🤖 Artificial Intelligence (AI)")
    st.markdown("""<div class="story-box" style="border-left:4px solid #f45d6d">
    <b>What it is:</b> Any system that can perform tasks that normally require human intelligence —
    understanding language, recognizing images, making decisions, playing games.
    <br><br>
    <b>Analogy:</b> AI is the <b>dream</b> — making machines "smart." It's the broadest term.
    A chess program from 1997 is AI. Siri is AI. ChatGPT is AI. A self-driving car is AI.
    <br><br>
    <b>Two types:</b>
    <br>• <b>Narrow AI:</b> Good at ONE specific task (Siri, spam filters, recommendation engines). This is what exists today.
    <br>• <b>General AI (AGI):</b> Good at ANY intellectual task a human can do. This doesn't exist yet.
    <br><br>
    <b>Real-world examples:</b> Voice assistants, fraud detection, medical diagnosis, autonomous vehicles, language translation.
    </div>""", unsafe_allow_html=True)

    # ── ML ──
    st.markdown("### 🤖 Machine Learning (ML)")
    st.markdown("""<div class="story-box" style="border-left:4px solid #f5b731">
    <b>What it is:</b> A <b>subset of AI</b> where machines learn patterns from data instead of being
    explicitly programmed with rules.
    <br><br>
    <b>Traditional programming:</b> Human writes rules → computer follows them.
    <br><i>"If temperature > 100°F AND humidity > 80%, then alert = True"</i>
    <br><br>
    <b>Machine learning:</b> Human gives data + answers → computer discovers the rules itself.
    <br><i>"Here are 10,000 weather records labeled 'alert' or 'no alert.' Figure out the pattern."</i>
    <br><br>
    <b>Why it's powerful:</b> Some patterns are too complex for humans to write rules for.
    How do you write rules to recognize a cat in a photo? You can't — but ML can learn it from 10,000 cat photos.
    <br><br>
    <b>Real-world examples:</b> Netflix recommendations, email spam filters, credit scoring, predictive maintenance, customer churn prediction.
    </div>""", unsafe_allow_html=True)

    # ── DS ──
    st.markdown("### 📊 Data Science (DS)")
    st.markdown("""<div class="story-box" style="border-left:4px solid #22d3a7">
    <b>What it is:</b> The practice of <b>extracting insights and knowledge from data</b> using statistics,
    programming, and domain expertise. It overlaps with ML but is broader.
    <br><br>
    <b>A data scientist:</b>
    <br>• Asks the right <b>questions</b> ("Why are customers leaving?")
    <br>• Collects and <b>cleans</b> data
    <br>• <b>Explores</b> data to find patterns (EDA)
    <br>• Builds <b>models</b> to predict or explain (using ML)
    <br>• <b>Communicates</b> insights to stakeholders
    <br><br>
    <b>Key difference from ML:</b> ML is a tool. Data Science is the whole process of using data to solve problems — ML is just one tool in the toolbox.
    <br><br>
    <b>Analogy:</b> If ML is a <b>hammer</b>, Data Science is <b>building a house</b>. You need the hammer, but also blueprints, measurements, materials, and someone who knows what they're building.
    </div>""", unsafe_allow_html=True)

    # ── Why do we need them? ──
    st.markdown("### 🤔 Why Do We Need Them?")

    reasons = [
        ("📊", "Too much data for humans", "#7c6aff",
         "A telecom company generates <b>terabytes of data daily</b> — call records, network logs, customer interactions. No human can read it all. DS/ML processes it automatically and finds patterns humans would miss."),
        ("🔮", "Prediction saves money", "#22d3a7",
         "Predicting which customers will churn <b>before they leave</b> lets you intervene. Predicting equipment failure <b>before it breaks</b> prevents downtime. Prediction turns reactive businesses into proactive ones."),
        ("⚡", "Speed of decisions", "#f5b731",
         "A fraud detection system needs to decide in <b>milliseconds</b> whether a transaction is legitimate. A human can't review 10,000 transactions per second — but an ML model can."),
        ("🎯", "Personalization at scale", "#f45d6d",
         "Netflix has 200M+ users. Each one sees a <b>different homepage</b> tailored to their taste. No human could manually curate 200M homepages — ML does it automatically."),
    ]

    for icon, title, color, desc in reasons:
        st.markdown(f"""<div class="story-box" style="border-left:4px solid {color}">
        <span style="font-size:1.3rem">{icon}</span> <b>{title}</b>
        <br>{desc}
        </div>""", unsafe_allow_html=True)

    # ── How do we use them? ──
    st.markdown("### 🛠️ How Are They Used? (Real Examples)")

    st.markdown("""<div class="story-box">
    <b>📞 Telecom (your domain):</b>
    <br>• <b>DS:</b> "Which KPIs are declining? Why is throughput low in cell X?" → EDA + visualization
    <br>• <b>ML:</b> "Which customers will churn next month?" → Classification model
    <br>• <b>AI:</b> "Automatically optimize network routing in real-time" → Reinforcement learning
    <br><br>
    <b>🏦 Banking:</b>
    <br>• <b>DS:</b> "What's our risk exposure?" → Statistical analysis
    <br>• <b>ML:</b> "Is this transaction fraud?" → Anomaly detection model
    <br>• <b>AI:</b> "Chatbot that handles customer queries" → NLP
    <br><br>
    <b>🏥 Healthcare:</b>
    <br>• <b>DS:</b> "Which patients are at risk?" → Data analysis
    <br>• <b>ML:</b> "Predict disease from medical images" → Computer vision
    <br>• <b>AI:</b> "AI assistant that helps doctors diagnose" → LLM + medical knowledge
    </div>""", unsafe_allow_html=True)

    # ── Quick comparison table ──
    st.markdown("### 📋 Quick Comparison")
    comparison = pd.DataFrame({
        "": ["What is it?", "Input", "Output", "Example", "Who does it?"],
        "Data Science 📊": [
            "Extracting insights from data",
            "Raw data + business question",
            "Insights, dashboards, recommendations",
            "\"Churn increased 15% — here's why\"",
            "Data Scientist / Analyst",
        ],
        "Machine Learning 🤖": [
            "Machines learning patterns from data",
            "Labeled data (features + target)",
            "A trained model that makes predictions",
            "\"This customer has 73% churn probability\"",
            "ML Engineer / Data Scientist",
        ],
        "Artificial Intelligence 🧠": [
            "Machines performing human-like tasks",
            "Any data (text, images, sensor data)",
            "Intelligent behavior / decisions",
            "\"Self-driving car navigates traffic\"",
            "AI Researcher / ML Engineer",
        ],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("""<div class="key-box">
    <b>🎯 Key Takeaway:</b> AI is the big dream. ML is the main technique to get there. Data Science is the
    practical discipline of using data (including ML) to solve real business problems. You need all three —
    but you <b>start with Data Science</b> because it teaches you to think with data.
    </div>""", unsafe_allow_html=True)

    iq([
        {"q": "Explain the difference between AI, ML, and Data Science to a non-technical executive.", "d": "Easy", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>AI</b> is the broad goal of making machines intelligent — like a self-driving car or a chatbot. <b>ML</b> is the main technique to achieve AI — instead of writing rules, we let machines learn patterns from data. <b>Data Science</b> is the practice of using data to answer questions and make decisions — ML is one of its tools. <b>Analogy:</b> AI is the destination, ML is the engine, Data Science is the driver who knows where to go.",
         "t": "Use a simple analogy. Executives don't want technical definitions — they want to understand the value."},
        {"q": "When would you use a simple rule-based system vs ML?", "d": "Medium", "c": ["Amazon", "Google", "Apple"],
         "a": "<b>Rule-based:</b> When the logic is simple, well-understood, and doesn't change. Example: 'If order > $1000, require manager approval.' Clear, auditable, no data needed. <b>ML:</b> When patterns are complex, change over time, or are hard to articulate. Example: 'Which emails are spam?' — the patterns are subtle, evolving, and impossible to capture with fixed rules. <b>Rule of thumb:</b> If you can write the rules in an afternoon, don't use ML. If you can't, ML is your friend.",
         "t": "Saying 'don't use ML when rules work' shows maturity. Many candidates over-engineer with ML."},
        {"q": "What's the difference between a Data Scientist and an ML Engineer?", "d": "Easy", "c": ["Meta", "Google", "Netflix"],
         "a": "<b>Data Scientist:</b> Focuses on analysis, insights, and experimentation. Asks 'what's happening and why?' Works in notebooks, builds prototypes, communicates with stakeholders. <b>ML Engineer:</b> Focuses on building, deploying, and scaling ML models in production. Asks 'how do we make this work reliably at scale?' Writes production code, builds pipelines, monitors models. <b>Overlap:</b> Both need statistics and ML knowledge. The DS leans toward analysis, the MLE leans toward engineering.",
         "t": "Know which role you're interviewing for and tailor your answer accordingly."},
        {"q": "Give me a real-world example where ML is better than traditional programming.", "d": "Easy", "c": ["Amazon", "General"],
         "a": "<b>Email spam detection.</b> Traditional approach: write rules like 'if contains Nigerian prince, flag as spam.' Problem: spammers constantly change tactics. You'd need to update rules daily. <b>ML approach:</b> Train a model on millions of labeled emails (spam/not spam). The model learns subtle patterns — word combinations, sender behavior, link patterns — and adapts as spam evolves. <b>Result:</b> Gmail's spam filter catches 99.9% of spam using ML. No human could write rules that effective.",
         "t": "Pick an example everyone uses daily — email, Netflix, Google Search."},
        {"q": "A business stakeholder says 'We need AI.' How do you respond?", "d": "Hard", "c": ["Google", "Meta", "Amazon"],
         "a": "I'd ask clarifying questions: <b>(1) What problem are you trying to solve?</b> (Maybe they need a dashboard, not AI.) <b>(2) What data do you have?</b> (No data = no ML.) <b>(3) What does success look like?</b> (Measurable outcome.) <b>(4) What's the current process?</b> (Maybe a simple rule or SQL query solves it.) <b>Then I'd recommend the simplest solution that works:</b> Sometimes that's a SQL query. Sometimes a dashboard. Sometimes ML. The goal is to solve the problem, not to use fancy technology.",
         "t": "This tests business sense, not technical skill. Show you start with the problem, not the solution."},
    ])


# ═══════════════════════════════════════
# M1: DATA LIFECYCLE
# ═══════════════════════════════════════
elif module == "🔄 M1: Data Lifecycle":
    st.markdown("# 🔄 Module 1: Data Lifecycle")
    st.caption("Week 1 · Where data comes from, how it flows, and what can go wrong.")

    st.markdown("""<div class="story-box">
    Before you analyze data, you need to understand its <b>journey</b>. Data doesn't appear magically in a CSV —
    it's collected, stored, transformed, and delivered. At each step, things can go wrong.
    </div>""", unsafe_allow_html=True)

    stages = [
        ("1️⃣", "Collection", "#7c6aff", "Data is generated — sensors, user clicks, surveys, transactions.",
         "Sensors malfunction, users skip fields, systems go offline. <b>Garbage in → garbage out.</b>"),
        ("2️⃣", "Storage", "#22d3a7", "Data is saved — databases, data lakes, CSV files, APIs.",
         "Schema changes break pipelines, storage limits cause data loss, encoding issues corrupt text."),
        ("3️⃣", "Processing", "#f5b731", "Data is cleaned, transformed, aggregated.",
         "Joins go wrong (duplicates), timezone mismatches, aggregation hides important details."),
        ("4️⃣", "Analysis", "#f45d6d", "Patterns are found, hypotheses tested, models built.",
         "Confirmation bias (seeing what you want), overfitting, wrong statistical test."),
        ("5️⃣", "Action", "#e879a8", "Decisions are made based on insights.",
         "Correlation mistaken for causation, insights not communicated clearly, stakeholders ignore data."),
    ]

    for num, title, color, desc, risk in stages:
        st.markdown(f"""<div class="story-box" style="border-left:4px solid {color}">
        <b>{num} {title}</b> — {desc}
        <br><span style="color:#d8a8b8">⚠️ What can go wrong: {risk}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="green-box">
    👉 <b>Telecom example:</b> A cell tower sends performance data (collection) → stored in a database (storage)
    → aggregated hourly (processing) → analyst spots a pattern (analysis) → network team fixes the issue (action).
    If the sensor was miscalibrated at step 1, every downstream step is wrong.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🎮 Try It: Spot the Data Quality Issues")
    st.caption("This is a sample of raw telecom data. Can you spot the problems?")

    bad_data = pd.DataFrame({
        "customer_id": [1, 2, 3, 3, 5, 6, 7, 8, 9, 10],
        "age": [25, -5, 35, 35, 150, 42, None, 28, 33, 29],
        "monthly_charges": [50.0, 65.0, None, 72.0, 72.0, 85.0, 45.0, 999.99, 55.0, 60.0],
        "contract": ["Monthly", "Annual", "Monthly", "Monthly", "annual", "Monthly", "2-Year", "Monthly", "monthly", "Annual"],
        "signup_date": ["2023-01-15", "2023-02-30", "2023-03-10", "2023-03-10", "2023-04-01", "2023-05-15", "2023-06-01", "2023-07-20", "2023-08-10", "2023-09-01"],
    })
    st.dataframe(bad_data, use_container_width=True)

    with st.expander("🔍 Reveal the issues"):
        issues = [
            ("Row 2", "age = -5", "Impossible value — age can't be negative", "#f45d6d"),
            ("Row 3 & 4", "Duplicate customer_id = 3", "Same customer appears twice", "#f5b731"),
            ("Row 5", "age = 150", "Outlier — nobody is 150 years old", "#f45d6d"),
            ("Row 7", "age = None", "Missing value", "#7c6aff"),
            ("Row 3", "monthly_charges = None", "Missing value", "#7c6aff"),
            ("Row 8", "monthly_charges = 999.99", "Suspicious — likely a placeholder or error", "#f5b731"),
            ("Row 2", "signup_date = 2023-02-30", "Invalid date — February doesn't have 30 days", "#f45d6d"),
            ("Multiple", "contract: Monthly vs monthly vs annual vs Annual", "Inconsistent casing — same value written differently", "#f5b731"),
        ]
        for row, field, desc, color in issues:
            st.markdown(f"""<div class="key-box" style="border-left-color:{color}"><b>{row}:</b> {field} — {desc}</div>""", unsafe_allow_html=True)

    iq([
        {"q": "Walk me through how you'd approach a new dataset you've never seen before.", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "<b>1. Shape & structure:</b> df.shape, df.dtypes, df.head(). How many rows/columns? What types? <b>2. Missing values:</b> df.isna().sum(). How much is missing? Which columns? <b>3. Distributions:</b> df.describe() for numeric, df.value_counts() for categorical. Any weird values? <b>4. Duplicates:</b> df.duplicated().sum(). <b>5. Relationships:</b> Correlation matrix, scatter plots. <b>6. Domain sense-check:</b> Do the numbers make sense? Age=150? Price=-50? <b>7. Formulate questions:</b> What patterns do I see? What's surprising?",
         "t": "Interviewers want a structured approach, not 'I'd just look at it.' Have a checklist."},
        {"q": "What is 'garbage in, garbage out' and how do you prevent it?", "d": "Easy", "c": ["Amazon", "General"],
         "a": "If your input data is bad (errors, bias, missing values), your output (analysis, model) will be bad too — no matter how sophisticated your methods. <b>Prevention:</b> (1) Validate data at collection (input constraints, range checks). (2) Automated quality checks in pipelines. (3) EDA before modeling. (4) Monitor data drift in production. (5) Document data sources and known issues.",
         "t": "Give a concrete example: 'A model trained on data where age=-5 will learn nonsense.'"},
        {"q": "What's the difference between structured and unstructured data?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Structured:</b> Organized in rows and columns (databases, CSVs, spreadsheets). Easy to query and analyze. Example: customer table with name, age, purchase amount. <b>Unstructured:</b> No predefined format — text, images, audio, video, logs. Harder to analyze, requires NLP/CV techniques. Example: customer reviews, support call recordings. <b>Semi-structured:</b> Has some organization but not rigid — JSON, XML, logs with patterns.",
         "t": "Mention that ~80% of real-world data is unstructured — that's why NLP and computer vision are so important."},
    ])


# ═══════════════════════════════════════
# M2: DATA CLEANING
# ═══════════════════════════════════════
elif module == "🧹 M2: Data Cleaning":
    st.markdown("# 🧹 Module 2: Data Cleaning")
    st.caption("Week 1–2 · Making messy data usable — the most time-consuming part of data science.")

    st.markdown("""<div class="story-box">
    Data scientists spend <b>60–80% of their time</b> cleaning data. It's not glamorous, but it's where
    the real value is. A clean dataset with a simple model beats a dirty dataset with a fancy model every time.
    </div>""", unsafe_allow_html=True)

    df = get_telecom_data(500)

    # ── Missing Data ──
    st.markdown("### 🕳️ Handling Missing Data")
    st.markdown("""<div class="story-box">
    Missing data isn't just "empty cells." Every blank has a reason — and that reason matters.
    <br><br><b>Three types:</b>
    <br>• <b>MCAR:</b> Missing completely at random (coffee spill on a form)
    <br>• <b>MAR:</b> Missing depends on other columns (young people skip income question)
    <br>• <b>MNAR:</b> Missing depends on the value itself (rich people hide income) — hardest to fix
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Fix the Missing Ages")
    missing_count = int(df["age"].isna().sum())
    st.markdown(f"**{missing_count} out of {len(df)}** customers have missing ages.")

    strategy = st.radio("Pick a fix:", ["Drop missing rows", "Fill with mean", "Fill with median", "Fill with mode"], horizontal=True, key="p2_miss")
    df_fixed = df.copy()
    if strategy == "Drop missing rows":
        df_fixed = df_fixed.dropna(subset=["age"])
    elif strategy == "Fill with mean":
        df_fixed["age"] = df_fixed["age"].fillna(df_fixed["age"].mean())
    elif strategy == "Fill with median":
        df_fixed["age"] = df_fixed["age"].fillna(df_fixed["age"].median())
    else:
        df_fixed["age"] = df_fixed["age"].fillna(df_fixed["age"].mode()[0])

    mc = st.columns(4)
    mc[0].metric("Before — Rows", len(df))
    mc[1].metric("After — Rows", len(df_fixed))
    mc[2].metric("Before — Mean Age", f"{df['age'].mean():.1f}")
    mc[3].metric("After — Mean Age", f"{df_fixed['age'].mean():.1f}", f"{df_fixed['age'].mean()-df['age'].mean():+.1f}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["age"].dropna(), name="Before", marker_color="#7c6aff", opacity=0.4, nbinsx=20))
    fig.add_trace(go.Histogram(x=df_fixed["age"], name="After", marker_color="#22d3a7", opacity=0.6, nbinsx=20))
    fig.update_layout(barmode="overlay", height=280, title="Age Distribution: Before vs After Fix", **DL)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Outliers ──
    st.markdown("### 📏 Detecting Outliers")
    col = st.selectbox("Pick a column:", ["monthly_charges", "total_charges", "support_tickets", "tenure_months"], key="p2_out_col")
    data_col = df[col].dropna()
    Q1, Q3 = data_col.quantile(0.25), data_col.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = data_col[(data_col < lower) | (data_col > upper)]

    mc = st.columns(4)
    mc[0].metric("Q1", f"{Q1:.1f}")
    mc[1].metric("Q3", f"{Q3:.1f}")
    mc[2].metric("IQR", f"{IQR:.1f}")
    mc[3].metric("Outliers Found", len(outliers))

    fig_box = go.Figure(go.Box(x=data_col, marker_color="#22d3a7", boxmean=True, name=col))
    fig_box.update_layout(height=180, **{k: v for k, v in DL.items() if k != 'yaxis'}, yaxis=dict(visible=False))
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    # ── Duplicates ──
    st.markdown("### 🔁 Handling Duplicates")
    st.markdown("""<div class="story-box">
    Duplicates inflate your dataset and bias your analysis. A customer counted twice gets double the weight.
    <br><br><b>Types:</b> Exact duplicates (identical rows) vs near-duplicates (same person, slightly different data — "John Smith" vs "john smith").
    </div>""", unsafe_allow_html=True)

    st.markdown(f"**Exact duplicates in this dataset:** {df.duplicated().sum()}")

    iq([
        {"q": "You get a dataset with 30% missing values in a key column. What do you do?", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "<b>1. Understand why:</b> Is it MCAR/MAR/MNAR? Check if missingness correlates with other columns. <b>2. Assess impact:</b> 30% is a lot — dropping loses too much data. <b>3. Strategy:</b> Numeric → median (robust to outliers). Categorical → mode. If relationships exist → model-based imputation (KNN). <b>4. Create a flag:</b> Add 'is_missing' binary column — missingness itself can be informative. <b>5. Validate:</b> Compare distributions before/after.",
         "t": "Never say 'just drop them' for 30%. Show you think about WHY data is missing."},
        {"q": "How do you handle outliers in a dataset?", "d": "Medium", "c": ["Amazon", "Apple"],
         "a": "<b>1. Detect:</b> IQR method, Z-score, box plots. <b>2. Investigate:</b> Is it an error? Fraud? Rare but real? <b>3. Decide:</b> Error → remove. Fraud → keep (it's the signal). Rare but real → cap/winsorize or use robust methods. <b>4. Document:</b> Always record what you removed and why. <b>Key:</b> Never blindly remove — always investigate first.",
         "t": "Say 'I'd investigate before removing' — shows judgment, not just technique."},
        {"q": "What's the difference between data cleaning and data preprocessing?", "d": "Easy", "c": ["General"],
         "a": "<b>Cleaning:</b> Fixing errors — missing values, duplicates, wrong types, impossible values. Making data <b>correct</b>. <b>Preprocessing:</b> Transforming correct data for modeling — scaling, encoding, normalization, train/test split. Making data <b>model-ready</b>. Cleaning comes first, preprocessing comes second.",
         "t": "Simple distinction: cleaning = fix errors, preprocessing = prepare for models."},
        {"q": "How would you validate that your data cleaning didn't introduce bias?", "d": "Hard", "c": ["Google", "Netflix"],
         "a": "<b>1. Compare distributions:</b> Plot before/after histograms for key columns. <b>2. Check statistics:</b> Mean, median, std should be similar (unless you intentionally changed them). <b>3. Segment analysis:</b> Did cleaning affect some groups more than others? (e.g., did dropping missing rows remove mostly young customers?) <b>4. Downstream check:</b> Train a model with and without cleaning — does performance change unexpectedly? <b>5. Domain review:</b> Have a domain expert sanity-check the cleaned data.",
         "t": "Mentioning 'segment analysis' shows you think about fairness and bias — very valued at FAANG."},
    ])


# ═══════════════════════════════════════
# M3: FEATURE ENGINEERING
# ═══════════════════════════════════════
elif module == "🏗️ M3: Feature Engineering":
    st.markdown("# 🏗️ Module 3: Feature Engineering")
    st.caption("Week 2 · Creating new columns that reveal hidden patterns.")

    st.markdown("""<div class="story-box">
    A doctor has your <b>height</b> and <b>weight</b>. Useful, but limited. Combine them into <b>BMI</b>
    (weight / height²) and suddenly you have a powerful health indicator. That's feature engineering —
    transforming raw data into something more meaningful.
    <br><br><b>"Better features beat better algorithms."</b> — Every senior data scientist
    </div>""", unsafe_allow_html=True)

    df = get_telecom_data(500)

    # ── Types of features ──
    st.markdown("### 📚 Types of Features You Can Create")
    types = [
        ("💰 Ratios", "Combine two numbers into a meaningful ratio", "revenue_per_employee, cost_per_click, charges_per_month_of_tenure"),
        ("📦 Binning", "Turn continuous numbers into categories", "age → Young/Adult/Senior, tenure → New/Mid/Loyal"),
        ("📅 DateTime", "Extract time components from dates", "day_of_week, month, is_weekend, days_since_signup"),
        ("🏷️ Encoding", "Turn categories into numbers", "One-hot: Red/Blue/Green → three 0/1 columns"),
        ("✖️ Interactions", "Multiply features together", "monthly_charges × support_tickets = frustration_score"),
        ("🚩 Flags", "Create binary indicators", "is_high_spender, has_complained, is_month_to_month"),
    ]
    for emoji, desc, example in types:
        st.markdown(f"""<div class="key-box"><b>{emoji}</b> — {desc}<br><span style="color:#8892b0">Example: {example}</span></div>""", unsafe_allow_html=True)

    # ── Interactive feature builder ──
    st.markdown("### 🎮 Try It: Create Features and See Their Impact on Churn")
    st.caption("Toggle features on/off and see which ones correlate most with customer churn.")

    c1, c2, c3 = st.columns(3)
    add_ratio = c1.checkbox("💰 Charges per tenure month", key="p2_fe1")
    add_flag = c2.checkbox("🚩 Month-to-month flag", key="p2_fe2")
    add_interact = c3.checkbox("✖️ Charges × Tickets", key="p2_fe3")
    c4, c5, c6 = st.columns(3)
    add_bin = c4.checkbox("📦 Tenure group (New/Mid/Loyal)", key="p2_fe4")
    add_high = c5.checkbox("🚩 High spender flag (top 25%)", key="p2_fe5")
    add_ticket_rate = c6.checkbox("💰 Tickets per tenure month", key="p2_fe6")

    if add_ratio:
        df["charges_per_tenure"] = (df["monthly_charges"] / df["tenure_months"].clip(1)).round(2)
    if add_flag:
        df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)
    if add_interact:
        df["charges_x_tickets"] = df["monthly_charges"] * df["support_tickets"]
    if add_bin:
        df["is_new_customer"] = (df["tenure_months"] <= 12).astype(int)
    if add_high:
        df["high_spender"] = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)
    if add_ticket_rate:
        df["ticket_rate"] = (df["support_tickets"] / df["tenure_months"].clip(1) * 12).round(3)

    # Show correlations with churn
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "customer_id" in num_cols:
        num_cols.remove("customer_id")
    corr_with_churn = df[num_cols].corr()["churned"].drop("churned").sort_values(key=abs, ascending=False)

    orig_cols = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "age"]
    new_cols = [c for c in corr_with_churn.index if c not in orig_cols]
    colors = ['#f5b731' if c in new_cols else '#7c6aff' for c in corr_with_churn.index]

    fig = go.Figure(go.Bar(
        x=corr_with_churn.index, y=corr_with_churn.values,
        marker=dict(color=colors, cornerradius=6),
        text=[f"{v:.3f}" for v in corr_with_churn.values], textposition="outside",
        textfont=dict(color="#8892b0", size=9),
    ))
    fig.update_layout(height=380, title="Correlation with Churn (🟡 = New Features, 🟣 = Original)", **DL)
    fig.update_yaxes(range=[-0.5, 0.5])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if new_cols:
        best = max(new_cols, key=lambda c: abs(corr_with_churn[c]))
        st.markdown(f"""<div class="green-box">💡 Best new feature: <b>{best}</b> (r = {corr_with_churn[best]:.3f}). {'Stronger than any original feature!' if abs(corr_with_churn[best]) > abs(corr_with_churn[orig_cols].values[0]) else 'Try toggling more features!'}</div>""", unsafe_allow_html=True)
    else:
        st.info("👆 Toggle some features above to see their impact.")

    iq([
        {"q": "What is feature engineering and why is it important?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "Creating new input variables from raw data to improve model performance. <b>Why:</b> Models can only learn from features you give them. Raw data often doesn't capture real patterns. Example: 'date_built' is useless, but 'house_age_years' is predictive. <b>The saying:</b> 'Better features beat better algorithms.'",
         "t": "Give a concrete before/after example."},
        {"q": "How do you handle categorical variables with 10,000 unique values?", "d": "Hard", "c": ["Google", "Meta", "Airbnb"],
         "a": "One-hot encoding creates 10K columns — impractical. <b>Better:</b> (1) Target encoding (replace with avg target value — watch for leakage). (2) Frequency encoding (replace with count). (3) Group into clusters (10K zips → 50 regions). (4) Hash encoding (fixed buckets). (5) Embedding layers (deep learning).",
         "t": "Mention target encoding first and always flag the leakage risk."},
        {"q": "What's the difference between label encoding and one-hot encoding?", "d": "Easy", "c": ["Amazon", "Google"],
         "a": "<b>Label:</b> Red=0, Blue=1, Green=2. Implies order (Green > Blue). Fine for ordinal data or tree models. <b>One-hot:</b> Red=[1,0,0], Blue=[0,1,0]. No false ordering. Required for linear models, neural nets. <b>Rule:</b> Ordinal → label. Nominal → one-hot. Trees → either works.",
         "t": "Mention tree-based models handle label encoding fine even for nominal data."},
        {"q": "How do you decide which features to keep?", "d": "Medium", "c": ["Meta", "Apple", "Netflix"],
         "a": "<b>1.</b> Correlation with target. <b>2.</b> Feature importance from tree models. <b>3.</b> Mutual information (captures non-linear). <b>4.</b> RFE (recursive elimination). <b>5.</b> Domain knowledge. <b>6.</b> Drop collinear features (r > 0.9). <b>Also check:</b> Is any feature 'too good'? Might be data leakage.",
         "t": "Mention data leakage check — shows you think about production, not just notebooks."},
        {"q": "Give me 5 features you'd create from a 'signup_date' column.", "d": "Medium", "c": ["Amazon", "Netflix", "Uber"],
         "a": "<b>1.</b> days_since_signup (tenure). <b>2.</b> signup_month (seasonal patterns). <b>3.</b> signup_day_of_week (weekday vs weekend signups). <b>4.</b> is_recent_signup (< 30 days). <b>5.</b> signup_quarter (Q1-Q4 trends). <b>Bonus:</b> signup_year (cohort analysis), is_holiday_signup, days_until_next_billing.",
         "t": "Show creativity beyond the obvious 'tenure' feature."},
    ])


# ═══════════════════════════════════════
# M4: DATA VISUALIZATION
# ═══════════════════════════════════════
elif module == "📊 M4: Data Visualization":
    st.markdown("# 📊 Module 4: Data Visualization")
    st.caption("Week 2–3 · Telling stories with charts.")

    st.markdown("""<div class="story-box">
    A table of 10,000 numbers is meaningless. A chart of those same numbers can reveal patterns in seconds.
    Visualization is how you <b>communicate insights</b> — to yourself during exploration, and to stakeholders
    when presenting results.
    <br><br><b>The goal:</b> Every chart should answer a specific question. If it doesn't, it's decoration.
    </div>""", unsafe_allow_html=True)

    df = get_telecom_data(500)

    # ── Chart picker ──
    st.markdown("### 🎮 Interactive Chart Builder")
    st.caption("Pick a chart type, select columns, and see the result. Learn when to use each chart.")

    chart = st.radio("Chart type:", ["📊 Histogram", "📦 Box Plot", "🔵 Scatter Plot", "📊 Bar Chart", "🗺️ Heatmap"], horizontal=True, key="p2_viz")

    num_cols = ["monthly_charges", "total_charges", "tenure_months", "support_tickets", "age"]
    cat_cols = ["contract", "churned"]

    if chart == "📊 Histogram":
        st.markdown("""<div class="key-box"><b>When to use:</b> See the distribution (shape) of a single numeric variable. "How are monthly charges distributed?"</div>""", unsafe_allow_html=True)
        col = st.selectbox("Column:", num_cols, key="p2_hist_col")
        hue = st.selectbox("Color by:", [None] + cat_cols, key="p2_hist_hue")
        bins = st.slider("Number of bins:", 5, 60, 25, key="p2_hist_bins")
        data_col = df[col].dropna()
        fig = go.Figure()
        if hue:
            for val in df[hue].unique():
                fig.add_trace(go.Histogram(x=df[df[hue]==val][col].dropna(), name=str(val), opacity=0.6, nbinsx=bins))
            fig.update_layout(barmode="overlay")
        else:
            fig.add_trace(go.Histogram(x=data_col, nbinsx=bins, marker_color="#22d3a7", opacity=0.7))
        fig.add_vline(x=data_col.mean(), line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: {data_col.mean():.1f}")
        fig.update_layout(height=350, title=f"Distribution of {col}", **DL)

    elif chart == "📦 Box Plot":
        st.markdown("""<div class="key-box"><b>When to use:</b> Compare distributions across groups AND spot outliers. "How do charges differ by contract type?"</div>""", unsafe_allow_html=True)
        val_col = st.selectbox("Value:", num_cols, key="p2_box_val")
        grp_col = st.selectbox("Group by:", cat_cols, key="p2_box_grp")
        fig = px.box(df, x=grp_col, y=val_col, color=grp_col, color_discrete_sequence=["#7c6aff", "#22d3a7", "#f5b731"])
        fig.update_layout(height=350, **DL)

    elif chart == "🔵 Scatter Plot":
        st.markdown("""<div class="key-box"><b>When to use:</b> See the relationship between two numeric variables. "Does tenure relate to total charges?"</div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X axis:", num_cols, index=0, key="p2_sc_x")
        y_col = c2.selectbox("Y axis:", num_cols, index=1, key="p2_sc_y")
        hue = st.selectbox("Color by:", [None] + cat_cols, key="p2_sc_hue")
        fig = px.scatter(df, x=x_col, y=y_col, color=hue, opacity=0.6,
                         color_discrete_sequence=["#7c6aff", "#22d3a7", "#f5b731"])
        r = df[[x_col, y_col]].dropna().corr().iloc[0, 1]
        fig.update_layout(height=380, title=f"{x_col} vs {y_col} (r = {r:.3f})", **DL)

    elif chart == "📊 Bar Chart":
        st.markdown("""<div class="key-box"><b>When to use:</b> Compare a metric across categories. "What's the average charges by contract type?"</div>""", unsafe_allow_html=True)
        grp = st.selectbox("Group by:", cat_cols, key="p2_bar_grp")
        val = st.selectbox("Value:", num_cols, key="p2_bar_val")
        agg = st.radio("Aggregation:", ["mean", "median", "sum", "count"], horizontal=True, key="p2_bar_agg")
        grouped = df.groupby(grp)[val].agg(agg).sort_values(ascending=False).reset_index()
        fig = px.bar(grouped, x=grp, y=val, color=grp, text=val,
                     color_discrete_sequence=["#7c6aff", "#22d3a7", "#f5b731"])
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=350, **DL)

    else:  # Heatmap
        st.markdown("""<div class="key-box"><b>When to use:</b> See correlations between ALL numeric variables at once. "Which features are related?"</div>""", unsafe_allow_html=True)
        sel = st.multiselect("Columns:", num_cols, default=num_cols[:4], key="p2_hm_cols")
        if len(sel) >= 2:
            corr = df[sel].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=[[0,'#f45d6d'],[0.5,'#1a1d2e'],[1,'#22d3a7']], zmin=-1, zmax=1,
                text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=12)))
            fig.update_layout(height=380, **DL)
        else:
            fig = go.Figure()
            st.warning("Select at least 2 columns.")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Chart selection guide ──
    st.markdown("### 📋 Which Chart When?")
    guide = pd.DataFrame({
        "Question": ["What's the distribution?", "Any outliers?", "How do groups compare?", "Are two things related?", "Which features correlate?"],
        "Chart": ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot", "Heatmap"],
        "Example": ["Monthly charges distribution", "Outliers in support tickets", "Avg charges by contract", "Tenure vs total charges", "All numeric correlations"],
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)

    iq([
        {"q": "When would you use a histogram vs a bar chart?", "d": "Easy", "c": ["Google", "Meta"],
         "a": "<b>Histogram:</b> Distribution of a single NUMERIC variable (continuous). Bins show frequency. Example: distribution of ages. <b>Bar chart:</b> Comparing a metric across CATEGORIES. Each bar = one category. Example: average revenue by region. <b>Key:</b> Histogram = shape of one variable. Bar chart = comparison across groups.",
         "t": "This seems basic but many candidates confuse them. Be precise."},
        {"q": "How do you choose the right visualization for your data?", "d": "Medium", "c": ["Amazon", "Netflix"],
         "a": "<b>Start with the question:</b> (1) Distribution? → Histogram/KDE. (2) Comparison? → Bar/Box. (3) Relationship? → Scatter. (4) Composition? → Pie/Stacked bar. (5) Trend over time? → Line chart. (6) All correlations? → Heatmap. <b>Then consider audience:</b> Technical → detailed plots. Executives → simple, clear, one insight per chart.",
         "t": "Say 'I start with the question I'm trying to answer' — shows structured thinking."},
        {"q": "What makes a good data visualization?", "d": "Medium", "c": ["Google", "Apple"],
         "a": "<b>1. Answers a specific question</b> (not just 'looks cool'). <b>2. Clear title and labels</b> (someone should understand it without explanation). <b>3. Right chart type</b> for the data. <b>4. No clutter</b> (remove gridlines, legends, colors that don't add meaning). <b>5. Highlights the insight</b> (use color/annotation to draw attention to the key finding). <b>6. Honest</b> (no truncated axes, no misleading scales).",
         "t": "Mention 'honest' — truncated Y-axes are a common trick that good data scientists avoid."},
        {"q": "How would you present a correlation heatmap to a non-technical audience?", "d": "Medium", "c": ["Meta", "Airbnb"],
         "a": "I wouldn't show the full heatmap — it's overwhelming. Instead: <b>1.</b> Extract the top 3-5 strongest correlations. <b>2.</b> Show them as simple scatter plots with trend lines. <b>3.</b> Use plain language: 'Customers who pay more per month tend to churn more (r=0.65).' <b>4.</b> Add business context: 'This suggests our pricing might be driving customers away.' <b>5.</b> End with a recommendation, not just a finding.",
         "t": "Show you think about the audience, not just the analysis."},
    ])
