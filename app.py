import streamlit as st

st.set_page_config(page_title="🧭 AI & DS Learning Roadmap", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.phase-card {
    background: linear-gradient(135deg, #1a1d2e, #252840);
    border: 1px solid #2d3148; border-radius: 20px; padding: 1.8rem;
    border-top: 4px solid; min-height: 260px;
}
.phase-num { font-size: 2.5rem; font-weight: 800; opacity: 0.12; }
.phase-title { font-size: 1.2rem; font-weight: 700; margin: 0.3rem 0; }
.phase-desc { font-size: 0.84rem; color: #8892b0; line-height: 1.6; }
.phase-weeks { font-size: 0.75rem; font-weight: 600; margin-top: 0.8rem; }
.advice-box {
    background: #181a27; border: 1px solid #2d3148; border-radius: 14px;
    padding: 1.2rem 1.5rem; border-left: 4px solid; margin-bottom: 0.6rem;
    color: #c8cfe0; font-size: 0.92rem; line-height: 1.7;
}
.advice-box b { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧭 Complete AI & Data Science Learning Roadmap")
st.caption("A structured path from statistics foundations to building AI agents — with interactive visuals, real-world stories, and FAANG interview prep at every step.")

st.divider()

phases = [
    ("01", "📊 Statistics Foundation", "3–4 Weeks", "#7c6aff",
     "Descriptive stats, probability, distributions, hypothesis testing, correlation, regression intuition.",
     "✅ Available"),
    ("02", "📊 Data Science", "2–3 Weeks", "#22d3a7",
     "Data lifecycle, cleaning, feature engineering, visualization, EDA — turning raw data into insights.",
     "✅ Available"),
    ("03", "🤖 Machine Learning", "4–6 Weeks", "#f5b731",
     "Supervised learning, regression, classification, decision trees, random forest, overfitting, bias-variance.",
     "🔒 Coming Soon"),
    ("04", "🧠 Artificial Intelligence", "2–3 Weeks", "#f45d6d",
     "AI vs ML vs DL, rule-based vs learning systems, search, planning, optimization, decision making.",
     "🔒 Coming Soon"),
    ("05", "💬 LLMs & Generative AI", "4–6 Weeks", "#e879a8",
     "Tokens, embeddings, transformers, prompt engineering, RAG, vector search, hallucination, temperature.",
     "🔒 Coming Soon"),
    ("06", "🤖 Agents", "2–3 Weeks", "#5eaeff",
     "Autonomous AI systems, tools & actions, planning vs execution, multi-step reasoning, orchestration.",
     "🔒 Coming Soon"),
]

cols = st.columns(3)
for i, (num, title, weeks, color, desc, status) in enumerate(phases):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="phase-card" style="border-top-color:{color}">
            <div class="phase-num" style="color:{color}">PHASE {num}</div>
            <div class="phase-title">{title}</div>
            <div class="phase-desc">{desc}</div>
            <div class="phase-weeks" style="color:{color}">⏱ {weeks} · {status}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

st.divider()

# Final outcome
st.markdown("### 🎯 After Completing All 6 Phases")
outcomes = st.columns(3)
for col, (icon, text) in zip(outcomes, [
    ("📊", "Understand statistics deeply & think like a data scientist"),
    ("🤖", "Understand ML models, LLMs & AI agents intuitively"),
    ("🚀", "Be ready to build real AI systems & ace interviews"),
]):
    col.markdown(f"""<div class="advice-box" style="border-left-color:#22d3a7;text-align:center">
    <span style="font-size:2rem">{icon}</span><br>{text}
    </div>""", unsafe_allow_html=True)

st.divider()

# Key advice
st.markdown("### ⚡ Key Advice")
advices = [
    ("#7c6aff", "🧠 Don't rush statistics", "This is your foundation — every ML concept, every model, every interview question traces back to stats. Take it seriously."),
    ("#22d3a7", "💡 Focus on intuition", "Always ask: <b>\"What does this mean in real life?\"</b> Formulas are tools, not goals. If you can explain it to a non-technical person, you truly understand it."),
    ("#f5b731", "🔗 Connect to your domain", "Every concept becomes 10× clearer when you tie it to real problems — telecom KPIs, network latency, customer churn, system behavior."),
]
for color, title, desc in advices:
    st.markdown(f"""<div class="advice-box" style="border-left-color:{color}">
    <b>{title}</b><br>{desc}
    </div>""", unsafe_allow_html=True)

st.caption("👈 Use the sidebar to navigate to each phase.")
