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
.story-box {
    background: linear-gradient(135deg, #1a1d2e, #1f2235);
    border: 1px solid #2d3148; border-radius: 16px; padding: 1.4rem 1.6rem;
    margin: 0.8rem 0; line-height: 1.9; font-size: 0.95rem; color: #c8cfe0;
}
.story-box b, .story-box strong { color: #e2e8f0; }
.green-box {
    background: linear-gradient(135deg, #1a2a1f, #1f3528);
    border: 1px solid #2a5a3a; border-radius: 14px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; font-size: 0.92rem; color: #c8d8c0; line-height: 1.8;
}
.green-box b { color: #d0f0e0; }
.red-box {
    background: linear-gradient(135deg, #2a1a1e, #351f25);
    border: 1px solid #5a2a3a; border-radius: 14px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; font-size: 0.92rem; color: #d8a8b8; line-height: 1.8;
}
.red-box b { color: #f0c8d8; }
.key-box {
    background: #252840; border-left: 4px solid #7c6aff;
    border-radius: 0 10px 10px 0; padding: 0.8rem 1.2rem;
    margin: 0.6rem 0; font-size: 0.9rem; color: #c8cfe0; line-height: 1.8;
}
.key-box b { color: #e2e8f0; }
.analogy-box {
    background: linear-gradient(135deg, #1e1a2e, #251f35);
    border: 1px solid #3d2d58; border-radius: 14px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; font-size: 0.92rem; color: #d0c8e0; line-height: 1.8;
    border-left: 4px solid #e879a8;
}
.analogy-box b { color: #f0d8e8; }
.quiz-correct {
    background: linear-gradient(135deg, #1a2a1f, #1f3528);
    border: 2px solid #22d3a7; border-radius: 14px; padding: 1rem 1.3rem;
    margin: 0.5rem 0; font-size: 0.92rem; color: #c8d8c0;
}
.quiz-wrong {
    background: linear-gradient(135deg, #2a1a1e, #351f25);
    border: 2px solid #f45d6d; border-radius: 14px; padding: 1rem 1.3rem;
    margin: 0.5rem 0; font-size: 0.92rem; color: #d8a8b8;
}
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


def check_quiz(key, question, options, correct_idx, explanation):
    """Reusable quiz component with drag-and-drop feel."""
    st.markdown(f"**🧩 Quick Check:** {question}")
    answer = st.radio("Pick your answer:", options, key=key, index=None)
    if answer is not None:
        if options.index(answer) == correct_idx:
            st.markdown(f'<div class="quiz-correct">✅ <b>Correct!</b> {explanation}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="quiz-wrong">❌ <b>Not quite.</b> The answer is <b>{options[correct_idx]}</b>. {explanation}</div>', unsafe_allow_html=True)


def match_quiz(key, title, items, targets, correct_mapping, explanations):
    """Matching quiz — user matches items to categories using dropdowns."""
    st.markdown(f"**🔗 Match It:** {title}")
    st.caption("Use the dropdowns to match each item to the right category.")
    all_correct = True
    for i, item in enumerate(items):
        col1, col2 = st.columns([2, 2])
        col1.markdown(f"**{item}**")
        choice = col2.selectbox(f"Match for {item}", ["— Pick one —"] + targets, key=f"{key}_{i}", label_visibility="collapsed")
        if choice != "— Pick one —":
            if choice == correct_mapping[i]:
                st.markdown(f'<div class="quiz-correct" style="padding:0.5rem 1rem;margin:0.2rem 0">✅ {explanations[i]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="quiz-wrong" style="padding:0.5rem 1rem;margin:0.2rem 0">❌ Try again! {explanations[i]}</div>', unsafe_allow_html=True)
                all_correct = False


def order_quiz(key, title, items, correct_order, explanation):
    """Ordering quiz — user arranges items in the right order using number inputs."""
    st.markdown(f"**📋 Put It In Order:** {title}")
    st.caption("Assign a number (1, 2, 3...) to each item to put them in the right order.")
    user_order = {}
    for i, item in enumerate(items):
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"**{item}**")
        user_order[item] = col2.number_input(f"Order for {item}", min_value=1, max_value=len(items), value=i+1, key=f"{key}_{i}", label_visibility="collapsed")

    if st.button("Check my order", key=f"{key}_check"):
        sorted_items = sorted(user_order.keys(), key=lambda x: user_order[x])
        if sorted_items == correct_order:
            st.markdown(f'<div class="quiz-correct">✅ <b>Perfect!</b> {explanation}</div>', unsafe_allow_html=True)
        else:
            correct_str = " → ".join(f"**{x}**" for x in correct_order)
            st.markdown(f'<div class="quiz-wrong">❌ <b>Not quite.</b> The correct order is: {correct_str}. {explanation}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════
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
    st.caption("3–4 Weeks · Understand data, uncertainty, and decision-making — no math degree needed.")

    st.markdown("""<div class="story-box">
    <b>🎬 Imagine this:</b> You just got hired at a company. Your boss walks in and says,
    "We have 10 million users. Are they happy? Is our product getting better or worse? Should we launch this new feature?"
    <br><br>
    You can't talk to 10 million people. But you CAN look at the <b>data</b> — and statistics is the language
    that lets you read it, understand it, and make smart decisions from it.
    <br><br>
    <b>Statistics is not about formulas.</b> It's about asking the right questions and knowing when the data
    is telling you something real vs. when it's just noise.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="analogy-box">
    🍕 <b>Think of it like cooking:</b> You don't need to be a chemist to cook great food.
    But understanding a few basics — heat, timing, seasoning — makes you 10x better.
    Statistics is the "cooking basics" of data. Once you get it, everything else (ML, AI, data science) becomes way easier.
    </div>""", unsafe_allow_html=True)

    modules = [
        ("🧩", "Descriptive Statistics", "Week 1", "#7c6aff", "How to summarize thousands of numbers into a few meaningful ones. Like reading the highlights instead of the whole book."),
        ("🎲", "Probability", "Week 1–2", "#22d3a7", "How likely is something to happen? The math behind predictions, risk, and 'what are the odds?'"),
        ("📈", "Distributions", "Week 2", "#f5b731", "What shape does your data make? Bell curves, lopsided data, and why the shape matters."),
        ("🧪", "Inferential Statistics", "Week 3", "#f45d6d", "Is this result real or just a coincidence? The science of making conclusions from limited data."),
        ("🔗", "Correlation", "Week 3", "#e879a8", "Do two things move together? And the biggest trap: just because they do doesn't mean one causes the other."),
        ("📉", "Regression Intuition", "Week 4", "#5eaeff", "Drawing the best line through data to make predictions. This is where statistics meets machine learning."),
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
    • Look at any dataset and immediately understand what's going on<br>
    • Know when a result is real vs. random luck<br>
    • Understand ML concepts easily (because they're all built on this)<br>
    • Be ready for data science interview questions at any company
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# MODULE 1: DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════
elif module == "🧩 M1: Descriptive Statistics":
    st.markdown("# 🧩 Module 1: Descriptive Statistics")
    st.caption("Week 1 · Summarize data in a few meaningful numbers — like a movie trailer for your dataset.")

    # ── The Story ──
    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> You're a manager at a pizza chain with 500 stores. Every store reports daily sales.
    You can't read 500 numbers every morning. You need a <b>summary</b> — a few numbers that tell you:
    <br><br>
    🔹 What's a <b>typical</b> store making? (Center)<br>
    🔹 How <b>different</b> are stores from each other? (Spread)<br>
    🔹 Are there any stores that are <b>weirdly</b> high or low? (Outliers)
    <br><br>
    That's exactly what descriptive statistics does. Let's learn each tool.
    </div>""", unsafe_allow_html=True)

    # ── Mean, Median, Mode ──
    st.markdown("### 📖 The Three Averages: Mean, Median, Mode")

    st.markdown("""<div class="analogy-box">
    🏠 <b>Real-life analogy:</b> Imagine 5 friends share their monthly rent:<br>
    $800, $900, $950, $1000, <b>$8000</b> (one friend lives in a penthouse)
    <br><br>
    • <b>Mean (average):</b> Add them up, divide by 5 = <b>$2,330</b>. Sounds like everyone pays a lot — but that's misleading! The penthouse pulled the average way up.<br>
    • <b>Median (middle value):</b> Sort them, pick the middle = <b>$950</b>. This better represents what a "typical" friend pays.<br>
    • <b>Mode (most common):</b> No repeats here, but if three friends paid $900, the mode would be $900.
    <br><br>
    <b>💡 The lesson:</b> When there's an extreme value (outlier), the <b>median</b> is more trustworthy than the mean.
    That's why news reports say "median household income" not "average household income."
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It Yourself: Enter any numbers and see all three")
    user_data = st.text_input("Enter numbers separated by commas:", "25, 30, 35, 40, 45, 50, 200", key="m1_data")
    try:
        vals = np.array([float(x.strip()) for x in user_data.split(",") if x.strip()])
        from scipy import stats as sp_stats

        mc = st.columns(4)
        mc[0].metric("🔵 Mean", f"{vals.mean():.1f}", help="Add all numbers, divide by count")
        mc[1].metric("🟣 Median", f"{np.median(vals):.1f}", help="The middle number when sorted")
        mode_result = sp_stats.mode(vals, keepdims=True)
        mc[2].metric("🟠 Mode", f"{mode_result.mode[0]:.1f}", help="The most frequent number")
        gap = vals.mean() - np.median(vals)
        mc[3].metric("Gap (Mean − Median)", f"{gap:.1f}", help="Big gap = data is lopsided")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=15, marker_color="#7c6aff", opacity=0.7, name="Your data"))
        fig.add_vline(x=vals.mean(), line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: {vals.mean():.1f}")
        fig.add_vline(x=np.median(vals), line_dash="dash", line_color="#22d3a7", annotation_text=f"Median: {np.median(vals):.1f}")
        fig.update_layout(height=300, title="Your Data — Where do Mean and Median land?", xaxis_title="Value", yaxis_title="Count", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if abs(gap) > vals.std() * 0.3:
            st.markdown(f"""<div class="red-box">⚠️ <b>Notice:</b> Mean ({vals.mean():.1f}) and Median ({np.median(vals):.1f}) are far apart!
            This means your data is <b>lopsided</b> (skewed). The median is a better "typical" value here.
            Try removing the extreme number and watch them come together.</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="green-box">✅ Mean and Median are close — your data is fairly <b>balanced</b> (symmetric).</div>""", unsafe_allow_html=True)
    except:
        st.error("Please enter valid numbers separated by commas.")

    # ── Quiz: Mean vs Median ──
    check_quiz("m1_q1",
        "A company reports 'average salary is $150,000.' But most employees earn around $60,000. What's likely happening?",
        ["The company is lying", "A few executives with huge salaries are pulling the mean up", "The median must also be $150,000", "Salaries follow a bell curve"],
        1,
        "A few very high salaries (CEO, VPs) drag the mean way up, while the median stays near $60K. This is why median is better for skewed data like income."
    )

    # ── Variance & Std Dev ──
    st.markdown("---")
    st.markdown("### 📖 Spread: How Different Are the Numbers?")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> Two coffee shops both sell an average of 100 cups/day. But:
    <br><br>
    ☕ <b>Shop A:</b> Sells 95–105 cups every day. Very consistent. You can predict tomorrow easily.<br>
    ☕ <b>Shop B:</b> Some days 20 cups, some days 200. Wild swings. Hard to plan for.
    <br><br>
    Both have the <b>same average</b>, but they're completely different businesses!
    The average alone doesn't tell the full story. You need to know the <b>spread</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="analogy-box">
    🌡️ <b>Weather analogy:</b> San Francisco and Kansas City both have an average temperature of ~60°F.
    But San Francisco stays 55–65°F year-round (low spread), while Kansas City swings from 20°F to 100°F (high spread).
    Same average, totally different experience!
    <br><br>
    <b>Standard Deviation</b> measures this spread. It answers: <b>"How far does a typical value sit from the average?"</b>
    <br>• Small std dev → values are clustered close to the average (consistent)<br>
    • Large std dev → values are spread out (unpredictable)
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Same Average, Different Spread")
    c1, c2 = st.columns(2)
    sd_a = c1.slider("☕ Shop A — How consistent? (std dev)", 1, 25, 5, key="m1_sda")
    sd_b = c2.slider("☕ Shop B — How wild? (std dev)", 1, 25, 18, key="m1_sdb")

    np.random.seed(42)
    team_a = np.random.normal(100, sd_a, 300).clip(0, 200)
    team_b = np.random.normal(100, sd_b, 300).clip(0, 200)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=team_a, name=f"Shop A (std={team_a.std():.0f})", marker_color="#7c6aff", opacity=0.5, nbinsx=25))
    fig2.add_trace(go.Histogram(x=team_b, name=f"Shop B (std={team_b.std():.0f})", marker_color="#22d3a7", opacity=0.5, nbinsx=25))
    fig2.update_layout(barmode="overlay", height=300, title="Same Average (100 cups), Different Spread", xaxis_title="Cups Sold", yaxis_title="Days", **DL)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">
    💡 <b>What you're seeing:</b> Both shops average ~100 cups. But Shop A (purple) is tightly packed — predictable.
    Shop B (green) is spread out — some great days, some terrible days.
    <b>Drag the sliders</b> to see how changing the spread changes the shape!
    </div>""", unsafe_allow_html=True)

    # ── 68-95-99.7 Rule ──
    st.markdown("---")
    st.markdown("### 📖 The 68-95-99.7 Rule (The Magic Numbers)")
    st.caption("If your data makes a bell shape, these three numbers tell you almost everything.")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> Your class takes an exam. Average score = 75, std dev = 10. Without looking at every score, you already know:
    <br><br>
    🟢 <b>68%</b> of students scored between <b>65 and 85</b> (within 1 std dev)<br>
    🔵 <b>95%</b> scored between <b>55 and 95</b> (within 2 std devs)<br>
    🟣 <b>99.7%</b> scored between <b>45 and 105</b> (within 3 std devs)
    <br><br>
    Someone scored 110? That's beyond 3 std devs — only 0.3% of people score that high. Either a genius or a grading error! 🤔
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Play With It: Adjust the bell curve")
    rc1, rc2 = st.columns(2)
    rule_mean = rc1.slider("Average (μ):", 40, 100, 75, 5, key="m1_rule_mean")
    rule_std = rc2.slider("Spread (σ):", 2, 20, 10, 1, key="m1_rule_std")

    x_rule = np.linspace(rule_mean - 4*rule_std, rule_mean + 4*rule_std, 500)
    from scipy.stats import norm
    y_rule = norm.pdf(x_rule, rule_mean, rule_std)

    fig_rule = go.Figure()
    fig_rule.add_trace(go.Scatter(x=x_rule, y=y_rule, mode='lines', line=dict(color='#7c6aff', width=2.5), name='Bell Curve'))
    fig_rule.add_vrect(x0=rule_mean-rule_std, x1=rule_mean+rule_std, fillcolor="rgba(34,211,167,0.18)", line_width=0, annotation_text="68%", annotation_position="top")
    fig_rule.add_vrect(x0=rule_mean-2*rule_std, x1=rule_mean+2*rule_std, fillcolor="rgba(124,106,255,0.08)", line_width=0)
    for mult in [1, 2, 3]:
        fig_rule.add_vline(x=rule_mean+mult*rule_std, line_dash="dot", line_color="#2d3148")
        fig_rule.add_vline(x=rule_mean-mult*rule_std, line_dash="dot", line_color="#2d3148")
    fig_rule.add_annotation(x=rule_mean, y=max(y_rule)*0.5, text="<b>68%</b>", showarrow=False, font=dict(color="#22d3a7", size=14))
    fig_rule.update_layout(height=300, title=f"The 68-95-99.7 Rule: Average={rule_mean}, Spread={rule_std}", **DL)
    st.plotly_chart(fig_rule, use_container_width=True, config={"displayModeBar": False})

    mc = st.columns(3)
    mc[0].metric("68% are between", f"{rule_mean-rule_std} – {rule_mean+rule_std}")
    mc[1].metric("95% are between", f"{rule_mean-2*rule_std} – {rule_mean+2*rule_std}")
    mc[2].metric("99.7% are between", f"{rule_mean-3*rule_std} – {rule_mean+3*rule_std}")

    check_quiz("m1_q2",
        f"If the average is {rule_mean} and std dev is {rule_std}, a value of {rule_mean + 3*rule_std + 5} would be...",
        ["Totally normal", "A little unusual", "Extremely rare (beyond 3 std devs) — investigate!"],
        2,
        f"That value is more than 3 standard deviations from the mean. Only 0.3% of data falls this far out. It's either an error, a special case, or something worth investigating."
    )

    # ── Outliers ──
    st.markdown("---")
    st.markdown("### 📖 Outliers: The Weird Ones")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> You're analyzing salaries at a startup. Everyone earns $50K–$80K... except the CEO who earns $2 million.
    <br><br>
    That CEO salary is an <b>outlier</b> — a data point that's far from the rest. The key question isn't "should I remove it?"
    The key question is: <b>"WHY is it there?"</b>
    <br><br>
    🔹 <b>It's an error</b> (typo, sensor glitch) → Remove it<br>
    🔹 <b>It's real but rare</b> (CEO salary, fraud transaction) → Keep it, it might be the most important data point!<br>
    🔹 <b>It's from a different group</b> (mixing student and professional salaries) → Separate the groups
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Add an outlier and watch the damage")
    outlier_v = st.slider("💰 Add a salary (drag right to add an outlier):", 50, 500, 75, 25, key="m1_out",
                          help="Normal salaries are 50-100K. Drag higher to see what happens!")
    np.random.seed(42)
    base = np.random.normal(75, 10, 50)
    data_out = np.append(base, outlier_v) if outlier_v > 100 else base

    mc = st.columns(4)
    mc[0].metric("Mean", f"${data_out.mean():.0f}K", f"{data_out.mean()-base.mean():+.0f}K" if outlier_v > 100 else None)
    mc[1].metric("Median", f"${np.median(data_out):.0f}K", f"{np.median(data_out)-np.median(base):+.0f}K" if outlier_v > 100 else None)
    mc[2].metric("Std Dev", f"{data_out.std():.0f}K", f"{data_out.std()-base.std():+.0f}K" if outlier_v > 100 else None)
    mc[3].metric("Mean Distortion", f"{((data_out.mean()/base.mean())-1)*100:+.0f}%" if outlier_v > 100 else "0%")

    fig3 = go.Figure()
    colors = ['#f45d6d' if v == outlier_v and outlier_v > 100 else '#22d3a7' for v in data_out]
    sizes = [14 if c == '#f45d6d' else 6 for c in colors]
    fig3.add_trace(go.Scatter(x=list(range(len(data_out))), y=data_out, mode='markers', marker=dict(color=colors, size=sizes)))
    fig3.add_hline(y=data_out.mean(), line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: ${data_out.mean():.0f}K")
    fig3.add_hline(y=np.median(data_out), line_dash="dash", line_color="#7c6aff", annotation_text=f"Median: ${np.median(data_out):.0f}K")
    fig3.update_layout(height=300, title="Outlier Impact: Mean gets dragged, Median stays put", xaxis_title="Employee #", yaxis_title="Salary ($K)", **DL)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    if outlier_v > 100:
        st.markdown(f"""<div class="red-box">👀 <b>See it?</b> One person earning ${outlier_v}K pulled the mean from ${base.mean():.0f}K to ${data_out.mean():.0f}K.
        But the median barely moved! That's why the median is <b>robust</b> — it ignores extremes.</div>""", unsafe_allow_html=True)

    # ── Matching Quiz ──
    st.markdown("---")
    match_quiz("m1_match",
        "Match each scenario to the best measure of center:",
        ["House prices in a city", "Most popular shoe size", "Test scores (no outliers)"],
        ["Mean", "Median", "Mode"],
        ["Median", "Mode", "Mean"],
        [
            "House prices are skewed (mansions pull the mean up). Median is better.",
            "Shoe sizes are categories — Mode tells you the most common one.",
            "With no outliers and symmetric data, Mean works great."
        ]
    )

    iq([
        {"q": "Explain mean, median, and mode. When would you use each?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "<b>Mean:</b> Use when data is symmetric, no outliers. <b>Median:</b> Use when data is skewed or has outliers (income, house prices). <b>Mode:</b> Use for categorical data (most popular product, most common error type). <b>Key insight:</b> If mean >> median, data is right-skewed.",
         "t": "Always mention outlier sensitivity of the mean. Give a concrete example."},
        {"q": "What is standard deviation? Explain it to a 10-year-old.", "d": "Easy", "c": ["Meta", "Apple"],
         "a": "Imagine your class takes a test. The average score is 75. Std dev tells you <b>how far most students scored from 75</b>. If std dev is 5, almost everyone got 70–80. If std dev is 20, scores ranged from 55 to 95. It measures <b>how spread out</b> the data is.",
         "t": "Use the exam analogy — it's universally understood."},
        {"q": "How do you detect outliers? Compare IQR and Z-score methods.", "d": "Medium", "c": ["Google", "Amazon", "Meta"],
         "a": "<b>IQR:</b> Q1 - 1.5×IQR to Q3 + 1.5×IQR. Doesn't assume normal distribution. <b>Z-score:</b> Flag |z| > 2-3. Assumes roughly normal data. <b>Key difference:</b> IQR is robust (outliers don't affect Q1/Q3 much). Z-score uses mean and std, which ARE affected by outliers.",
         "t": "Mention that you'd always VISUALIZE first. Box plots catch things formulas miss."},
        {"q": "You're told the average response time is 200ms. Is the system healthy?", "d": "Medium", "c": ["Amazon", "Google", "Netflix"],
         "a": "Not enough info! I'd ask: <b>What's the median?</b> <b>What's the std dev?</b> <b>What's the 95th/99th percentile?</b> P99 = 2000ms means 1% of users wait 2 seconds — terrible even if the average looks fine. <b>The average alone is never enough.</b>",
         "t": "This is a CLASSIC system design question. Always ask for percentiles, not just averages."},
    ])


# ═══════════════════════════════════════
# MODULE 2: PROBABILITY
# ═══════════════════════════════════════
elif module == "🎲 M2: Probability":
    st.markdown("# 🎲 Module 2: Probability")
    st.caption("Week 1–2 · The math behind 'what are the odds?'")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> You're at a carnival. There's a game where you pick a door — behind one is a prize.
    Should you play? How do you decide? You think about the <b>chances</b>.
    <br><br>
    Probability is just a fancy word for <b>"how likely is something to happen?"</b>
    It's a number from 0 (impossible — the sun won't rise in the west) to 1 (certain — you'll eventually get hungry).
    <br><br>
    Every prediction, every risk assessment, every "should we launch this feature?" decision is built on probability.
    </div>""", unsafe_allow_html=True)

    # ── Basics ──
    st.markdown("### 📖 The Basics: Counting Chances")

    st.markdown("""<div class="analogy-box">
    🎯 <b>The simplest formula in all of statistics:</b>
    <br><br>
    <b>Probability = (ways it can happen) ÷ (total possible outcomes)</b>
    <br><br>
    🪙 Coin flip: 1 way to get heads ÷ 2 total outcomes = <b>0.5 (50%)</b><br>
    🎲 Rolling a 6: 1 way ÷ 6 total outcomes = <b>0.167 (16.7%)</b><br>
    🃏 Drawing an Ace: 4 aces ÷ 52 cards = <b>0.077 (7.7%)</b>
    </div>""", unsafe_allow_html=True)

    # ── Three Rules ──
    st.markdown("### 📖 The Three Rules You Actually Need")

    st.markdown("""<div class="story-box" style="border-left:4px solid #7c6aff">
    <b style="color:#7c6aff">Rule 1: "OR" — Add (but don't double-count)</b>
    <br><br>
    <b>Question:</b> "What's the chance of A happening OR B happening?"
    <br><b>Formula:</b> P(A or B) = P(A) + P(B) − P(both)
    <br><br>
    <b>🍕 Pizza analogy:</b> In a class of 30 kids, 10 like pepperoni, 8 like mushroom, 3 like both.
    How many like at least one? Not 10+8=18 (you counted the 3 twice!). It's 10+8−3 = <b>15</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box" style="border-left:4px solid #22d3a7">
    <b style="color:#22d3a7">Rule 2: "AND" — Multiply (if independent)</b>
    <br><br>
    <b>Question:</b> "What's the chance of A AND B both happening?"
    <br><b>Formula:</b> P(A and B) = P(A) × P(B) — but ONLY if they don't affect each other!
    <br><br>
    <b>🪙 Example:</b> Flip a coin AND roll a die. P(heads AND six) = ½ × ⅙ = <b>1/12</b>.
    The coin doesn't care what the die does — they're <b>independent</b>.
    <br><br>
    <b>☔ Counter-example:</b> P(umbrella AND rain) — you CAN'T just multiply these!
    If it's raining, you're way more likely to carry an umbrella. They're <b>dependent</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="story-box" style="border-left:4px solid #f5b731">
    <b style="color:#f5b731">Rule 3: "NOT" — Subtract from 1</b>
    <br><br>
    <b>Question:</b> "What's the chance of something NOT happening?"
    <br><b>Formula:</b> P(not A) = 1 − P(A)
    <br><br>
    <b>🎲 Trick:</b> "What's the probability of getting at least one 6 in four dice rolls?"
    Hard to calculate directly. But P(zero sixes) = (5/6)⁴ = 0.48. So P(at least one 6) = 1 − 0.48 = <b>0.52</b>. Easy!
    </div>""", unsafe_allow_html=True)

    # ── Interactive Coin Flip ──
    st.markdown("#### 🎮 Try It: Coin Flip Simulator")
    st.caption("Flip a coin many times and watch the probability settle toward 0.5. This is the Law of Large Numbers in action!")

    n_flips = st.slider("Number of flips:", 10, 50000, 1000, 100, key="m2_flips")
    if st.button("🪙 Flip the coins!", key="m2_flip_btn"):
        np.random.seed(None)
        results = np.random.choice([0, 1], n_flips)
        running = np.cumsum(results) / np.arange(1, n_flips + 1)
        mc = st.columns(2)
        mc[0].metric("Heads %", f"{results.mean()*100:.1f}%", f"{(results.mean()-0.5)*100:+.1f}% from 50%")
        mc[1].metric("Total Flips", f"{n_flips:,}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=running, mode='lines', line=dict(color='#7c6aff', width=1.5), name="Running average"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#22d3a7", annotation_text="Theory: 50%")
        fig.update_layout(height=300, title="Watch it converge! More flips → closer to 50%", xaxis_title="Flip #", yaxis_title="% Heads", **DL)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""<div class="green-box">💡 <b>The Law of Large Numbers:</b> With few flips, the result is wild (maybe 70% heads!).
        With thousands of flips, it always settles near 50%. This is why casinos always win in the long run — they play millions of "flips."</div>""", unsafe_allow_html=True)

    # ── Conditional Probability ──
    st.markdown("---")
    st.markdown("### 📖 Conditional Probability: 'Given that X happened...'")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> A school has 200 students. You know:
    <br>• 120 play sports (60%)
    <br>• 80 get A grades (40%)
    <br>• 50 do BOTH (25%)
    <br><br>
    Now someone says: <i>"I picked a random student who plays sports."</i>
    What's the chance they ALSO get A grades?
    <br><br>
    <b>Key insight:</b> You're no longer looking at all 200 students. You've <b>zoomed in</b> to just the 120 athletes.
    Of those 120, how many get A's? <b>50.</b>
    <br><br>
    <div style="text-align:center;font-size:1.2rem;padding:0.6rem;background:rgba(124,106,255,0.08);border-radius:10px;margin:0.5rem 0">
    P(A grades | plays sports) = 50 / 120 = <b>41.7%</b>
    </div>
    <br>
    Compare to P(A grades) for ALL students = 40%. Knowing they play sports <b>slightly changed</b> the probability.
    The "given" information <b>narrows your world</b>.
    </div>""", unsafe_allow_html=True)

    # ── Bayes' Theorem ──
    st.markdown("---")
    st.markdown("### 📖 Bayes' Theorem: The Plot Twist of Probability")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> A disease affects 1 in 1,000 people. You take a test that's 99% accurate. It comes back <b>positive</b>.
    <br><br>
    <b>Quick — what's the chance you're actually sick?</b>
    <br><br>
    Most people say "99%!" But the real answer is shocking...
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="analogy-box">
    🧮 <b>Let's walk through it with 10,000 people:</b>
    <br><br>
    • <b>10 people</b> actually have the disease (1 in 1,000)<br>
    • Of those 10, the test correctly catches <b>~10</b> (99% sensitivity)<br>
    • Of the 9,990 healthy people, the test <b>falsely flags ~500</b> (5% false positive rate)<br>
    <br>
    So 510 people test positive, but only 10 are actually sick!
    <br><br>
    <div style="text-align:center;font-size:1.2rem;padding:0.6rem;background:rgba(244,93,109,0.1);border-radius:10px;margin:0.5rem 0">
    P(sick | positive test) = 10 / 510 ≈ <b>2%</b> 😱
    </div>
    <br>
    <b>Why so low?</b> The disease is so rare that false positives overwhelm true positives.
    This is called the <b>base rate fallacy</b> — ignoring how rare something is.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Bayes' Calculator")
    st.caption("Adjust the numbers and see how the answer changes. Try making the disease more common!")
    c1, c2, c3 = st.columns(3)
    prevalence = c1.slider("How common is the disease?", 0.001, 0.1, 0.001, 0.001, format="%.3f", key="m2_prev",
                           help="0.001 = 1 in 1000 people have it")
    sensitivity = c2.slider("Test accuracy (catches sick people):", 0.5, 1.0, 0.99, 0.01, key="m2_sens")
    fpr = c3.slider("False alarm rate:", 0.01, 0.2, 0.05, 0.01, key="m2_fpr",
                    help="How often the test says 'positive' for healthy people")

    p_pos = sensitivity * prevalence + fpr * (1 - prevalence)
    p_disease_given_pos = (sensitivity * prevalence) / p_pos

    st.markdown(f"""<div class="{'green-box' if p_disease_given_pos > 0.5 else 'red-box'}">
    <b>P(actually sick | tested positive) = {p_disease_given_pos:.1%}</b>
    <br><br>{'✅ The test is reliable here — a positive result likely means disease.' if p_disease_given_pos > 0.5 else f'⚠️ Despite a {sensitivity:.0%} accurate test, a positive result only means a <b>{p_disease_given_pos:.1%}</b> chance of disease! The disease is so rare that false positives dominate.'}
    <br><br>💡 <b>Try it:</b> Increase the disease prevalence and watch the probability jump. The rarer the condition, the less you can trust a single positive test.
    </div>""", unsafe_allow_html=True)

    # ── Quiz ──
    check_quiz("m2_q1",
        "You flip a fair coin 10 times and get 10 heads. What's the probability of heads on the 11th flip?",
        ["Almost 100% — it's on a streak!", "Less than 50% — tails is 'due'", "Exactly 50% — each flip is independent"],
        2,
        "Each flip is independent — the coin has no memory! This mistake is called the Gambler's Fallacy."
    )

    iq([
        {"q": "Explain Bayes' Theorem with a real example.", "d": "Hard", "c": ["Google", "Apple", "Amazon"],
         "a": "Bayes' updates beliefs with new evidence. <b>Example:</b> Disease affects 1/1000. Test is 99% accurate, 5% false positive. You test positive. Out of 10,000 people: 10 true positives + 500 false positives = 510 positives. P(sick|positive) = 10/510 ≈ 2%. The 'base rate' (rarity of disease) dominates.",
         "t": "Walk through the numbers step by step. Draw a 2×2 table if you can."},
        {"q": "What's the difference between independent and dependent events?", "d": "Easy", "c": ["Google", "Meta"],
         "a": "<b>Independent:</b> One event doesn't affect the other. P(A and B) = P(A) × P(B). Example: two coin flips. <b>Dependent:</b> One event changes the probability of the other. P(A and B) = P(A) × P(B|A). Example: drawing cards without replacement.",
         "t": "Use the card deck example — it's concrete and universally understood."},
        {"q": "A spam filter has 95% accuracy. 2% of emails are spam. An email is flagged. What's the probability it's actually spam?", "d": "Hard", "c": ["Google", "Netflix"],
         "a": "Bayes': P(spam|flagged) = P(flagged|spam)×P(spam) / P(flagged). P(flagged) = 0.95×0.02 + 0.05×0.98 = 0.068. P(spam|flagged) = 0.019/0.068 ≈ <b>28%</b>. Despite 95% accuracy, only 28% of flagged emails are actually spam!",
         "t": "Same structure as the disease testing question. Show you can apply Bayes' to any domain."},
    ])


# ═══════════════════════════════════════
# MODULE 3: DISTRIBUTIONS
# ═══════════════════════════════════════
elif module == "📈 M3: Distributions":
    st.markdown("# 📈 Module 3: Distributions")
    st.caption("Week 2 · What shape does your data make? And why should you care?")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> Imagine you're pouring M&Ms from a bag onto a table and sorting them by color.
    You'd see a <b>pattern</b> — maybe lots of brown, fewer blue, very few green.
    <br><br>
    A <b>distribution</b> is just that pattern — it shows you <b>how your data is spread out</b>.
    Is it a nice bell shape? Is it lopsided? Does it have two humps? The shape tells you
    which tools work and what to expect.
    </div>""", unsafe_allow_html=True)

    # ── Normal Distribution ──
    st.markdown("### 📖 The Bell Curve (Normal Distribution)")

    st.markdown("""<div class="analogy-box">
    🏀 <b>Think of it like basketball players' heights:</b>
    <br><br>
    Most NBA players are around 6'6" (the peak of the bell). A few are shorter (6'0"), a few are taller (7'2"),
    but the extremes are rare. If you plotted all their heights, you'd get a beautiful bell shape.
    <br><br>
    The bell curve is defined by just <b>two numbers</b>:
    <br>• <b>μ (mu) = the average</b> — where the peak sits
    <br>• <b>σ (sigma) = the spread</b> — how wide the bell is
    <br><br>
    Small σ = everyone is similar height (tall, narrow bell)<br>
    Large σ = heights vary a lot (short, wide bell)
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Shape the Bell Curve Yourself")
    st.caption("Drag the sliders and watch the bell change shape!")

    c1, c2 = st.columns(2)
    mu = c1.slider("📍 Average (μ) — where's the peak?", -5.0, 5.0, 0.0, 0.5, key="m3_mu")
    sigma = c2.slider("📏 Spread (σ) — how wide?", 0.3, 4.0, 1.0, 0.1, key="m3_sigma")

    x = np.linspace(-10, 10, 500)
    y = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor='rgba(124,106,255,0.2)', line=dict(color='#7c6aff', width=2.5)))
    fig.add_vline(x=mu, line_dash="dash", line_color="#22d3a7", annotation_text=f"Peak (μ={mu})")
    fig.add_vrect(x0=mu-sigma, x1=mu+sigma, fillcolor="rgba(34,211,167,0.12)", line_width=0)
    fig.add_annotation(x=mu, y=max(y)*0.3, text="<b>68% of data</b>", showarrow=False, font=dict(color="#22d3a7", size=12))
    fig.update_layout(height=350, title=f"Bell Curve: Average={mu}, Spread={sigma}", xaxis_title="Value", yaxis_title="How common?", **DL)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">
    💡 <b>What you're seeing:</b> The peak is at μ={mu} — most values cluster here.
    The green band covers μ±σ = [{mu-sigma:.1f} to {mu+sigma:.1f}] — <b>68%</b> of all data falls in this range.
    {'The bell is narrow — data is very consistent.' if sigma < 1.0 else 'The bell is wide — data varies a lot.' if sigma > 2.0 else 'A moderate spread.'}
    <b>Try dragging σ</b> to see the bell get wider/narrower!
    </div>""", unsafe_allow_html=True)

    # ── Skewed ──
    st.markdown("---")
    st.markdown("### 📖 When Data Isn't a Bell: Skewed Distributions")

    st.markdown("""<div class="story-box">
    Not everything is a nice bell curve. <b>Income</b> is a classic example:
    most people earn $30K–$80K, but a few billionaires stretch the tail way to the right.
    <br><br>
    🔹 <b>Right-skewed</b> (tail goes right): Income, house prices, social media followers<br>
    🔹 <b>Left-skewed</b> (tail goes left): Age at retirement, easy exam scores<br>
    🔹 <b>Symmetric</b>: Heights, IQ scores, measurement errors
    <br><br>
    <b>Quick test:</b> If mean > median → right-skewed. If mean < median → left-skewed.
    </div>""", unsafe_allow_html=True)

    skew_type = st.radio("Pick a shape to explore:", ["🔔 Symmetric (Bell Curve)", "➡️ Right-Skewed (Income-like)", "⬅️ Left-Skewed"], horizontal=True, key="m3_skew")
    np.random.seed(42)
    if "Symmetric" in skew_type:
        sdata = np.random.normal(50, 10, 2000)
        story = "Perfectly balanced — mean and median are the same."
    elif "Right" in skew_type:
        sdata = np.random.exponential(20, 2000) + 20
        story = "Long tail to the right. A few very high values pull the mean above the median."
    else:
        sdata = 100 - np.random.exponential(20, 2000)
        story = "Long tail to the left. A few very low values pull the mean below the median."

    fig_s = go.Figure(go.Histogram(x=sdata, nbinsx=40, marker_color="#22d3a7", opacity=0.7))
    fig_s.add_vline(x=sdata.mean(), line_dash="dash", line_color="#f45d6d", annotation_text=f"Mean: {sdata.mean():.1f}")
    fig_s.add_vline(x=np.median(sdata), line_dash="dash", line_color="#7c6aff", annotation_text=f"Median: {np.median(sdata):.1f}")
    fig_s.update_layout(height=300, title=skew_type.split(" ", 1)[1], **DL)
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">💡 {story} Gap between mean and median = {abs(sdata.mean()-np.median(sdata)):.1f}</div>""", unsafe_allow_html=True)

    # ── Poisson ──
    st.markdown("---")
    st.markdown("### 📖 Poisson Distribution: Counting Rare Events")

    st.markdown("""<div class="analogy-box">
    📞 <b>Real-life examples:</b>
    <br>• How many customers call support per hour?
    <br>• How many typos per page in a book?
    <br>• How many goals in a soccer match?
    <br><br>
    These are all <b>counts of events</b> in a fixed time/space. The Poisson distribution models them perfectly.
    It has just one number: <b>λ (lambda) = the average rate</b>.
    <br><br>
    If your call center averages 4 calls/hour (λ=4), Poisson tells you the probability of getting 0, 1, 2, ... 10+ calls in any given hour.
    </div>""", unsafe_allow_html=True)

    lam = st.slider("📞 Average events per hour (λ):", 1, 20, 4, key="m3_lam")
    pois = np.random.poisson(lam, 5000)
    fig_p = go.Figure(go.Histogram(x=pois, nbinsx=int(max(pois)-min(pois)+1), marker_color="#f5b731", opacity=0.7))
    fig_p.add_vline(x=lam, line_dash="dash", line_color="#f45d6d", annotation_text=f"Average: λ={lam}")
    fig_p.update_layout(height=280, title=f"If you average {lam} events/hour, here's what to expect", xaxis_title="Events in one hour", yaxis_title="How often", **DL)
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">💡 If λ={lam} and you suddenly see <b>{lam*3}+ events</b> in one hour, that's a red flag — something unusual is happening!</div>""", unsafe_allow_html=True)

    # ── Binomial ──
    st.markdown("---")
    st.markdown("### 📖 Binomial Distribution: Counting Successes")

    st.markdown("""<div class="analogy-box">
    🎯 <b>The setup:</b> You do something a fixed number of times, and each time it either <b>succeeds</b> or <b>fails</b>.
    <br><br>
    • Send 100 emails → how many get opened? (n=100, p=open rate)<br>
    • Ask 50 customers → how many buy? (n=50, p=conversion rate)<br>
    • Flip a coin 20 times → how many heads? (n=20, p=0.5)
    <br><br>
    Two numbers define it: <b>n</b> (how many tries) and <b>p</b> (chance of success each try).
    </div>""", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    n_trials = bc1.slider("🔄 Number of tries (n):", 5, 100, 20, key="m3_binom_n")
    p_success = bc2.slider("🎯 Success rate (p):", 0.01, 0.99, 0.3, 0.01, key="m3_binom_p")

    np.random.seed(42)
    binom_data = np.random.binomial(n_trials, p_success, 5000)
    expected = n_trials * p_success

    fig_b = go.Figure(go.Histogram(x=binom_data, nbinsx=min(n_trials+1, 50), marker_color="#e879a8", opacity=0.7))
    fig_b.add_vline(x=expected, line_dash="dash", line_color="#22d3a7", annotation_text=f"Expected: {expected:.1f}")
    fig_b.update_layout(height=280, title=f"Out of {n_trials} tries with {p_success:.0%} success rate", xaxis_title="Number of Successes", yaxis_title="How often", **DL)
    st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">💡 With {n_trials} tries and {p_success:.0%} success rate, expect about <b>{expected:.0f} successes</b>.
    But it could be anywhere from {int(max(0, expected-3*np.sqrt(expected*(1-p_success))))} to {int(min(n_trials, expected+3*np.sqrt(expected*(1-p_success))))} — that's normal variation!</div>""", unsafe_allow_html=True)

    # ── Why Distributions Matter ──
    st.markdown("---")
    st.markdown("### 🧠 Why Should You Care About Distributions?")

    reasons = [
        ("🔍 Choosing the right tool", "#7c6aff", "Many statistical tests assume your data is bell-shaped. If it's not, those tests give wrong answers. Knowing the shape tells you which tools are safe."),
        ("🚨 Spotting problems", "#f45d6d", "If you know 'normal' looks like 4 errors/hour, then 15 errors/hour is a red flag. Distributions define what 'normal' looks like."),
        ("🤖 Better ML models", "#22d3a7", "ML models often work better when data is bell-shaped. That's why data scientists transform skewed data before modeling."),
        ("📊 Picking the right model", "#f5b731", "Count data → Poisson regression. Yes/no data → Logistic regression. Continuous data → Linear regression. The distribution guides the choice."),
    ]
    for title, color, desc in reasons:
        st.markdown(f"""<div class="story-box" style="border-left:4px solid {color}"><b>{title}</b><br>{desc}</div>""", unsafe_allow_html=True)

    # ── Matching Quiz ──
    match_quiz("m3_match",
        "Match each real-world scenario to the right distribution:",
        ["Heights of adults", "Number of website crashes per month", "Emails opened out of 100 sent"],
        ["Normal (Bell Curve)", "Poisson", "Binomial"],
        ["Normal (Bell Curve)", "Poisson", "Binomial"],
        [
            "Heights cluster around an average with a bell shape — classic Normal distribution.",
            "Counting rare events (crashes) in a fixed time period — that's Poisson.",
            "Fixed number of trials (100 emails), each with a success/fail outcome — Binomial."
        ]
    )

    iq([
        {"q": "What is the normal distribution and why is it important?", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "Bell-shaped, symmetric, defined by mean and std dev. <b>Important because:</b> (1) Many natural things follow it. (2) The Central Limit Theorem says sample averages become normal. (3) Many statistical tests assume it. (4) Foundation of confidence intervals.",
         "t": "Mention CLT — it's the bridge between distributions and inferential statistics."},
        {"q": "When would you use Poisson vs Binomial vs Normal?", "d": "Medium", "c": ["Google", "Netflix"],
         "a": "<b>Normal:</b> Continuous measurements (height, temperature). <b>Binomial:</b> Count of successes in fixed trials (emails opened out of 100). <b>Poisson:</b> Count of events in fixed time with no upper limit (calls per hour). Key: Binomial has a ceiling (n), Poisson doesn't.",
         "t": "Show you understand the relationships between distributions."},
    ])


# ═══════════════════════════════════════
# MODULE 4: INFERENTIAL STATISTICS
# ═══════════════════════════════════════
elif module == "🧪 M4: Inferential Statistics":
    st.markdown("# 🧪 Module 4: Inferential Statistics")
    st.caption("Week 3 · Z-Scores → T-Tests → P-Values — one story, three tools.")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> You manage PizzaChain — 50 stores across East and West regions.
    Three questions keep coming up:
    <br><br>
    🔹 <b>"Is Store #1 abnormally good?"</b> → You need a <b>Z-Score</b><br>
    🔹 <b>"Are East stores really better than West?"</b> → You need a <b>T-Test</b><br>
    🔹 <b>"Can I trust this result?"</b> → You need a <b>P-Value</b>
    <br><br>
    Let's use the <b>same data</b> to walk through all three — step by step, with the actual math.
    </div>""", unsafe_allow_html=True)

    # ── Generate the shared dataset ──
    np.random.seed(42)
    n_stores = 50
    store_sales = np.round(np.random.normal(500, 100, n_stores)).astype(int)
    store_sales[0] = 850  # flagship store
    mu_all = store_sales.mean()
    sigma_all = store_sales.std()

    # East vs West split
    east_sales = store_sales[:22] + 45  # East is slightly better
    west_sales = store_sales[22:]

    st.markdown("""<div class="key-box">
    <b>📋 Our Data:</b> 50 pizza stores. Average daily sales = ${:.0f}, Std Dev = ${:.0f}.
    Store #1 (flagship) sells $850/day. East region has 22 stores, West has 28.
    </div>""".format(mu_all, sigma_all), unsafe_allow_html=True)

    # ═══════════════════════════════════
    # PART 1: Z-SCORE
    # ═══════════════════════════════════
    st.markdown("---")
    st.markdown("## 📏 Part 1: Z-Score — 'Is This Store Abnormal?'")

    st.markdown("""<div class="story-box">
    <b>🎬 Scene 1:</b> Your boss points at Store #1: "They made $850 yesterday. Is that genuinely exceptional,
    or could any store have a day like that?"
    <br><br>
    The Z-score answers this by measuring <b>how many standard deviations</b> a value is from the average.
    Think of it as a <b>ruler for weirdness</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="analogy-box">
    🧮 <b>The Math — Step by Step:</b>
    <br><br>
    <b>Formula:</b> z = (value − mean) / std dev
    <br><br>
    <b>Our numbers:</b><br>
    • Store #1 sales = $850<br>
    • Chain average (μ) = ${:.0f}<br>
    • Chain spread (σ) = ${:.0f}
    <br><br>
    <b>Calculation:</b><br>
    z = (850 − {:.0f}) / {:.0f}<br>
    z = {:.0f} / {:.0f}<br>
    z = <b>{:.2f}</b>
    </div>""".format(mu_all, sigma_all, mu_all, sigma_all, 850 - mu_all, sigma_all, (850 - mu_all) / sigma_all), unsafe_allow_html=True)

    z_flagship = (850 - mu_all) / sigma_all
    from scipy.stats import norm
    pct_flagship = norm.cdf(z_flagship) * 100

    mc = st.columns(4)
    mc[0].metric("Store #1 Sales", "$850")
    mc[1].metric("Z-Score", f"{z_flagship:.2f}")
    mc[2].metric("Percentile", f"{pct_flagship:.1f}%")
    mc[3].metric("Verdict", "😱 Very rare!" if abs(z_flagship) > 2.5 else "🤔 Unusual" if abs(z_flagship) > 2 else "😐 Normal")

    # Bell curve visualization
    x_bell = np.linspace(mu_all - 4*sigma_all, mu_all + 4*sigma_all, 500)
    y_bell = norm.pdf(x_bell, mu_all, sigma_all)
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=x_bell, y=y_bell, fill='tozeroy', fillcolor='rgba(124,106,255,0.15)', line=dict(color='#7c6aff', width=2), name='All stores'))
    x_beyond = x_bell[x_bell >= 850]
    y_beyond = norm.pdf(x_beyond, mu_all, sigma_all)
    fig_z.add_trace(go.Scatter(x=np.append(x_beyond, [x_beyond[-1], x_beyond[0]]), y=np.append(y_beyond, [0, 0]),
                                fill='toself', fillcolor='rgba(244,93,109,0.35)', line=dict(color='rgba(0,0,0,0)'), name=f'Top {100-pct_flagship:.1f}%'))
    fig_z.add_vline(x=850, line_dash="dash", line_color="#f45d6d", annotation_text=f"Store #1: $850 (z={z_flagship:.2f})")
    fig_z.add_vline(x=mu_all, line_dash="dot", line_color="#22d3a7", annotation_text=f"Average: ${mu_all:.0f}")
    for mult in [1, 2, 3]:
        fig_z.add_vline(x=mu_all + mult*sigma_all, line_dash="dot", line_color="#2d3148")
        fig_z.add_vline(x=mu_all - mult*sigma_all, line_dash="dot", line_color="#2d3148")
    fig_z.update_layout(height=350, title="Where does Store #1 fall on the bell curve?", xaxis_title="Daily Sales ($)", yaxis_title="How common?", **DL)
    st.plotly_chart(fig_z, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">
    💡 <b>Inference:</b> z = {z_flagship:.2f} means Store #1 is {z_flagship:.1f} standard deviations above average.
    Only <b>{100-pct_flagship:.1f}%</b> of stores perform this well. This is NOT normal variation — something special
    is happening at this store. Investigate what they're doing differently!
    <br><br>
    <b>Z-Score Cheat Sheet:</b><br>
    • |z| < 1.5 → Normal (nothing to see here)<br>
    • |z| 1.5–2.5 → Unusual (worth a look)<br>
    • |z| > 2.5 → Very rare (investigate immediately!)
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: Check Any Store")
    zc1, zc2, zc3 = st.columns(3)
    z_mean = zc1.number_input("Chain average (μ):", value=float(round(mu_all)), key="m4_zmean")
    z_std = zc2.number_input("Chain spread (σ):", value=float(round(sigma_all)), min_value=1.0, key="m4_zstd")
    z_val = zc3.number_input("Store's sales:", value=850.0, key="m4_zval")
    z_score = (z_val - z_mean) / z_std
    percentile = norm.cdf(z_score) * 100
    mc = st.columns(3)
    mc[0].metric("Z-Score", f"{z_score:.2f}")
    mc[1].metric("Percentile", f"{percentile:.1f}%")
    mc[2].metric("How unusual?", "😐 Normal" if abs(z_score) < 1.5 else "🤔 Unusual" if abs(z_score) < 2.5 else "😱 Very rare!")

    # ═══════════════════════════════════
    # PART 2: T-TEST
    # ═══════════════════════════════════
    st.markdown("---")
    st.markdown("## 🔬 Part 2: T-Test — 'Are East Stores Really Better?'")

    st.markdown("""<div class="story-box">
    <b>🎬 Scene 2:</b> Your boss looks at the regional report: East stores average ${:.0f}/day,
    West stores average ${:.0f}/day. "East is clearly better! Let's copy whatever they're doing."
    <br><br>
    But wait — is that ${:.0f} difference <b>real</b>, or could it just be random luck?
    Maybe East just happened to have a few good days. The <b>T-test</b> answers this.
    </div>""".format(east_sales.mean(), west_sales.mean(), east_sales.mean() - west_sales.mean()), unsafe_allow_html=True)

    st.markdown("""<div class="analogy-box">
    ⚖️ <b>When to use which test:</b>
    <br><br>
    <b>Z-Score:</b> "Is this ONE value weird?" (one store vs the chain)<br>
    <b>T-Test:</b> "Are these TWO GROUPS different?" (East vs West)<br>
    <br>
    <b>Z-Test vs T-Test:</b> Z-test requires knowing the population σ (rare). T-test estimates σ from your sample (what you'll use 99% of the time).
    </div>""", unsafe_allow_html=True)

    from scipy.stats import ttest_ind
    mean_diff = east_sales.mean() - west_sales.mean()
    se = np.sqrt(east_sales.var()/len(east_sales) + west_sales.var()/len(west_sales))
    t_stat_manual = mean_diff / se
    t_stat, p_val_tt = ttest_ind(east_sales, west_sales)

    st.markdown("""<div class="analogy-box">
    🧮 <b>The Math — Step by Step:</b>
    <br><br>
    <b>Formula:</b> t = (mean₁ − mean₂) / SE, where SE = √(s₁²/n₁ + s₂²/n₂)
    <br><br>
    <b>Step 1: Group stats</b><br>
    • East: n={}, mean=${:.1f}, std=${:.1f}<br>
    • West: n={}, mean=${:.1f}, std=${:.1f}
    <br><br>
    <b>Step 2: Difference</b><br>
    mean_east − mean_west = {:.1f} − {:.1f} = <b>${:.1f}</b>
    <br><br>
    <b>Step 3: Standard Error</b> (how much could this differ by chance?)<br>
    SE = √({:.1f}²/{} + {:.1f}²/{}) = √({:.1f} + {:.1f}) = <b>{:.2f}</b>
    <br><br>
    <b>Step 4: T-statistic</b> (signal ÷ noise)<br>
    t = {:.1f} / {:.2f} = <b>{:.2f}</b>
    <br><br>
    <b>Step 5: P-value</b> = <b>{:.4f}</b>
    </div>""".format(
        len(east_sales), east_sales.mean(), east_sales.std(),
        len(west_sales), west_sales.mean(), west_sales.std(),
        east_sales.mean(), west_sales.mean(), mean_diff,
        east_sales.std(), len(east_sales), west_sales.std(), len(west_sales),
        east_sales.var()/len(east_sales), west_sales.var()/len(west_sales), se,
        mean_diff, se, t_stat,
        p_val_tt
    ), unsafe_allow_html=True)

    mc = st.columns(4)
    mc[0].metric("East Mean", f"${east_sales.mean():.0f}")
    mc[1].metric("West Mean", f"${west_sales.mean():.0f}")
    mc[2].metric("T-Statistic", f"{t_stat:.2f}")
    mc[3].metric("P-Value", f"{p_val_tt:.4f}")

    # Visualization
    fig_tt = go.Figure()
    fig_tt.add_trace(go.Histogram(x=east_sales, name=f"East (${east_sales.mean():.0f})", marker_color="#7c6aff", opacity=0.5, nbinsx=12))
    fig_tt.add_trace(go.Histogram(x=west_sales, name=f"West (${west_sales.mean():.0f})", marker_color="#22d3a7", opacity=0.5, nbinsx=12))
    fig_tt.add_vline(x=east_sales.mean(), line_dash="dash", line_color="#7c6aff")
    fig_tt.add_vline(x=west_sales.mean(), line_dash="dash", line_color="#22d3a7")
    fig_tt.update_layout(barmode="overlay", height=300, title="East vs West: Do the distributions actually differ?", xaxis_title="Daily Sales ($)", **DL)
    st.plotly_chart(fig_tt, use_container_width=True, config={"displayModeBar": False})

    if p_val_tt < 0.05:
        st.markdown(f"""<div class="green-box">✅ <b>p = {p_val_tt:.4f} < 0.05 → Significant!</b>
        The ${mean_diff:.0f} difference is real, not random noise. East stores genuinely outperform West.
        <br>ACTION: Investigate what East is doing differently (location? staff? marketing?).</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="red-box">❌ <b>p = {p_val_tt:.4f} ≥ 0.05 → Not significant.</b>
        The difference could be random variation. We can't conclude East is truly better.</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="key-box">
    <b>📋 Three Types of T-Tests:</b><br>
    • <b>One-sample:</b> "Is our average delivery time ≠ 30 min SLA?" (one group vs a target)<br>
    • <b>Two-sample:</b> "East vs West sales" (two separate groups) ← what we just did<br>
    • <b>Paired:</b> "Same stores before vs after renovation" (same group, two measurements)
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════
    # PART 3: P-VALUE
    # ═══════════════════════════════════
    st.markdown("---")
    st.markdown("## 🎯 Part 3: P-Value — 'Can I Trust This?'")

    st.markdown("""<div class="story-box">
    <b>🎬 Scene 3:</b> You present the t-test results to your boss. They ask: "What does p = {:.4f} actually mean?"
    <br><br>
    The p-value is the <b>probability of seeing a result this extreme IF there were truly no difference</b>.
    <br><br>
    Think of it like a courtroom:<br>
    🔹 <b>H₀ (null):</b> "East and West are the same" (innocent until proven guilty)<br>
    🔹 <b>H₁ (alternative):</b> "East and West are different"<br>
    🔹 <b>p-value:</b> "If they ARE the same, what's the chance we'd see a ${:.0f} gap just by luck?"<br>
    🔹 <b>p < 0.05:</b> "Less than 5% chance → reject H₀ → the difference is real"
    </div>""".format(p_val_tt, mean_diff), unsafe_allow_html=True)

    st.markdown("#### 🎮 Try It: The Suspicious Coin")
    st.caption("A simpler example to build intuition. Your friend's coin lands heads 60 out of 100 times. Fair or rigged?")

    n_heads = st.slider("Heads out of 100 flips:", 40, 75, 60, 1, key="m4_pval_heads")
    from scipy.stats import binom
    p_value_coin = 1 - binom.cdf(n_heads - 1, 100, 0.5)

    mc = st.columns(3)
    mc[0].metric("Heads", f"{n_heads} / 100")
    mc[1].metric("P-Value", f"{p_value_coin:.4f}")
    mc[2].metric("Verdict", "🟢 Looks fair" if p_value_coin > 0.05 else "🔴 Suspicious!" if p_value_coin > 0.01 else "🔴 Rigged!")

    x_vals = np.arange(30, 71)
    y_vals = binom.pmf(x_vals, 100, 0.5)
    colors_pv = ['#f45d6d' if x >= n_heads else '#7c6aff' for x in x_vals]
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Bar(x=x_vals, y=y_vals, marker_color=colors_pv, opacity=0.7))
    fig_pv.add_vline(x=n_heads, line_dash="dash", line_color="#f5b731", annotation_text=f"Your result: {n_heads}")
    fig_pv.add_vline(x=50, line_dash="dot", line_color="#22d3a7", annotation_text="Expected: 50")
    fig_pv.update_layout(height=320, title=f"Red area = p-value = {p_value_coin:.4f}", xaxis_title="Number of Heads", yaxis_title="Probability", **DL)
    st.plotly_chart(fig_pv, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="{'green-box' if p_value_coin > 0.05 else 'red-box'}">
    📖 <b>Reading this chart:</b> Blue bars = all possible outcomes with a fair coin.
    <b style="color:#f45d6d">Red bars</b> = outcomes as extreme as yours ({n_heads}+ heads).
    Red area = p-value = {p_value_coin:.4f}.
    {'Not unusual for a fair coin.' if p_value_coin > 0.05 else 'Very unlikely with a fair coin — probably rigged!'}
    </div>""", unsafe_allow_html=True)

    # ── Common Mistakes ──
    st.markdown("### ⚠️ Three Things P-Value Does NOT Mean")
    st.markdown("""<div class="red-box">
    <b>❌</b> "There's a 3% chance the null is true." → <b>✅</b> "IF the null is true, there's a 3% chance of this result."
    <br><br>
    <b>❌</b> "The effect is big and important." → <b>✅</b> P-value says nothing about SIZE. A tiny difference can have p=0.001 with enough data.
    <br><br>
    <b>❌</b> "p < 0.05 means it's definitely real." → <b>✅</b> 1 in 20 "significant" results is a false alarm by definition.
    </div>""", unsafe_allow_html=True)

    # ── A/B Test Simulator ──
    st.markdown("---")
    st.markdown("### 🎮 A/B Test Simulator")
    st.caption("Test a new pizza menu. Does it actually improve sales?")

    c1, c2 = st.columns(2)
    conv_a = c1.slider("🔵 Old menu conversion %:", 1.0, 20.0, 10.0, 0.5, key="m4_ca")
    conv_b = c2.slider("🟢 New menu conversion %:", 1.0, 20.0, 11.0, 0.5, key="m4_cb")
    n_users = st.slider("👥 Customers per group:", 100, 50000, 5000, 500, key="m4_users")

    np.random.seed(42)
    group_a = np.random.binomial(1, conv_a/100, n_users)
    group_b = np.random.binomial(1, conv_b/100, n_users)
    t_ab, p_ab = ttest_ind(group_a, group_b)
    lift = (group_b.mean() - group_a.mean()) / group_a.mean() * 100 if group_a.mean() > 0 else 0

    mc = st.columns(4)
    mc[0].metric("Old Menu", f"{group_a.mean()*100:.2f}%")
    mc[1].metric("New Menu", f"{group_b.mean()*100:.2f}%", f"{lift:+.1f}% lift")
    mc[2].metric("P-Value", f"{p_ab:.4f}")
    mc[3].metric("Significant?", "✅ Yes!" if p_ab < 0.05 else "❌ Not yet")

    if p_ab < 0.05:
        st.markdown(f"""<div class="green-box">✅ <b>p = {p_ab:.4f}</b> — Ship it! 🚀</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="red-box">❌ <b>p = {p_ab:.4f}</b> — Need more data or a bigger effect.</div>""", unsafe_allow_html=True)

    # ── Decision Flow ──
    st.markdown("---")
    st.markdown("### 🗺️ The Decision Flow: When to Use What")
    st.markdown("""<div class="key-box">
    <b>1. Is ONE value weird?</b> → Z-Score<br>
    <b>2. Is a GROUP different from a target?</b> → One-sample T-Test<br>
    <b>3. Are TWO GROUPS different?</b> → Two-sample T-Test<br>
    <b>4. Did the SAME group change?</b> → Paired T-Test<br>
    <b>5. Check p < 0.05?</b> → Statistically significant<br>
    <b>6. Check effect size?</b> → Practically meaningful
    </div>""", unsafe_allow_html=True)

    check_quiz("m4_q1",
        "Store #1 has z=3.2. East vs West t-test gives p=0.03. What should you do?",
        ["Ignore both — statistics is unreliable", "Investigate Store #1 AND roll out East's strategy to West", "Only look at Store #1, ignore the regional difference"],
        1,
        "z=3.2 means Store #1 is genuinely exceptional — learn from them. p=0.03 means East is significantly better — find out why and replicate it."
    )

    iq([
        {"q": "Explain p-value to a non-technical stakeholder.", "d": "Medium", "c": ["Meta", "Google"],
         "a": "A p-value answers: 'If there's truly no effect, how likely would we see results this extreme by pure chance?' p < 0.05 = 'less than 5% chance this is luck.' <b>Critical:</b> p-value does NOT tell you how big the effect is.",
         "t": "Always mention what p-value does NOT mean."},
        {"q": "When do you use Z-score vs T-test?", "d": "Medium", "c": ["Google", "Amazon"],
         "a": "<b>Z-score:</b> Compare ONE value to a population (is this store unusual?). <b>T-test:</b> Compare TWO GROUPS to each other (is East ≠ West?). Z-test needs known σ (rare). T-test estimates σ from the sample (use this 99% of the time).",
         "t": "Say 'I use t-test 99% of the time' — shows practical experience."},
        {"q": "What's the difference between Type I and Type II errors?", "d": "Medium", "c": ["Amazon", "Apple"],
         "a": "<b>Type I (False Positive):</b> You say there's an effect when there isn't. <b>Type II (False Negative):</b> You miss a real effect. <b>Tradeoff:</b> Reducing one increases the other. In A/B testing, Type I is usually worse (launching bad features).",
         "t": "Give a real-world example and say which error is worse in that context."},
        {"q": "Your A/B test shows p=0.04. Should you launch?", "d": "Hard", "c": ["Meta", "Google", "Netflix"],
         "a": "Not automatically. Check: (1) Effect size — is the improvement meaningful? (2) Multiple comparisons — did we test many metrics? (3) Guardrail metrics — did anything get worse? (4) Novelty effect — is it just because it's new? Recommend: 'Promising, but validate with a longer test.'",
         "t": "Show you don't blindly trust p-values. This is a CLASSIC Meta/Google question."},
    ])




# ═══════════════════════════════════════
# MODULE 5: CORRELATION
# ═══════════════════════════════════════
elif module == "🔗 M5: Correlation":
    st.markdown("# 🔗 Module 5: Correlation & Relationships")
    st.caption("Week 3 · Do two things move together? And the biggest trap in data science.")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> You notice that cities with more ice cream shops also have more crime.
    Does ice cream cause crime? 🍦🔪
    <br><br>
    Of course not! Both go up in <b>summer</b> (more people outside = more ice cream AND more crime).
    The hidden factor (summer) is called a <b>confounding variable</b>.
    <br><br>
    <b>Correlation</b> measures how two things move together. It's a number from <b>-1 to +1</b>:
    <br>• <b>+1:</b> They go up together perfectly (study hours ↔ grades)
    <br>• <b>0:</b> No relationship (shoe size ↔ favorite color)
    <br>• <b>-1:</b> One goes up, the other goes down (price ↔ demand)
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="red-box">
    🚨 <b>THE GOLDEN RULE:</b> Correlation does NOT mean causation.
    Just because two things move together doesn't mean one causes the other.
    Only controlled experiments (A/B tests) can prove causation.
    </div>""", unsafe_allow_html=True)

    # ── Interactive Correlation ──
    st.markdown("### 🎮 Try It: See What Different Correlations Look Like")
    st.caption("Drag the slider to change the relationship strength.")

    target_r = st.slider("Correlation strength (r):", -1.0, 1.0, 0.7, 0.05, key="m5_r")

    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    noise = np.random.normal(0, 1, 200)
    y = target_r * x + np.sqrt(max(0.01, 1 - target_r**2)) * noise
    actual_r = np.corrcoef(x, y)[0, 1]

    abs_r = abs(actual_r)
    strength = "💪 Very strong" if abs_r > 0.8 else "Strong" if abs_r > 0.6 else "Moderate" if abs_r > 0.4 else "Weak" if abs_r > 0.2 else "🤷 None"
    direction = "positive (both go up together)" if actual_r > 0.05 else "negative (one goes up, other goes down)" if actual_r < -0.05 else "no relationship"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#22d3a7', size=5, opacity=0.6)))
    z = np.polyfit(x, y, 1); x_l = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(x=x_l, y=np.polyval(z, x_l), mode='lines', line=dict(color='#f45d6d', width=2, dash='dash'), name='Trend line'))
    fig.update_layout(height=350, title=f"r = {actual_r:.2f} — {strength} {direction}", xaxis_title="Variable X", yaxis_title="Variable Y", **DL)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="green-box">
    💡 <b>What you're seeing:</b> Each dot is one data point. The trend line shows the overall direction.
    <br>• r near +1 → dots form a tight upward line
    <br>• r near 0 → dots are a random cloud (no pattern)
    <br>• r near -1 → dots form a tight downward line
    <br><br><b>Drag the slider</b> from -1 to +1 and watch the pattern change!
    </div>""", unsafe_allow_html=True)

    # ── Correlation Heatmap ──
    st.markdown("---")
    st.markdown("### 🗺️ Correlation Heatmap: The Big Picture")
    st.caption("A heatmap shows ALL relationships at once. Green = positive, Red = negative, Dark = no relationship.")

    np.random.seed(42); n = 300
    age = np.random.normal(35, 10, n)
    hm = pd.DataFrame({
        "Age": age,
        "Income": age*1200 + np.random.normal(0, 8000, n),
        "Spending": age*800 + np.random.normal(0, 5000, n),
        "Satisfaction": np.random.normal(7, 2, n)
    })
    corr = hm.corr()
    fig_hm = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,'#f45d6d'],[0.5,'#1a1d2e'],[1,'#22d3a7']], zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=13)))
    fig_hm.update_layout(height=380, title="Which variables are related?", **DL)
    st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="green-box">
    💡 <b>How to read this:</b> Age, Income, and Spending are all correlated (older people tend to earn and spend more).
    But Satisfaction has near-zero correlation with everything — it's independent.
    <br><br>In real data science, this is your <b>first step in feature selection</b> — find which variables are related to your target.
    </div>""", unsafe_allow_html=True)

    # ── Quiz ──
    check_quiz("m5_q1",
        "A study finds that countries with more Nobel Prize winners also consume more chocolate. What should you conclude?",
        ["Chocolate makes you smarter!", "There's a hidden factor (like wealth) causing both", "Nobel Prizes cause chocolate consumption"],
        1,
        "Wealthy countries can afford both more chocolate AND better education/research. Wealth is the confounding variable. Correlation ≠ causation!"
    )

    iq([
        {"q": "Explain correlation to a non-technical PM.", "d": "Easy", "c": ["Google", "Meta"],
         "a": "Correlation measures how two things move together, from -1 to +1. Example: 'Customers who call support more are more likely to churn (r=0.65). But support calls don't CAUSE churn — the underlying service problems cause both.'",
         "t": "Always end with 'correlation ≠ causation.'"},
        {"q": "Pearson vs Spearman — when do you use each?", "d": "Medium", "c": ["Google", "Netflix"],
         "a": "<b>Pearson:</b> Linear relationships, continuous data. <b>Spearman:</b> Monotonic relationships (consistently up/down but not necessarily a straight line), ordinal data, robust to outliers.",
         "t": "Mention that Spearman is often more useful for feature selection in ML."},
        {"q": "Two features have r=0.95. Should you keep both in your model?", "d": "Medium", "c": ["Amazon", "Google"],
         "a": "Probably not — they carry the same information (collinear). Keep the one more correlated with the target, or combine them. Tree-based models handle this better than linear models.",
         "t": "Mention VIF > 5-10 as a formal collinearity check."},
    ])


# ═══════════════════════════════════════
# MODULE 6: REGRESSION INTUITION
# ═══════════════════════════════════════
elif module == "📉 M6: Regression Intuition":
    st.markdown("# 📉 Module 6: Regression Intuition")
    st.caption("Week 4 · Drawing the best line through data — where statistics meets machine learning.")

    st.markdown("""<div class="story-box">
    <b>🎬 The Story:</b> You're a real estate agent. A client asks: "I have a 1,500 sq ft house. What should I price it?"
    <br><br>
    You look at recent sales: 1,000 sqft sold for $200K, 1,200 sqft for $240K, 1,800 sqft for $350K...
    You see a pattern — bigger houses cost more. If you could draw a <b>line</b> through this data,
    you could predict the price for ANY size.
    <br><br>
    That line is <b>linear regression</b>: <b>price = slope × sqft + base price</b>
    <br>• <b>Slope:</b> "For every extra square foot, price goes up by $___"
    <br>• <b>Intercept:</b> The starting price (when sqft = 0, which is theoretical)
    <br><br>
    This is the simplest ML model — and understanding it deeply makes everything else click.
    </div>""", unsafe_allow_html=True)

    # ── Interactive Regression ──
    st.markdown("### 🎮 Try It: Fit the Line Yourself!")
    st.caption("Drag the slope and intercept to match the data. Then compare with the computer's best answer.")

    np.random.seed(42)
    x_data = np.linspace(0, 10, 30)
    y_true = 3 * x_data + 10 + np.random.normal(0, 5, 30)

    c1, c2 = st.columns(2)
    user_m = c1.slider("📐 Your slope (how steep?):", 0.0, 6.0, 1.0, 0.1, key="m6_m")
    user_b = c2.slider("📍 Your intercept (where does it start?):", 0.0, 20.0, 5.0, 0.5, key="m6_b")

    best_m, best_b = np.polyfit(x_data, y_true, 1)
    user_pred = user_m * x_data + user_b
    best_pred = best_m * x_data + best_b
    user_mse = np.mean((y_true - user_pred)**2)
    best_mse = np.mean((y_true - best_pred)**2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', marker=dict(color='#22d3a7', size=8), name='Actual data'))
    fig.add_trace(go.Scatter(x=x_data, y=user_pred, mode='lines', line=dict(color='#f5b731', width=2.5), name=f'Your line'))
    fig.add_trace(go.Scatter(x=x_data, y=best_pred, mode='lines', line=dict(color='#7c6aff', width=2, dash='dash'), name=f'Best fit'))
    fig.update_layout(height=380, title="Can you beat the computer?", xaxis_title="X", yaxis_title="Y", **DL)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    mc = st.columns(3)
    mc[0].metric("Your Error", f"{user_mse:.1f}", help="Lower is better!")
    mc[1].metric("Best Possible Error", f"{best_mse:.1f}")
    ratio = user_mse/best_mse if best_mse > 0 else float('inf')
    mc[2].metric("How close?", f"{ratio:.1f}x", help="1.0x = perfect match")

    if ratio < 1.2:
        st.markdown("""<div class="green-box">🎯 <b>Nailed it!</b> Your line is very close to the optimal one. You've got good intuition!</div>""", unsafe_allow_html=True)
    elif ratio < 2.0:
        st.markdown("""<div class="story-box">🤏 <b>Getting close!</b> Try adjusting the slope to about {:.1f} and intercept to about {:.1f}.</div>""".format(best_m, best_b), unsafe_allow_html=True)
    else:
        st.markdown("""<div class="red-box">📐 <b>Keep trying!</b> The best slope is around {:.1f} and intercept around {:.1f}. Drag the sliders closer to those values.</div>""".format(best_m, best_b), unsafe_allow_html=True)

    # ── Residuals ──
    st.markdown("---")
    st.markdown("### 📖 Residuals: How Wrong Is Each Prediction?")

    st.markdown("""<div class="analogy-box">
    🎯 <b>Archery analogy:</b> Each prediction is like shooting an arrow at a target.
    The <b>residual</b> is how far your arrow landed from the bullseye.
    <br><br>
    • Residual = actual value − predicted value<br>
    • Good model → small, random residuals (arrows scattered evenly around the bullseye)<br>
    • Bad model → residuals show a pattern (arrows consistently off to one side)
    </div>""", unsafe_allow_html=True)

    residuals = y_true - best_pred
    fig_r = go.Figure()
    colors_r = ['#22d3a7' if r > 0 else '#f45d6d' for r in residuals]
    fig_r.add_trace(go.Bar(x=list(range(len(residuals))), y=residuals, marker_color=colors_r, opacity=0.7))
    fig_r.add_hline(y=0, line_dash="dash", line_color="#7c6aff")
    fig_r.update_layout(height=250, title="Residuals — green (over-predicted) vs red (under-predicted)", xaxis_title="Data Point", yaxis_title="Error", **DL)
    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="green-box">💡 <b>Good sign:</b> The errors are scattered randomly above and below zero — no pattern.
    If you saw a curve or funnel shape, the model would be missing something important.</div>""", unsafe_allow_html=True)

    # ── Overfitting ──
    st.markdown("---")
    st.markdown("### 📖 Overfitting vs Underfitting: The Goldilocks Problem")

    st.markdown("""<div class="story-box">
    <b>🎬 The Student Analogy:</b>
    <br><br>
    📚 <b>Underfitting</b> = A student who only reads chapter titles. They get the gist but miss all the details. On the exam, they fail.
    <br><br>
    📝 <b>Overfitting</b> = A student who memorizes the textbook word-for-word, including typos. They ace practice tests but bomb the real exam because they memorized noise, not concepts.
    <br><br>
    ✅ <b>Good fit</b> = A student who understands the concepts. They can handle new questions they've never seen before.
    </div>""", unsafe_allow_html=True)

    fit_type = st.radio("See each type:", ["📉 Underfit (too simple)", "✅ Good fit (just right)", "⚠️ Overfit (too complex)"], horizontal=True, key="m6_fit")

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', marker=dict(color='#22d3a7', size=8), name='Data'))
    x_s = np.linspace(0, 10, 200)
    if "Underfit" in fit_type:
        fig_f.add_trace(go.Scatter(x=x_s, y=np.full_like(x_s, y_true.mean()), mode='lines', line=dict(color='#f5b731', width=3), name='Model (flat line)'))
        msg = "The model is too simple — it just predicts the average for everything. It misses the obvious upward trend."
    elif "Good" in fit_type:
        fig_f.add_trace(go.Scatter(x=x_s, y=np.polyval(np.polyfit(x_data, y_true, 1), x_s), mode='lines', line=dict(color='#7c6aff', width=3), name='Model (straight line)'))
        msg = "The model captures the real trend without chasing random noise. It will work well on new data."
    else:
        fig_f.add_trace(go.Scatter(x=x_s, y=np.polyval(np.polyfit(x_data, y_true, 15), x_s).clip(-10, 60), mode='lines', line=dict(color='#f45d6d', width=3), name='Model (wiggly line)'))
        msg = "The model passes through every point perfectly — but it's memorizing noise. On new data, those wiggles will be completely wrong."
    fig_f.update_layout(height=320, title=fit_type.split(" ", 1)[1], **DL)
    st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="{'green-box' if 'Good' in fit_type else 'red-box'}">{msg}</div>""", unsafe_allow_html=True)

    # ── Order Quiz ──
    order_quiz("m6_order",
        "Put these regression steps in the right order:",
        ["Check residuals for patterns", "Collect and explore data", "Fit the model (find best line)", "Split data into train/test"],
        ["Collect and explore data", "Split data into train/test", "Fit the model (find best line)", "Check residuals for patterns"],
        "First explore your data, then split it (so you can test later), then fit the model, then check if the model is any good by looking at residuals."
    )

    iq([
        {"q": "Explain linear regression to a non-technical person.", "d": "Easy", "c": ["Google", "Amazon"],
         "a": "Linear regression draws the <b>best straight line</b> through data to make predictions. Example: 'For every extra square foot, house price goes up by $150.' The line captures the relationship: y = 150×sqft + 50,000.",
         "t": "Use the house price example — everyone understands it."},
        {"q": "What is overfitting? How do you detect and prevent it?", "d": "Medium", "c": ["Google", "Meta", "Amazon"],
         "a": "<b>Overfitting:</b> Model performs great on training data but poorly on new data. <b>Detection:</b> Big gap between training and test accuracy. <b>Prevention:</b> Train/test split, cross-validation, regularization, simpler model, more data.",
         "t": "Use the student analogy: memorizing answers vs understanding concepts."},
        {"q": "What assumptions does linear regression make?", "d": "Hard", "c": ["Google", "Apple"],
         "a": "<b>1. Linearity</b> (relationship is a straight line). <b>2. Independence</b> (observations don't affect each other). <b>3. Constant spread</b> (residuals don't fan out). <b>4. Normal residuals</b> (for inference). <b>5. No multicollinearity</b> (features aren't copies of each other). Mild violations are usually fine with large samples.",
         "t": "Don't just list them — say which ones matter most in practice."},
    ])
