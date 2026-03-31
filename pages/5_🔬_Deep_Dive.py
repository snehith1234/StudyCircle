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
        "🌳 Random Forest",
    ], label_visibility="collapsed")


# ═══════════════════════════════════════
# RANDOM FOREST DEEP DIVE
# ═══════════════════════════════════════
if topic == "🌳 Random Forest":
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
