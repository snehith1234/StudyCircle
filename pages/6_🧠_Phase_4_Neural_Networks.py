# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="🧠 Phase 4: Neural Networks", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
.block-container, [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #000000 !important;
}
.stApp, .stMarkdown, p, span, label { color: #e2e8f0 !important; }
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
.stApp { font-family: 'Inter', sans-serif; }
.concept-card {
    background: linear-gradient(135deg, #1a1d2e, #1f2235);
    border: 1px solid #2d3148; border-radius: 14px; padding: 1.1rem 1.3rem;
    margin: 0.4rem 0; line-height: 1.85; font-size: 0.91rem; color: #c8cfe0;
}
.concept-card b { color: #e2e8f0; }
.math-box {
    background: #252840; border-left: 4px solid #f5b731;
    border-radius: 0 10px 10px 0; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.87rem; color: #c8cfe0; line-height: 1.8;
}
.math-box b { color: #f5b731; }
.insight-box {
    background: linear-gradient(135deg, #1a2a1f, #1f3528);
    border: 1px solid #2a5a3a; border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 0.4rem 0; font-size: 0.87rem; color: #c8d8c0; line-height: 1.7;
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
    st.markdown("## 🧠 Phase 4: Neural Networks")
    st.caption("Interactive Deep Learning Concepts")
    st.divider()
    tab = st.radio("Module:", [
        "🏗️ Architecture",
        "⚡ Single Neuron",
        "📈 Activations",
        "➡️ Forward Pass",
        "📉 Loss Functions",
        "🔄 Backpropagation",
        "🏋️ Training Loop",
        "🖼️ CNNs",
        "🔁 RNNs & LSTMs",
        "🤖 Transformers",
        "📝 NLP Basics",
    ], label_visibility="collapsed")

# ═══════════════════════════════════════
# HELPER: Draw network architecture
# ═══════════════════════════════════════
def draw_network(layers, activations=None, highlight_layer=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.axis('off')

    n_layers = len(layers)
    max_neurons = max(layers)
    x_spacing = 1.0 / (n_layers + 1)
    colors = ['#f5b731', '#5eaeff', '#7c6aff', '#22d3a7']
    labels = ['Input', 'Hidden', 'Hidden', 'Output']
    if activations is None:
        activations = ['—'] + ['ReLU'] * (n_layers - 2) + ['Sigmoid']

    positions = {}
    for i, n in enumerate(layers):
        x = (i + 1) * x_spacing
        y_spacing = 0.8 / max(n, 1)
        y_start = 0.5 - (n - 1) * y_spacing / 2
        for j in range(n):
            y = y_start + j * y_spacing
            positions[(i, j)] = (x, y)

    # Draw connections
    for i in range(n_layers - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                x1, y1 = positions[(i, j)]
                x2, y2 = positions[(i + 1, k)]
                alpha = 0.15 if highlight_layer is None else (0.4 if i == highlight_layer else 0.08)
                ax.plot([x1, x2], [y1, y2], color='#8892b0', alpha=alpha, linewidth=0.8)

    # Draw neurons
    for i, n in enumerate(layers):
        c = colors[min(i, len(colors) - 1)]
        if i == n_layers - 1:
            c = '#22d3a7'
        for j in range(n):
            x, y = positions[(i, j)]
            circle = plt.Circle((x, y), 0.02, color=c, ec='#2d3148', linewidth=1.5, zorder=5)
            ax.add_patch(circle)

    # Labels
    for i, n in enumerate(layers):
        x = (i + 1) * x_spacing
        lbl = labels[min(i, len(labels) - 1)]
        ax.text(x, 0.05, f"{lbl}\n{n} neurons\n{activations[i]}", ha='center', va='center',
                fontsize=9, color='#8892b0', fontfamily='monospace')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Neural Network Architecture", color='#e2e8f0', fontsize=14, pad=10)
    return fig


# ═══════════════════════════════════════
# TAB: Architecture
# ═══════════════════════════════════════
if tab == "🏗️ Architecture":
    st.markdown("# 🏗️ Neural Network Architecture")
    st.caption("Build and visualize your own network")

    st.markdown("""<div class="concept-card">
    A neural network is layers of neurons connected together. The <b>input layer</b> receives your data,
    <b>hidden layers</b> learn intermediate features, and the <b>output layer</b> makes the prediction.
    <br><br>Use the sliders below to build your own network and see how it looks.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        n_hidden = st.slider("Number of hidden layers", 1, 4, 2)
        layer_sizes = [st.slider(f"Hidden layer {i+1} neurons", 1, 10, 4, key=f"hl_{i}") for i in range(n_hidden)]
        input_size = st.slider("Input features", 1, 8, 2)
        output_size = st.slider("Output neurons", 1, 5, 1)

    layers = [input_size] + layer_sizes + [output_size]
    acts = ['—'] + ['ReLU'] * n_hidden + ['Sigmoid' if output_size == 1 else 'Softmax']

    with col2:
        fig = draw_network(layers, acts)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    total_params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
    st.markdown(f"""<div class="math-box">
    <b>Total parameters:</b> {total_params:,}
    <br>Formula: Σ (neurons_in × neurons_out + bias) for each layer
    <br>Layers: {' → '.join(str(l) for l in layers)}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TAB: Single Neuron
# ═══════════════════════════════════════
elif tab == "⚡ Single Neuron":
    st.markdown("# ⚡ A Single Neuron")
    st.caption("The building block of all neural networks")

    st.markdown("""<div class="concept-card">
    A neuron does 3 things: <b>multiply</b> inputs by weights, <b>add</b> a bias, and apply an <b>activation function</b>.
    It's identical to logistic regression — the building block of every neural network.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Inputs (Pizza Store)")
        rating = st.slider("⭐ Rating", 1.0, 5.0, 4.5, 0.1)
        delivery = st.slider("🚚 Delivery (min)", 10, 60, 20)

        st.markdown("#### Weights (Random Init)")
        w1 = st.slider("w₁ (Rating weight)", -2.0, 2.0, 1.0, 0.1)
        w2 = st.slider("w₂ (Delivery weight)", -0.2, 0.2, -0.05, 0.01)
        b = st.slider("b (Bias)", -5.0, 5.0, -2.0, 0.1)

    with col2:
        z = w1 * rating + w2 * delivery + b
        sigmoid = 1 / (1 + np.exp(-z))

        st.markdown("#### Computation")
        st.latex(f"z = {w1:.1f} \\times {rating:.1f} + ({w2:.2f}) \\times {delivery} + ({b:.1f})")
        st.latex(f"z = {w1*rating:.2f} + ({w2*delivery:.2f}) + ({b:.1f}) = {z:.3f}")
        st.latex(f"\\sigma(z) = \\frac{{1}}{{1 + e^{{-{z:.3f}}}}} = {sigmoid:.4f}")

        if sigmoid >= 0.5:
            st.success(f"**Prediction: ✅ Successful ({sigmoid*100:.1f}%)**")
        else:
            st.error(f"**Prediction: ❌ Not Successful ({sigmoid*100:.1f}%)**")

        st.markdown(f"""<div class="insight-box">
        <b>What the weights mean:</b><br>
        w₁ = {w1:.1f}: {'Higher rating helps' if w1 > 0 else 'Higher rating hurts' if w1 < 0 else 'Rating ignored'}<br>
        w₂ = {w2:.2f}: {'Longer delivery helps' if w2 > 0 else 'Longer delivery hurts' if w2 < 0 else 'Delivery ignored'}<br>
        b = {b:.1f}: Baseline offset
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TAB: Activations
# ═══════════════════════════════════════
elif tab == "📈 Activations":
    st.markdown("# 📈 Activation Functions")
    st.caption("Why non-linearity matters")

    st.markdown("""<div class="concept-card">
    Without activation functions, a 100-layer network equals a 1-layer network (just linear math).
    Activations add <b>non-linearity</b>, letting the network learn curves, circles, and complex patterns.
    </div>""", unsafe_allow_html=True)

    z = np.linspace(-6, 6, 300)
    sigmoid_vals = 1 / (1 + np.exp(-z))
    tanh_vals = np.tanh(z)
    relu_vals = np.maximum(0, z)
    leaky_relu_vals = np.where(z > 0, z, 0.01 * z)

    # Gradients
    sig_grad = sigmoid_vals * (1 - sigmoid_vals)
    tanh_grad = 1 - tanh_vals**2
    relu_grad = np.where(z > 0, 1.0, 0.0)

    show_grad = st.checkbox("Show gradients (derivatives)", value=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=sigmoid_vals, name='Sigmoid', line=dict(color='#f5b731', width=3)))
    fig.add_trace(go.Scatter(x=z, y=tanh_vals, name='Tanh', line=dict(color='#7c6aff', width=3)))
    fig.add_trace(go.Scatter(x=z, y=relu_vals, name='ReLU', line=dict(color='#22d3a7', width=3)))
    fig.add_trace(go.Scatter(x=z, y=leaky_relu_vals, name='Leaky ReLU', line=dict(color='#5eaeff', width=2, dash='dash')))

    if show_grad:
        fig.add_trace(go.Scatter(x=z, y=sig_grad, name='Sigmoid gradient', line=dict(color='#f5b731', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=z, y=tanh_grad, name='Tanh gradient', line=dict(color='#7c6aff', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=z, y=relu_grad, name='ReLU gradient', line=dict(color='#22d3a7', width=2, dash='dot')))

    fig.update_layout(**DL, title="Activation Functions", xaxis_title="z (input)", yaxis_title="output", height=450)
    fig.add_hline(y=0, line_dash="dash", line_color="#2d3148")
    fig.add_vline(x=0, line_dash="dash", line_color="#2d3148")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="math-box">
        <b>Sigmoid</b><br>σ(z) = 1/(1+e⁻ᶻ)<br>Range: (0, 1)<br>Max gradient: <b>0.25</b><br>
        Use: output layer (binary)<br>⚠️ Vanishing gradients
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="math-box">
        <b>Tanh</b><br>tanh(z) = (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)<br>Range: (−1, 1)<br>Max gradient: <b>1.0</b><br>
        Use: RNNs, centered output<br>⚠️ Still vanishes for large |z|
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="insight-box">
        <b>ReLU ✅ Default</b><br>ReLU(z) = max(0, z)<br>Range: [0, ∞)<br>Gradient: <b>0 or 1</b><br>
        Use: hidden layers<br>✅ No vanishing gradient!
        </div>""", unsafe_allow_html=True)

    with st.expander("🔬 Why does sigmoid's gradient max at 0.25?"):
        st.markdown("""
        The sigmoid gradient is: **σ'(z) = σ(z) × (1 − σ(z))**

        Both σ(z) and (1−σ(z)) are between 0 and 1. Multiplying two numbers in (0,1) always gives something smaller.
        The product is maximized when both are equal: σ(z) = 0.5, giving 0.5 × 0.5 = **0.25**.

        Through 10 layers: 0.25¹⁰ = 0.00000095 — the gradient reaching layer 1 is essentially **zero**.
        """)
        st.latex(r"\sigma'(z) = \sigma(z) \times (1 - \sigma(z)) \leq 0.25")


# ═══════════════════════════════════════
# TAB: Forward Pass
# ═══════════════════════════════════════
elif tab == "➡️ Forward Pass":
    st.markdown("# ➡️ Forward Pass")
    st.caption("How data flows through the network")

    st.markdown("""<div class="concept-card">
    The forward pass computes the prediction: data enters the input layer, gets transformed by each hidden layer
    (weighted sum → activation), and produces an output. We'll trace Store S1 (Rating=4.5, Delivery=20) through a 2-2-1 network.
    </div>""", unsafe_allow_html=True)

    np.random.seed(42)
    W1 = np.array([[0.5, -0.3], [-0.02, 0.04]])
    b1 = np.array([-1.0, 0.5])
    W2 = np.array([[0.8], [-0.6]])
    b2 = np.array([-0.1])

    x = np.array([4.5, 20.0])

    step = st.radio("Step:", ["1. Input", "2. Hidden Layer", "3. Output Layer", "4. Full Summary"], horizontal=True)

    if step == "1. Input":
        st.markdown(f"""<div class="math-box">
        <b>Input:</b> x₁ = {x[0]} (Rating), x₂ = {x[1]} (Delivery)
        <br>These are the raw features from our pizza store data.
        </div>""", unsafe_allow_html=True)

    elif step == "2. Hidden Layer":
        z1 = W1.T @ x + b1
        a1 = 1 / (1 + np.exp(-z1))
        st.markdown(f"""<div class="math-box">
        <b>Hidden neuron h₁:</b><br>
        z₁ = {W1[0,0]:.1f}×{x[0]} + ({W1[1,0]:.2f})×{x[1]} + ({b1[0]:.1f}) = <b>{z1[0]:.3f}</b><br>
        a₁ = σ({z1[0]:.3f}) = <b>{a1[0]:.3f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="math-box">
        <b>Hidden neuron h₂:</b><br>
        z₂ = {W1[0,1]:.1f}×{x[0]} + ({W1[1,1]:.2f})×{x[1]} + ({b1[1]:.1f}) = <b>{z1[1]:.3f}</b><br>
        a₂ = σ({z1[1]:.3f}) = <b>{a1[1]:.3f}</b>
        </div>""", unsafe_allow_html=True)

    elif step == "3. Output Layer":
        z1 = W1.T @ x + b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = W2.T @ a1 + b2
        a2 = 1 / (1 + np.exp(-z2))
        st.markdown(f"""<div class="math-box">
        <b>Output neuron:</b><br>
        z_out = {W2[0,0]:.1f}×{a1[0]:.3f} + ({W2[1,0]:.1f})×{a1[1]:.3f} + ({b2[0]:.1f}) = <b>{z2[0]:.3f}</b><br>
        ŷ = σ({z2[0]:.3f}) = <b>{a2[0]:.4f}</b> = {a2[0]*100:.1f}%
        </div>""", unsafe_allow_html=True)
        if a2[0] >= 0.5:
            st.success(f"Prediction: ✅ Successful ({a2[0]*100:.1f}%)")
        else:
            st.error(f"Prediction: ❌ Not Successful ({a2[0]*100:.1f}%)")

    else:
        z1 = W1.T @ x + b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = W2.T @ a1 + b2
        a2 = 1 / (1 + np.exp(-z2))
        st.markdown(f"""<div class="concept-card">
        <b>Full forward pass for S1 (Rating=4.5, Delivery=20):</b><br><br>
        Input: [{x[0]}, {x[1]}]<br>
        → Hidden: z=[{z1[0]:.3f}, {z1[1]:.3f}] → a=[{a1[0]:.3f}, {a1[1]:.3f}]<br>
        → Output: z={z2[0]:.3f} → ŷ=<b>{a2[0]:.4f}</b> ({a2[0]*100:.1f}%)<br><br>
        Actual: 1 (Successful) | Loss = {-np.log(a2[0]):.3f}
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TAB: Loss Functions
# ═══════════════════════════════════════
elif tab == "📉 Loss Functions":
    st.markdown("# 📉 Loss Functions")
    st.caption("How the model measures 'how wrong am I?'")

    st.markdown("""<div class="concept-card">
    The loss function tells the model how far off its prediction is from the truth.
    Different tasks need different loss functions — each is derived from the probability distribution that matches the task.
    </div>""", unsafe_allow_html=True)

    loss_type = st.radio("Loss type:", ["Binary Cross-Entropy", "MSE (Regression)"], horizontal=True)

    if loss_type == "Binary Cross-Entropy":
        p = np.linspace(0.01, 0.99, 200)
        loss_y1 = -np.log(p)
        loss_y0 = -np.log(1 - p)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p, y=loss_y1, name='When actual = 1', line=dict(color='#22d3a7', width=3)))
        fig.add_trace(go.Scatter(x=p, y=loss_y0, name='When actual = 0', line=dict(color='#f45d6d', width=3)))
        fig.update_layout(**DL, title="Binary Cross-Entropy Loss", xaxis_title="Predicted probability (ŷ)", yaxis_title="Loss", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="math-box">
        <b>Formula:</b> Loss = −[y × log(ŷ) + (1−y) × log(1−ŷ)]<br><br>
        When actual = 1 (green): predicting ŷ=0.9 → loss=0.105 (good), ŷ=0.1 → loss=2.303 (terrible!)<br>
        When actual = 0 (red): predicting ŷ=0.1 → loss=0.105 (good), ŷ=0.9 → loss=2.303 (terrible!)<br><br>
        <b>Key insight:</b> Confident wrong predictions are penalized exponentially more than uncertain ones.
        </div>""", unsafe_allow_html=True)

    else:
        y_true = 85.0
        y_pred = np.linspace(30, 140, 200)
        mse = (y_true - y_pred) ** 2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=mse, name='MSE', line=dict(color='#5eaeff', width=3)))
        fig.add_vline(x=y_true, line_dash="dash", line_color="#22d3a7", annotation_text=f"True value = {y_true}")
        fig.update_layout(**DL, title="Mean Squared Error", xaxis_title="Predicted value (ŷ)", yaxis_title="Loss (y−ŷ)²", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")


# ═══════════════════════════════════════
# TAB: Backpropagation
# ═══════════════════════════════════════
elif tab == "🔄 Backpropagation":
    st.markdown("# 🔄 Backpropagation — Multi-Layer")
    st.caption("Watch gradients flow through a 2-layer network with actual numbers")

    st.markdown("""<div class="concept-card">
    We trace <b>S1 (Rating=4.5, Delivery=20, Actual=1)</b> through a <b>2-layer network</b>:
    2 inputs → 1 hidden neuron → 1 output. During <b>forward pass</b>, weights are FIXED.
    During <b>backprop</b>, we compute gradients for EVERY weight — both output AND hidden layer.
    </div>""", unsafe_allow_html=True)

    x1_v, x2_v, y_t = 4.5, 20.0, 1.0
    eta = st.slider("Learning rate (η)", 0.01, 1.0, 0.1, 0.01)

    wh1, wh2, bh = 0.5, -0.02, -1.0
    wo, bo = 0.8, -0.1

    bp_step = st.radio("Step:", [
        "1. Forward: Hidden Layer",
        "2. Forward: Output Layer",
        "3. Forward: Loss",
        "4. Backward: Output Gradients",
        "5. Backward: Hidden Gradients (Chain Rule)",
        "6. Update ALL Weights",
        "7. Full Picture"
    ], horizontal=False)

    zh = wh1 * x1_v + wh2 * x2_v + bh
    ah = 1 / (1 + np.exp(-zh))
    zo = wo * ah + bo
    ao = 1 / (1 + np.exp(-zo))
    loss = -(y_t * np.log(ao) + (1 - y_t) * np.log(1 - ao))

    dL_dzo = ao - y_t
    dL_dwo = dL_dzo * ah
    dL_dbo = dL_dzo
    dzo_dah = wo
    dL_dah = dL_dzo * dzo_dah
    dah_dzh = ah * (1 - ah)
    dL_dzh = dL_dah * dah_dzh
    dL_dwh1 = dL_dzh * x1_v
    dL_dwh2 = dL_dzh * x2_v
    dL_dbh = dL_dzh

    wo_n = wo - eta * dL_dwo
    bo_n = bo - eta * dL_dbo
    wh1_n = wh1 - eta * dL_dwh1
    wh2_n = wh2 - eta * dL_dwh2
    bh_n = bh - eta * dL_dbh

    zh2 = wh1_n * x1_v + wh2_n * x2_v + bh_n
    ah2 = 1 / (1 + np.exp(-zh2))
    zo2 = wo_n * ah2 + bo_n
    ao2 = 1 / (1 + np.exp(-zo2))
    loss2 = -(y_t * np.log(ao2) + (1 - y_t) * np.log(1 - ao2))

    if bp_step == "1. Forward: Hidden Layer":
        st.markdown("### ➡️ Step 1: Data Enters Hidden Layer")
        st.markdown("""<div class="concept-card">
        <b>Weights are FIXED during forward pass.</b> They don't change — data just flows through.
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="math-box">
        <b>z_h = w_h1 × x₁ + w_h2 × x₂ + b_h</b><br>
        = {wh1} × {x1_v} + ({wh2}) × {x2_v} + ({bh})<br>
        = {wh1*x1_v:.2f} + ({wh2*x2_v:.2f}) + ({bh})<br>
        <b>z_h = {zh:.4f}</b><br><br>
        a_h = σ({zh:.4f}) = <b>{ah:.6f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-box">
        The hidden neuron outputs {ah:.4f}. This intermediate value feeds into the output layer.
        </div>""", unsafe_allow_html=True)

    elif bp_step == "2. Forward: Output Layer":
        st.markdown("### ➡️ Step 2: Hidden → Output")
        st.markdown(f"""<div class="math-box">
        <b>z_o = w_o × a_h + b_o</b><br>
        = {wo} × {ah:.6f} + ({bo}) = <b>{zo:.6f}</b><br><br>
        ŷ = σ({zo:.6f}) = <b>{ao:.6f}</b> = {ao*100:.2f}%
        </div>""", unsafe_allow_html=True)
        if ao >= 0.5:
            st.success(f"Prediction: ✅ {ao*100:.2f}% — Actual: 1")
        else:
            st.error(f"Prediction: ❌ {ao*100:.2f}% — Actual: 1")

    elif bp_step == "3. Forward: Loss":
        st.markdown("### ➡️ Step 3: Compute Loss")
        st.markdown(f"""<div class="math-box">
        Loss = −log({ao:.6f}) = <b>{loss:.6f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-box">
        Now backprop adjusts ALL 5 weights (w_h1, w_h2, b_h, w_o, b_o) to reduce this loss.
        </div>""", unsafe_allow_html=True)

    elif bp_step == "4. Backward: Output Gradients":
        st.markdown("### ⬅️ Step 4: Output Layer Gradients")
        st.markdown(f"""<div class="math-box">
        <b>∂Loss/∂z_o = ŷ − y = {ao:.6f} − 1 = {dL_dzo:.6f}</b><br><br>
        <b>∂Loss/∂w_o</b> = (ŷ − y) × a_h = {dL_dzo:.6f} × {ah:.6f} = <b>{dL_dwo:.6f}</b><br>
        <b>∂Loss/∂b_o</b> = (ŷ − y) = <b>{dL_dbo:.6f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-box">
        Output gradients are straightforward. But the hidden layer doesn't see the loss directly...
        </div>""", unsafe_allow_html=True)

    elif bp_step == "5. Backward: Hidden Gradients (Chain Rule)":
        st.markdown("### ⬅️ Step 5: Hidden Layer Gradients via Chain Rule")
        st.markdown("""<div class="concept-card">
        The hidden layer doesn't connect to the loss directly. Its gradient must flow
        <b>backward through the output layer</b>. This is the "back" in backpropagation.
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="math-box">
        <b>5a. Error signal reaches hidden layer:</b><br>
        ∂Loss/∂a_h = (ŷ − y) × w_o = {dL_dzo:.6f} × {wo} = <b>{dL_dah:.6f}</b><br>
        <i>The output weight w_o={wo} scales how much error the hidden neuron "feels"</i><br><br>
        <b>5b. Pass through hidden sigmoid:</b><br>
        σ'(z_h) = a_h × (1 − a_h) = {ah:.6f} × {1-ah:.6f} = <b>{dah_dzh:.6f}</b><br>
        ∂Loss/∂z_h = {dL_dah:.6f} × {dah_dzh:.6f} = <b>{dL_dzh:.6f}</b><br>
        <i>Sigmoid derivative ({dah_dzh:.4f}) shrinks the gradient — vanishing gradient!</i><br><br>
        <b>5c. Hidden weight gradients:</b><br>
        ∂Loss/∂w_h1 = {dL_dzh:.6f} × {x1_v} = <b>{dL_dwh1:.6f}</b><br>
        ∂Loss/∂w_h2 = {dL_dzh:.6f} × {x2_v} = <b>{dL_dwh2:.6f}</b><br>
        ∂Loss/∂b_h = <b>{dL_dbh:.6f}</b>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="insight-box">
        Hidden gradient ({abs(dL_dwh1):.6f}) is much smaller than output gradient ({abs(dL_dwo):.6f})
        because it passed through sigmoid derivative ({dah_dzh:.4f}). With 10 layers, this shrinks to near zero.
        </div>""", unsafe_allow_html=True)

    elif bp_step == "6. Update ALL Weights":
        st.markdown("### 🔧 Step 6: Update Every Weight")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Output Layer**")
            st.metric("w_o", f"{wo_n:.4f}", f"{wo_n - wo:+.4f}")
            st.metric("b_o", f"{bo_n:.4f}", f"{bo_n - bo:+.4f}")
        with col2:
            st.markdown("**Hidden Layer**")
            st.metric("w_h1 (Rating)", f"{wh1_n:.4f}", f"{wh1_n - wh1:+.4f}")
            st.metric("w_h2 (Delivery)", f"{wh2_n:.4f}", f"{wh2_n - wh2:+.4f}")
            st.metric("b_h", f"{bh_n:.4f}", f"{bh_n - bh:+.4f}")

        st.markdown(f"""<div class="insight-box">
        Output weights changed more ({wo_n - wo:+.4f}) than hidden weights ({wh1_n - wh1:+.6f})
        — the vanishing gradient in action.
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("### 📊 Before vs After")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="math-box"><b>BEFORE</b><br>
            Prediction: {ao*100:.2f}% | Loss: {loss:.4f}</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="insight-box"><b>AFTER</b><br>
            Prediction: {ao2*100:.2f}% | Loss: {loss2:.4f}</div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Prediction", f"{ao2*100:.1f}%", f"{(ao2 - ao)*100:+.1f}%")
        col_b.metric("Loss", f"{loss2:.4f}", f"{loss_new - loss:+.4f}" if 'loss_new' in dir() else f"{loss2 - loss:+.4f}")

        labels = ['∂L/∂w_o', '∂L/∂b_o', '∂L/∂w_h1', '∂L/∂w_h2', '∂L/∂b_h']
        grads = [abs(dL_dwo), abs(dL_dbo), abs(dL_dwh1), abs(dL_dwh2), abs(dL_dbh)]
        colors = ['#22d3a7', '#22d3a7', '#f5b731', '#f5b731', '#f5b731']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=grads, marker_color=colors,
                             text=[f"{g:.5f}" for g in grads], textposition='outside',
                             textfont=dict(color='#e2e8f0')))
        fig.update_layout(**DL, title="Output (green) vs Hidden (yellow) gradients", height=350)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════
# TAB: Training Loop
# ═══════════════════════════════════════
elif tab == "🏋️ Training Loop":
    st.markdown("# 🏋️ Training Loop")
    st.caption("Watch the network learn in real-time")

    st.markdown("""<div class="concept-card">
    Training repeats: forward pass → compute loss → backward pass → update weights.
    The <b>learning rate</b> controls step size. Too large → overshoots. Too small → crawls.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        lr = st.slider("Learning rate (η)", 0.001, 2.0, 0.1, 0.001, format="%.3f")
        epochs = st.slider("Epochs", 10, 500, 100)
        run = st.button("🚀 Train!", use_container_width=True)

    # Simple 1D training: learn y = 2x + 1
    np.random.seed(42)
    X = np.random.randn(50) * 2
    y = 2 * X + 1 + np.random.randn(50) * 0.5

    if run or 'losses' not in st.session_state:
        w, b_val = 0.0, 0.0
        losses = []
        w_history = []
        b_history = []

        for epoch in range(epochs):
            y_pred = w * X + b_val
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)
            w_history.append(w)
            b_history.append(b_val)

            dw = -2 * np.mean(X * (y - y_pred))
            db = -2 * np.mean(y - y_pred)
            w -= lr * dw
            b_val -= lr * db

        st.session_state.losses = losses
        st.session_state.w_final = w
        st.session_state.b_final = b_val
        st.session_state.w_history = w_history
        st.session_state.b_history = b_history

    with col2:
        if 'losses' in st.session_state:
            losses = st.session_state.losses

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss',
                                     line=dict(color='#f45d6d', width=2)))
            fig.update_layout(**DL, title="Loss Curve", xaxis_title="Epoch", yaxis_title="MSE Loss", height=350)
            st.plotly_chart(fig, use_container_width=True)

            final_loss = losses[-1]
            w_f = st.session_state.w_final
            b_f = st.session_state.b_final

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Final Loss", f"{final_loss:.4f}")
            col_b.metric("Learned w", f"{w_f:.3f}", delta=f"true: 2.0")
            col_c.metric("Learned b", f"{b_f:.3f}", delta=f"true: 1.0")

            if final_loss > 10:
                st.warning("⚠️ Loss is high — try a smaller learning rate!")
            elif final_loss < 0.5:
                st.success("✅ Good convergence!")

            with st.expander("📊 See the fit"):
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data', marker=dict(color='#5eaeff', size=6)))
                x_line = np.linspace(X.min(), X.max(), 100)
                fig2.add_trace(go.Scatter(x=x_line, y=w_f * x_line + b_f, mode='lines', name=f'Learned: y={w_f:.2f}x+{b_f:.2f}',
                                          line=dict(color='#22d3a7', width=3)))
                fig2.add_trace(go.Scatter(x=x_line, y=2 * x_line + 1, mode='lines', name='True: y=2x+1',
                                          line=dict(color='#f5b731', width=2, dash='dash')))
                fig2.update_layout(**DL, title="Data vs Learned Line", height=350)
                st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════
# TAB: CNNs
# ═══════════════════════════════════════
elif tab == "🖼️ CNNs":
    st.markdown("# 🖼️ Convolutional Neural Networks")
    st.caption("How machines see images")

    cnn_view = st.radio("View:", ["Convolution Demo", "Pooling Demo", "Full Architecture", "Transfer Learning"], horizontal=True)

    if cnn_view == "Convolution Demo":
        st.markdown("""<div class="concept-card">
        A <b>filter</b> (3×3 matrix of weights) slides across the image, computing a dot product at each position.
        Different filters detect different patterns — edges, corners, textures.
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Input Image (5×5)")
            img = np.array([
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0]
            ], dtype=float)

            filter_type = st.selectbox("Filter type:", ["Vertical Edge", "Horizontal Edge", "Identity", "Blur"])
            filters = {
                "Vertical Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                "Horizontal Edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
                "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                "Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0,
            }
            filt = filters[filter_type]

            fig_img = go.Figure(data=go.Heatmap(z=img[::-1], colorscale='Blues', showscale=False,
                                                 text=img[::-1].astype(int).astype(str), texttemplate="%{text}"))
            fig_img.update_layout(**DL, title="Input Image", height=300, width=300)
            st.plotly_chart(fig_img, use_container_width=True)

            st.markdown(f"**Filter ({filter_type}):**")
            fig_f = go.Figure(data=go.Heatmap(z=filt[::-1], colorscale='RdBu', showscale=False,
                                               text=np.round(filt[::-1], 2).astype(str), texttemplate="%{text}"))
            fig_f.update_layout(**DL, title=f"{filter_type} Filter (3×3)", height=200, width=200)
            st.plotly_chart(fig_f, use_container_width=True)

        with col2:
            st.markdown("#### Output Feature Map (3×3)")
            out_size = img.shape[0] - filt.shape[0] + 1
            output = np.zeros((out_size, out_size))
            for i in range(out_size):
                for j in range(out_size):
                    patch = img[i:i+3, j:j+3]
                    output[i, j] = np.sum(patch * filt)

            fig_out = go.Figure(data=go.Heatmap(z=output[::-1], colorscale='Viridis', showscale=True,
                                                 text=np.round(output[::-1], 1).astype(str), texttemplate="%{text}"))
            fig_out.update_layout(**DL, title="Convolution Output", height=300, width=300)
            st.plotly_chart(fig_out, use_container_width=True)

            st.markdown(f"""<div class="math-box">
            Output size = (input − filter + 1) = ({img.shape[0]} − {filt.shape[0]} + 1) = <b>{out_size}</b><br>
            Each output value = sum of (input patch × filter)
            </div>""", unsafe_allow_html=True)

    elif cnn_view == "Pooling Demo":
        st.markdown("""<div class="concept-card">
        <b>Max Pooling</b> takes the maximum value in each 2×2 region, reducing size by 4× while keeping the strongest features.
        </div>""", unsafe_allow_html=True)

        pool_input = np.array([[1, 3, 2, 1], [0, 4, 6, 2], [1, 3, 1, 5], [0, 2, 3, 2]], dtype=float)
        pool_output = np.array([
            [max(pool_input[i:i+2, j:j+2].flatten()) for j in range(0, 4, 2)]
            for i in range(0, 4, 2)
        ])

        col1, col2 = st.columns(2)
        with col1:
            fig_p = go.Figure(data=go.Heatmap(z=pool_input[::-1], colorscale='YlOrRd', showscale=False,
                                               text=pool_input[::-1].astype(int).astype(str), texttemplate="%{text}"))
            fig_p.update_layout(**DL, title="Input (4×4)", height=300)
            st.plotly_chart(fig_p, use_container_width=True)
        with col2:
            fig_po = go.Figure(data=go.Heatmap(z=np.array(pool_output)[::-1], colorscale='YlOrRd', showscale=False,
                                                text=np.array(pool_output)[::-1].astype(int).astype(str), texttemplate="%{text}"))
            fig_po.update_layout(**DL, title="After 2×2 Max Pool (2×2)", height=300)
            st.plotly_chart(fig_po, use_container_width=True)

        st.markdown("""<div class="insight-box">
        4×4 → 2×2 = <b>4× size reduction</b>. Only the strongest activations survive.
        No learnable parameters — just takes the max.
        </div>""", unsafe_allow_html=True)

    elif cnn_view == "Full Architecture":
        st.markdown("""<div class="concept-card">
        A CNN stacks Conv → ReLU → Pool blocks. Spatial dimensions shrink while depth (channels) grows.
        </div>""", unsafe_allow_html=True)

        layers_data = [
            ("Input", "224×224×3", 224*224*3, "#f5b731"),
            ("Conv1 + ReLU", "112×112×32", 112*112*32, "#5eaeff"),
            ("MaxPool", "56×56×32", 56*56*32, "#7c6aff"),
            ("Conv2 + ReLU", "56×56×64", 56*56*64, "#5eaeff"),
            ("MaxPool", "28×28×64", 28*28*64, "#7c6aff"),
            ("Conv3 + ReLU", "28×28×128", 28*28*128, "#5eaeff"),
            ("MaxPool", "14×14×128", 14*14*128, "#7c6aff"),
            ("Flatten", "25,088", 25088, "#f5b731"),
            ("Dense 256", "256", 256, "#22d3a7"),
            ("Output", "1", 1, "#22d3a7"),
        ]

        names = [l[0] for l in layers_data]
        sizes = [l[1] for l in layers_data]
        values = [np.log10(l[2] + 1) for l in layers_data]
        colors = [l[3] for l in layers_data]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=values, marker_color=colors,
                             text=sizes, textposition='outside', textfont=dict(color='#e2e8f0', size=10)))
        fig.update_layout(**DL, title="CNN Architecture: Spatial Dimensions Shrink, Depth Grows",
                          yaxis_title="log₁₀(total values)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Transfer Learning
        st.markdown("""<div class="concept-card">
        Instead of training from scratch (millions of images, days of GPU), take a <b>pretrained</b> model
        (e.g., ResNet trained on ImageNet), freeze the early layers, and only retrain the last layer for your task.
        </div>""", unsafe_allow_html=True)

        steps = ["1. Load Pretrained\nResNet50", "2. Freeze Layers\n1-49", "3. Replace Last\nLayer", "4. Train New\nLayer Only"]
        effort = [100, 0, 1, 5]
        colors = ["#f5b731", "#5eaeff", "#7c6aff", "#22d3a7"]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=steps, y=effort, marker_color=colors,
                             text=["14M images trained", "Frozen (universal)", "New head", "100-1000 images"],
                             textposition='outside', textfont=dict(color='#e2e8f0')))
        fig.update_layout(**DL, title="Transfer Learning: Effort at Each Step", yaxis_title="Relative effort", height=350)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════
# TAB: RNNs & LSTMs
# ═══════════════════════════════════════
elif tab == "🔁 RNNs & LSTMs":
    st.markdown("# 🔁 RNNs & LSTMs")
    st.caption("Processing sequences with memory")

    rnn_view = st.radio("View:", ["RNN Basics", "LSTM Gates", "Not Good Demo", "RNN vs LSTM vs Transformer"], horizontal=True)

    if rnn_view == "RNN Basics":
        st.markdown("""<div class="concept-card">
        An RNN processes one word at a time, carrying a <b>hidden state</b> (memory) from step to step.
        The same weights are used at every step — so it works for any sequence length.
        </div>""", unsafe_allow_html=True)

        sentence = st.text_input("Enter a sentence:", "Great pizza fast delivery")
        words = sentence.strip().split()

        if words:
            np.random.seed(42)
            hidden = np.zeros(3)
            W_xh = np.random.randn(3) * 0.5
            W_hh = np.random.randn(3, 3) * 0.3

            states = [hidden.copy()]
            for w in words:
                x = np.array([ord(c) for c in w[:3].ljust(3)]) / 122.0
                hidden = np.tanh(W_hh @ hidden + W_xh * np.mean(x))
                states.append(hidden.copy())

            fig = go.Figure()
            for dim in range(3):
                fig.add_trace(go.Scatter(
                    x=["start"] + words, y=[s[dim] for s in states],
                    mode='lines+markers', name=f'Hidden dim {dim+1}',
                    line=dict(width=2)
                ))
            fig.update_layout(**DL, title="Hidden State Evolution", xaxis_title="Word", yaxis_title="Hidden value", height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""<div class="insight-box">
            Each word updates the hidden state. The final hidden state summarizes the entire sentence.
            Notice how each word shifts the hidden dimensions — the network is building a representation.
            </div>""", unsafe_allow_html=True)

    elif rnn_view == "LSTM Gates":
        st.markdown("""<div class="concept-card">
        LSTM has 3 gates controlling a <b>cell state</b> (memory highway):<br>
        🚪 <b>Forget gate</b>: what to erase (0=forget, 1=keep)<br>
        🚪 <b>Input gate</b>: what new info to write<br>
        🚪 <b>Output gate</b>: what to read out
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Interactive Gate Values")
        col1, col2, col3 = st.columns(3)
        with col1:
            forget = st.slider("Forget gate (f)", 0.0, 1.0, 0.9, 0.05)
        with col2:
            input_g = st.slider("Input gate (i)", 0.0, 1.0, 0.7, 0.05)
        with col3:
            output_g = st.slider("Output gate (o)", 0.0, 1.0, 0.8, 0.05)

        old_cell = st.slider("Old cell state", -2.0, 2.0, 0.5, 0.1)
        candidate = st.slider("New candidate value", -2.0, 2.0, 0.8, 0.1)

        new_cell = forget * old_cell + input_g * candidate
        hidden_out = output_g * np.tanh(new_cell)

        st.markdown(f"""<div class="math-box">
        <b>Cell state update:</b><br>
        c_new = f × c_old + i × candidate<br>
        = {forget:.2f} × {old_cell:.2f} + {input_g:.2f} × {candidate:.2f}<br>
        = {forget * old_cell:.3f} + {input_g * candidate:.3f} = <b>{new_cell:.3f}</b><br><br>
        <b>Hidden output:</b><br>
        h = o × tanh(c_new) = {output_g:.2f} × tanh({new_cell:.3f}) = {output_g:.2f} × {np.tanh(new_cell):.3f} = <b>{hidden_out:.3f}</b>
        </div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Old Cell', 'Forget×Old', 'Input×New', 'New Cell', 'Output'],
                             y=[old_cell, forget*old_cell, input_g*candidate, new_cell, hidden_out],
                             marker_color=['#5eaeff', '#f45d6d', '#22d3a7', '#7c6aff', '#f5b731']))
        fig.update_layout(**DL, title="LSTM Cell State Flow", height=300)
        st.plotly_chart(fig, use_container_width=True)

    elif rnn_view == "Not Good Demo":
        st.markdown("""<div class="concept-card">
        The classic test: can the model understand that <b>"not good"</b> is negative?
        A simple RNN overwrites "not" when it sees "good". An LSTM preserves the negation.
        </div>""", unsafe_allow_html=True)

        words_demo = ["The", "pizza", "was", "not", "good"]
        rnn_sentiment = [0.0, 0.1, 0.1, -0.3, 0.5]
        lstm_sentiment = [0.0, 0.1, 0.1, -0.5, -0.3]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=words_demo, y=rnn_sentiment, mode='lines+markers',
                                  name='Simple RNN', line=dict(color='#f45d6d', width=3)))
        fig.add_trace(go.Scatter(x=words_demo, y=lstm_sentiment, mode='lines+markers',
                                  name='LSTM', line=dict(color='#22d3a7', width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="#2d3148")
        fig.update_layout(**DL, title="Sentiment Through Sequence", xaxis_title="Word",
                          yaxis_title="Sentiment (neg=bad, pos=good)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.error("**RNN**: 'good' overwrites 'not' → Positive ❌")
        with col2:
            st.success("**LSTM**: forget gate preserves 'not' → Negative ✅")

    else:  # Comparison
        st.markdown("### Evolution: RNN → LSTM → Transformer")
        models = ['RNN (1986)', 'LSTM (1997)', 'Transformer (2017)']
        memory = [20, 200, 10000]
        speed = [1, 0.8, 10]
        parallel = [0, 0, 1]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Memory (steps)', x=models, y=memory, marker_color='#5eaeff'))
        fig.add_trace(go.Bar(name='Relative Speed', x=models, y=[s*20 for s in speed], marker_color='#22d3a7'))
        fig.update_layout(**DL, barmode='group', title="RNN vs LSTM vs Transformer", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <b>RNN</b>: forgets after ~20 steps. <b>LSTM</b>: remembers 200+ steps via cell state.
        <b>Transformer</b>: sees entire sequence at once via attention — no forgetting, fully parallel.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TAB: Transformers
# ═══════════════════════════════════════
elif tab == "🤖 Transformers":
    st.markdown("# 🤖 Transformers & Attention")
    st.caption("The architecture behind GPT, BERT, and all modern LLMs")

    tf_view = st.radio("View:", ["Self-Attention Demo", "Attention Heatmap", "BERT vs GPT", "Scaling Laws"], horizontal=True)

    if tf_view == "Self-Attention Demo":
        st.markdown("""<div class="concept-card">
        For each word, self-attention asks: <b>"Which other words should I pay attention to?"</b>
        Every word creates a Query (what am I looking for?), Key (what do I contain?), and Value (what info do I provide).
        </div>""", unsafe_allow_html=True)

        sentence = st.text_input("Enter a sentence:", "The pizza was great", key="tf_sent")
        words = sentence.strip().split()

        if len(words) >= 2:
            np.random.seed(42)
            n = len(words)
            embeddings = np.random.randn(n, 4) * 0.5
            Q = embeddings @ np.random.randn(4, 4) * 0.3
            K = embeddings @ np.random.randn(4, 4) * 0.3

            scores = Q @ K.T / np.sqrt(4)
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            attention = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            fig = go.Figure(data=go.Heatmap(
                z=attention[::-1], x=words, y=words[::-1],
                colorscale='Viridis', showscale=True,
                text=np.round(attention[::-1], 3).astype(str), texttemplate="%{text}"
            ))
            fig.update_layout(**DL, title="Attention Weights (each row = one word attending to all others)",
                              height=400, width=500)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""<div class="math-box">
            <b>Formula:</b> Attention(Q,K,V) = softmax(Q × K^T / √d_k) × V<br><br>
            Each cell shows how much word (row) attends to word (column).
            Brighter = more attention. Each row sums to 1.0.
            </div>""", unsafe_allow_html=True)

    elif tf_view == "Attention Heatmap":
        st.markdown("""<div class="concept-card">
        In a trained model, attention patterns reveal what the model learned.
        "great" attends to "pizza" (what was great?), "slow" attends to "delivery" (what was slow?).
        </div>""", unsafe_allow_html=True)

        words_ex = ["The", "pizza", "was", "great", "but", "delivery", "was", "slow"]
        attn = np.array([
            [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.4, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1],
            [0.1, 0.2, 0.3, 0.2, 0.0, 0.1, 0.0, 0.1],
            [0.05, 0.5, 0.1, 0.2, 0.0, 0.05, 0.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1],
            [0.05, 0.1, 0.05, 0.05, 0.1, 0.4, 0.1, 0.15],
            [0.05, 0.1, 0.1, 0.05, 0.1, 0.2, 0.3, 0.1],
            [0.05, 0.05, 0.05, 0.05, 0.1, 0.4, 0.1, 0.2],
        ])

        fig = go.Figure(data=go.Heatmap(
            z=attn[::-1], x=words_ex, y=words_ex[::-1],
            colorscale='Viridis', showscale=True,
            text=np.round(attn[::-1], 2).astype(str), texttemplate="%{text}"
        ))
        fig.update_layout(**DL, title="Trained Attention: 'great' → 'pizza', 'slow' → 'delivery'", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        Row "great" has highest attention on "pizza" (0.50) — it learned what was great.<br>
        Row "slow" has highest attention on "delivery" (0.40) — it learned what was slow.<br>
        This is how Transformers understand context without processing sequentially.
        </div>""", unsafe_allow_html=True)

    elif tf_view == "BERT vs GPT":
        st.markdown("""<div class="concept-card">
        Both use Transformers but differently. <b>BERT</b> sees all words (bidirectional) — good for understanding.
        <b>GPT</b> sees only left context (autoregressive) — good for generation.
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""<div class="math-box">
            <b>BERT (Encoder)</b><br>
            Sees: ← ALL words →<br>
            Training: fill in [MASK]<br>
            "The pizza was [?] but delivery was slow"<br>
            Uses "slow" to predict [?] = "great"<br><br>
            Best for: Classification, NER, Q&A, Embeddings
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="insight-box">
            <b>GPT (Decoder)</b><br>
            Sees: ← only LEFT context<br>
            Training: predict next word<br>
            "The pizza was ___"<br>
            Only sees left words, predicts "great"<br><br>
            Best for: Chatbots, Code gen, Creative writing
            </div>""", unsafe_allow_html=True)

        features = ['Bidirectional', 'Generation', 'Understanding', 'Speed', 'Parameters']
        bert_scores = [10, 3, 10, 7, 5]
        gpt_scores = [3, 10, 7, 6, 10]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=bert_scores, theta=features, fill='toself', name='BERT', line=dict(color='#5eaeff')))
        fig.add_trace(go.Scatterpolar(r=gpt_scores, theta=features, fill='toself', name='GPT', line=dict(color='#7c6aff')))
        fig.update_layout(polar=dict(bgcolor='#0e1117', radialaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0')),
                                      angularaxis=dict(gridcolor='#2d3148', tickfont=dict(color='#8892b0'))),
                          paper_bgcolor='#0e1117', font=dict(color='#e2e8f0'), title="BERT vs GPT Strengths", height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Scaling Laws
        st.markdown("### How LLMs Scale")
        models = ['BERT-base', 'GPT-2', 'GPT-3', 'GPT-4 (est.)']
        params = [110e6, 1.5e9, 175e9, 1.8e12]
        layers = [12, 48, 96, 120]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=models, y=[np.log10(p) for p in params], marker_color=['#f5b731', '#5eaeff', '#7c6aff', '#22d3a7'],
                             text=[f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M" for p in params],
                             textposition='outside', textfont=dict(color='#e2e8f0')))
        fig.update_layout(**DL, title="Model Size (log scale)", yaxis_title="log₁₀(parameters)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        Each generation is ~100× larger. More parameters + more data + more compute = better performance.
        This is the <b>scaling law</b> — performance improves predictably with scale.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TAB: NLP Basics
# ═══════════════════════════════════════
elif tab == "📝 NLP Basics":
    st.markdown("# 📝 NLP Text Processing")
    st.caption("Converting text to numbers for ML")

    nlp_view = st.radio("View:", ["Preprocessing Pipeline", "TF-IDF Calculator", "Word Embeddings", "BoW to Transformers"], horizontal=True)

    if nlp_view == "Preprocessing Pipeline":
        st.markdown("""<div class="concept-card">
        Raw text is messy. The preprocessing pipeline cleans it step by step:
        lowercase → remove punctuation → tokenize → remove stop words → stem/lemmatize.
        </div>""", unsafe_allow_html=True)

        text_input = st.text_input("Enter text:", "Great Pizza, Fast Delivery! The BEST in town.")

        if text_input:
            import re
            steps = []
            current = text_input
            steps.append(("Original", current))

            current = current.lower()
            steps.append(("1. Lowercase", current))

            current = re.sub(r'[^\w\s]', '', current)
            steps.append(("2. Remove Punctuation", current))

            tokens = current.split()
            steps.append(("3. Tokenize", str(tokens)))

            stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'is', 'was', 'are', 'were', 'it', 'its', 'this', 'that'}
            filtered = [t for t in tokens if t not in stop_words]
            steps.append(("4. Remove Stop Words", str(filtered)))

            for step_name, result in steps:
                if "Original" in step_name:
                    st.markdown(f"""<div class="math-box"><b>{step_name}:</b> {result}</div>""", unsafe_allow_html=True)
                elif "Stop" in step_name or "Token" in step_name:
                    st.markdown(f"""<div class="insight-box"><b>{step_name}:</b> {result}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="concept-card"><b>{step_name}:</b> {result}</div>""", unsafe_allow_html=True)

            st.markdown(f"**Final tokens:** {len(filtered)} words → ready for vectorization")

    elif nlp_view == "TF-IDF Calculator":
        st.markdown("""<div class="concept-card">
        <b>TF-IDF</b> = Term Frequency × Inverse Document Frequency.
        Common words (like "pizza") get low scores. Rare distinctive words (like "amazing") get high scores.
        </div>""", unsafe_allow_html=True)

        docs = [
            "great pizza fast delivery",
            "terrible cold pizza slow",
            "best pizza in town loved it",
            "awful service never ordering again",
        ]
        st.markdown("**Documents:**")
        for i, d in enumerate(docs):
            st.markdown(f"- Doc {i+1}: *{d}*")

        all_words = set()
        for d in docs:
            all_words.update(d.split())
        all_words = sorted(all_words)

        word = st.selectbox("Select a word to compute TF-IDF:", all_words)

        if word:
            doc_idx = st.selectbox("In which document?", [f"Doc {i+1}: {d}" for i, d in enumerate(docs)])
            idx = int(doc_idx.split(":")[0].split(" ")[1]) - 1
            doc_words = docs[idx].split()

            tf = doc_words.count(word) / len(doc_words)
            docs_with_word = sum(1 for d in docs if word in d.split())
            idf = np.log(len(docs) / max(docs_with_word, 1))
            tfidf = tf * idf

            col1, col2, col3 = st.columns(3)
            col1.metric("TF", f"{tf:.3f}", help=f"{doc_words.count(word)}/{len(doc_words)} words")
            col2.metric("IDF", f"{idf:.3f}", help=f"log({len(docs)}/{docs_with_word})")
            col3.metric("TF-IDF", f"{tfidf:.3f}")

            st.markdown(f"""<div class="math-box">
            TF = {doc_words.count(word)}/{len(doc_words)} = {tf:.3f} (frequency in this doc)<br>
            IDF = log({len(docs)}/{docs_with_word}) = {idf:.3f} ({'rare' if docs_with_word <= 1 else 'common' if docs_with_word >= 3 else 'moderate'} across docs)<br>
            <b>TF-IDF = {tf:.3f} × {idf:.3f} = {tfidf:.3f}</b>
            </div>""", unsafe_allow_html=True)

    elif nlp_view == "Word Embeddings":
        st.markdown("""<div class="concept-card">
        Word embeddings represent words as dense vectors where <b>similar words are close together</b>.
        "great" and "amazing" have similar vectors. "great" and "terrible" are far apart.
        </div>""", unsafe_allow_html=True)

        np.random.seed(42)
        word_vecs = {
            'great': [0.8, 0.9], 'amazing': [0.7, 0.85], 'good': [0.6, 0.7],
            'terrible': [-0.7, -0.8], 'awful': [-0.6, -0.75], 'bad': [-0.5, -0.6],
            'pizza': [0.1, -0.3], 'delivery': [-0.2, 0.4], 'fast': [0.3, 0.5],
        }

        fig = go.Figure()
        colors = {'great': '#22d3a7', 'amazing': '#22d3a7', 'good': '#22d3a7',
                  'terrible': '#f45d6d', 'awful': '#f45d6d', 'bad': '#f45d6d',
                  'pizza': '#f5b731', 'delivery': '#5eaeff', 'fast': '#5eaeff'}

        for word, vec in word_vecs.items():
            fig.add_trace(go.Scatter(x=[vec[0]], y=[vec[1]], mode='markers+text',
                                     text=[word], textposition='top center',
                                     marker=dict(size=12, color=colors[word]),
                                     textfont=dict(color=colors[word], size=12),
                                     showlegend=False))

        fig.update_layout(**DL, title="Word Embeddings (2D projection)",
                          xaxis_title="Dimension 1", yaxis_title="Dimension 2", height=450)
        fig.add_hline(y=0, line_dash="dash", line_color="#2d3148")
        fig.add_vline(x=0, line_dash="dash", line_color="#2d3148")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        🟢 Positive words cluster together (top-right)<br>
        🔴 Negative words cluster together (bottom-left)<br>
        🟡🔵 Neutral words are in different regions<br><br>
        This structure is learned automatically from reading millions of sentences.
        </div>""", unsafe_allow_html=True)

    else:  # Evolution
        st.markdown("### The Evolution of Text Representation")
        methods = ['Bag of Words\n(1990s)', 'TF-IDF\n(2000s)', 'Word2Vec\n(2013)', 'Transformers\n(2017)', 'LLMs\n(2020s)']
        capability = [1, 2, 4, 8, 10]
        colors = ['#f5b731', '#5eaeff', '#7c6aff', '#22d3a7', '#22d3a7']

        fig = go.Figure()
        fig.add_trace(go.Bar(x=methods, y=capability, marker_color=colors,
                             text=['No word meaning', 'Weights by rarity', 'Semantic vectors', 'Context-aware', 'Generate text'],
                             textposition='outside', textfont=dict(color='#e2e8f0', size=10)))
        fig.update_layout(**DL, title="NLP Evolution: Each Generation Solved a Limitation", yaxis_title="Capability", height=400)
        st.plotly_chart(fig, use_container_width=True)
