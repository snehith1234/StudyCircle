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
    st.markdown("# 🔄 Backpropagation — Step by Step")
    st.caption("Watch gradients flow forward then backward through actual numbers")

    st.markdown("""<div class="concept-card">
    We'll trace <b>Store S1 (Rating=4.5, Delivery=20, Actual=1)</b> through a simple network.
    First the <b>forward pass</b> computes the prediction. Then the <b>backward pass</b> computes
    how to fix each weight. Every number is computed — nothing is hidden.
    </div>""", unsafe_allow_html=True)

    # Network setup
    x1, x2, y_true = 4.5, 20.0, 1.0
    w1, w2, b = 1.0, -0.05, -2.0
    eta = st.slider("Learning rate (η)", 0.01, 1.0, 0.1, 0.01)

    bp_step = st.radio("Step:", [
        "1. Forward: Weighted Sum",
        "2. Forward: Activation",
        "3. Forward: Loss",
        "4. Backward: Output Gradient",
        "5. Backward: Sigmoid Gradient",
        "6. Backward: Weight Gradients",
        "7. Update Weights",
        "8. Full Picture"
    ], horizontal=False)

    # Compute everything
    z = w1 * x1 + w2 * x2 + b
    a = 1 / (1 + np.exp(-z))
    loss = -(y_true * np.log(a) + (1 - y_true) * np.log(1 - a))

    # Backward
    dL_da = -y_true / a + (1 - y_true) / (1 - a)  # ∂Loss/∂a
    da_dz = a * (1 - a)                              # ∂a/∂z (sigmoid derivative)
    dL_dz = dL_da * da_dz                             # ∂Loss/∂z = (a - y)
    dz_dw1 = x1                                       # ∂z/∂w1
    dz_dw2 = x2                                       # ∂z/∂w2
    dz_db = 1.0                                       # ∂z/∂b
    dL_dw1 = dL_dz * dz_dw1                           # ∂Loss/∂w1
    dL_dw2 = dL_dz * dz_dw2                           # ∂Loss/∂w2
    dL_db = dL_dz * dz_db                             # ∂Loss/∂b

    w1_new = w1 - eta * dL_dw1
    w2_new = w2 - eta * dL_dw2
    b_new = b - eta * dL_db

    # New prediction after update
    z_new = w1_new * x1 + w2_new * x2 + b_new
    a_new = 1 / (1 + np.exp(-z_new))
    loss_new = -(y_true * np.log(a_new) + (1 - y_true) * np.log(1 - a_new))

    if bp_step == "1. Forward: Weighted Sum":
        st.markdown("### ➡️ Step 1: Compute Weighted Sum (z)")
        st.markdown(f"""<div class="math-box">
        <b>z = w₁ × x₁ + w₂ × x₂ + b</b><br><br>
        z = {w1:.1f} × {x1:.1f} + ({w2:.2f}) × {x2:.0f} + ({b:.1f})<br>
        z = {w1*x1:.2f} + ({w2*x2:.2f}) + ({b:.1f})<br>
        <b>z = {z:.4f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-box">
        The weighted sum z combines all inputs. Positive z → sigmoid will output > 0.5.
        Negative z → sigmoid will output < 0.5. z = 0 → exactly 0.5 (coin flip).
        </div>""", unsafe_allow_html=True)

    elif bp_step == "2. Forward: Activation":
        st.markdown("### ➡️ Step 2: Apply Sigmoid Activation")
        st.markdown(f"""<div class="math-box">
        <b>ŷ = σ(z) = 1 / (1 + e⁻ᶻ)</b><br><br>
        ŷ = 1 / (1 + e^(-{z:.4f}))<br>
        ŷ = 1 / (1 + {np.exp(-z):.4f})<br>
        ŷ = 1 / {1 + np.exp(-z):.4f}<br>
        <b>ŷ = {a:.6f} = {a*100:.2f}%</b>
        </div>""", unsafe_allow_html=True)
        if a >= 0.5:
            st.success(f"Prediction: ✅ Successful ({a*100:.2f}%) — Actual: 1 (Successful)")
        else:
            st.error(f"Prediction: ❌ Not Successful ({a*100:.2f}%) — Actual: 1 (Successful)")

    elif bp_step == "3. Forward: Loss":
        st.markdown("### ➡️ Step 3: Compute Loss (How Wrong?)")
        st.markdown(f"""<div class="math-box">
        <b>Loss = −[y × log(ŷ) + (1−y) × log(1−ŷ)]</b><br><br>
        Since y = 1 (actual is Successful):<br>
        Loss = −[1 × log({a:.6f}) + 0 × log({1-a:.6f})]<br>
        Loss = −log({a:.6f})<br>
        <b>Loss = {loss:.6f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-box">
        A perfect prediction (ŷ=1.0) would give Loss = 0.<br>
        Our prediction ŷ={a:.4f} gives Loss = {loss:.4f}.<br>
        The model needs to push ŷ closer to 1.0 — that's what backprop will do.
        </div>""", unsafe_allow_html=True)

    elif bp_step == "4. Backward: Output Gradient":
        st.markdown("### ⬅️ Step 4: Gradient of Loss w.r.t. Prediction (∂Loss/∂ŷ)")
        st.markdown(f"""<div class="math-box">
        <b>∂Loss/∂ŷ = −y/ŷ + (1−y)/(1−ŷ)</b><br><br>
        Since y = 1:<br>
        ∂Loss/∂ŷ = −1/{a:.6f} + 0/(1−{a:.6f})<br>
        <b>∂Loss/∂ŷ = {dL_da:.6f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-box">
        This is <b>negative</b> ({dL_da:.4f}) because the actual is 1 but prediction is {a:.4f} (too low).
        A negative gradient means: "increase the prediction to reduce the loss."
        </div>""", unsafe_allow_html=True)

    elif bp_step == "5. Backward: Sigmoid Gradient":
        st.markdown("### ⬅️ Step 5: Gradient Through Sigmoid (∂ŷ/∂z)")
        st.markdown(f"""<div class="math-box">
        <b>∂ŷ/∂z = σ(z) × (1 − σ(z)) = ŷ × (1 − ŷ)</b><br><br>
        ∂ŷ/∂z = {a:.6f} × (1 − {a:.6f})<br>
        ∂ŷ/∂z = {a:.6f} × {1-a:.6f}<br>
        <b>∂ŷ/∂z = {da_dz:.6f}</b><br><br>
        <b>Combined: ∂Loss/∂z = ∂Loss/∂ŷ × ∂ŷ/∂z</b><br>
        ∂Loss/∂z = {dL_da:.6f} × {da_dz:.6f}<br>
        <b>∂Loss/∂z = {dL_dz:.6f}</b><br><br>
        <i>Shortcut: ∂Loss/∂z = ŷ − y = {a:.6f} − {y_true:.0f} = {a - y_true:.6f} ✓</i>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-box">
        The sigmoid gradient is {da_dz:.4f} — notice it's less than 0.25 (the maximum).
        This is why sigmoid causes vanishing gradients: each layer multiplies by a number ≤ 0.25.
        <br><br>The shortcut (ŷ − y) works because log loss and sigmoid are mathematically paired.
        </div>""", unsafe_allow_html=True)

    elif bp_step == "6. Backward: Weight Gradients":
        st.markdown("### ⬅️ Step 6: Gradient for Each Weight")
        st.markdown(f"""<div class="math-box">
        <b>∂Loss/∂w₁ = ∂Loss/∂z × ∂z/∂w₁ = (ŷ − y) × x₁</b><br>
        ∂Loss/∂w₁ = {dL_dz:.6f} × {x1:.1f} = <b>{dL_dw1:.6f}</b><br><br>
        <b>∂Loss/∂w₂ = ∂Loss/∂z × ∂z/∂w₂ = (ŷ − y) × x₂</b><br>
        ∂Loss/∂w₂ = {dL_dz:.6f} × {x2:.0f} = <b>{dL_dw2:.6f}</b><br><br>
        <b>∂Loss/∂b = ∂Loss/∂z × ∂z/∂b = (ŷ − y) × 1</b><br>
        ∂Loss/∂b = {dL_dz:.6f} × 1 = <b>{dL_db:.6f}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="insight-box">
        All gradients are <b>negative</b> because the prediction ({a:.4f}) is below the target (1.0).<br>
        Negative gradient → weights should <b>increase</b> (move opposite to gradient).<br><br>
        w₂'s gradient ({dL_dw2:.4f}) is larger in magnitude than w₁'s ({dL_dw1:.4f})
        because x₂ (Delivery={x2:.0f}) is numerically larger than x₁ (Rating={x1:.1f}).
        </div>""", unsafe_allow_html=True)

    elif bp_step == "7. Update Weights":
        st.markdown("### 🔧 Step 7: Update Weights")
        st.markdown(f"""<div class="math-box">
        <b>Update rule: w_new = w_old − η × gradient</b><br><br>
        w₁: {w1:.4f} − {eta:.3f} × ({dL_dw1:.6f}) = {w1:.4f} − ({eta * dL_dw1:.6f}) = <b>{w1_new:.6f}</b><br>
        w₂: {w2:.4f} − {eta:.3f} × ({dL_dw2:.6f}) = {w2:.4f} − ({eta * dL_dw2:.6f}) = <b>{w2_new:.6f}</b><br>
        b:  {b:.4f} − {eta:.3f} × ({dL_db:.6f}) = {b:.4f} − ({eta * dL_db:.6f}) = <b>{b_new:.6f}</b>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("w₁ (Rating)", f"{w1_new:.4f}", f"{w1_new - w1:+.4f}")
        col2.metric("w₂ (Delivery)", f"{w2_new:.4f}", f"{w2_new - w2:+.4f}")
        col3.metric("b (Bias)", f"{b_new:.4f}", f"{b_new - b:+.4f}")

        st.markdown(f"""<div class="insight-box">
        All weights <b>increased</b> (moved opposite to negative gradient) to push the prediction higher.<br>
        w₁ increased by {w1_new - w1:+.4f} → higher rating will now contribute more to success.<br>
        w₂ increased by {w2_new - w2:+.4f} → delivery penalty is now slightly less harsh.
        </div>""", unsafe_allow_html=True)

    else:  # Full Picture
        st.markdown("### 📊 Full Picture: Before vs After One Update")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="math-box">
            <b>BEFORE (Forward Pass)</b><br>
            Weights: w₁={w1:.2f}, w₂={w2:.2f}, b={b:.1f}<br>
            z = {z:.4f}<br>
            ŷ = σ(z) = {a:.4f} ({a*100:.1f}%)<br>
            Loss = {loss:.4f}
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="insight-box">
            <b>AFTER (One Gradient Step)</b><br>
            Weights: w₁={w1_new:.4f}, w₂={w2_new:.4f}, b={b_new:.4f}<br>
            z = {z_new:.4f}<br>
            ŷ = σ(z) = {a_new:.4f} ({a_new*100:.1f}%)<br>
            Loss = {loss_new:.4f}
            </div>""", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prediction", f"{a_new*100:.1f}%", f"{(a_new - a)*100:+.1f}%")
        col_b.metric("Loss", f"{loss_new:.4f}", f"{loss_new - loss:+.4f}")
        col_c.metric("Closer to target?", "Yes ✅" if a_new > a else "No ❌")

        st.markdown("### Gradient Flow Visualization")
        stages = ['∂Loss/∂ŷ', '× ∂ŷ/∂z', '= ∂Loss/∂z', '× x₁ = ∂L/∂w₁', '× x₂ = ∂L/∂w₂']
        values = [abs(dL_da), da_dz, abs(dL_dz), abs(dL_dw1), abs(dL_dw2)]
        colors = ['#f45d6d', '#f5b731', '#5eaeff', '#22d3a7', '#7c6aff']

        fig = go.Figure()
        fig.add_trace(go.Bar(x=stages, y=values, marker_color=colors,
                             text=[f"{v:.4f}" for v in values], textposition='outside',
                             textfont=dict(color='#e2e8f0')))
        fig.update_layout(**DL, title="Gradient Magnitude at Each Stage", yaxis_title="Gradient magnitude", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""<div class="concept-card">
        <b>The story in one sentence:</b> The prediction was {a:.4f} but should be 1.0.
        The error ({a - y_true:.4f}) flows backward through the sigmoid (×{da_dz:.4f}),
        then splits to each weight (×{x1:.1f} for w₁, ×{x2:.0f} for w₂).
        Each weight moves opposite to its gradient by η={eta:.3f}, pushing the prediction from {a*100:.1f}% to {a_new*100:.1f}%.
        Repeat this 1000 times and the network converges.
        </div>""", unsafe_allow_html=True)


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
