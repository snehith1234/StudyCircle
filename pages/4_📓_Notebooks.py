import streamlit as st
import json
import os

st.set_page_config(page_title="📓 Notebooks", page_icon="📓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.nb-code {
    background: #1a1d2e; border: 1px solid #2d3148; border-radius: 10px;
    padding: 1rem; margin: 0.5rem 0; font-family: 'Fira Code', monospace;
    font-size: 0.85rem; overflow-x: auto;
}
.nb-md {
    background: linear-gradient(135deg, #1a1d2e, #1f2235);
    border: 1px solid #2d3148; border-radius: 12px; padding: 1.2rem 1.4rem;
    margin: 0.6rem 0; line-height: 1.8; font-size: 0.93rem; color: #c8cfe0;
}
.nb-md b, .nb-md strong { color: #e2e8f0; }
.nb-md code { background: #252840; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }
.cell-num {
    display: inline-block; background: #252840; color: #7c6aff; font-size: 0.7rem;
    padding: 2px 8px; border-radius: 6px; margin-bottom: 4px; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

NOTEBOOKS_DIR = "notebooks"

NOTEBOOK_INFO = {
    "pizza_store_descriptive_stats": ("🍕 M1: Descriptive Stats — Pizza Store", "Summarize 50 pizza stores with mean, median, std dev, outliers, quartiles"),
    "M2_probability": ("🎲 M2: Probability — Real-Life Cases", "Late deliveries, churn prediction, fraud detection with Bayes' theorem"),
    "M3_distributions": ("📈 M3: Distributions — Real-Life Cases", "Normal, skewed, Poisson, binomial — which shape fits which data?"),
    "M4_inferential_statistics": ("🧪 M4: Z-Scores, T-Tests & P-Values", "One story through all three tools with step-by-step math"),
    "M5_correlation": ("🔗 M5: Correlation — Real-Life Cases", "Ad spend vs sales, employees vs delivery time, the causation trap"),
    "M6_regression": ("📉 M6: Regression — Real-Life Cases", "Predicting sales from ad spend, residuals, overfitting"),
}


def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def render_notebook(nb_data):
    cells = nb_data.get("cells", [])
    code_count = 0
    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if not source.strip():
            continue

        if cell_type == "markdown":
            st.markdown(source, unsafe_allow_html=True)

        elif cell_type == "code":
            code_count += 1
            st.markdown(f'<span class="cell-num">In [{code_count}]</span>', unsafe_allow_html=True)
            st.code(source, language="python")


# ── Sidebar ──
with st.sidebar:
    st.markdown("## 📓 Notebooks")
    st.caption("Interactive notebooks for each module — view here or download to run locally.")
    st.divider()

    nb_files = []
    if os.path.exists(NOTEBOOKS_DIR):
        for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
            if fname.endswith(".ipynb"):
                nb_files.append(fname)

    if not nb_files:
        st.warning("No notebooks found in /notebooks folder.")
        st.stop()

    display_names = []
    for f in nb_files:
        key = f.replace(".ipynb", "")
        if key in NOTEBOOK_INFO:
            display_names.append(NOTEBOOK_INFO[key][0])
        else:
            display_names.append(f.replace(".ipynb", "").replace("_", " "))

    selected_idx = st.radio("Pick a notebook:", range(len(nb_files)),
                            format_func=lambda i: display_names[i],
                            label_visibility="collapsed")

# ── Main content ──
selected_file = nb_files[selected_idx]
selected_key = selected_file.replace(".ipynb", "")
nb_path = os.path.join(NOTEBOOKS_DIR, selected_file)

title, desc = NOTEBOOK_INFO.get(selected_key, (selected_key, ""))
st.markdown(f"# {title}")
st.caption(desc)

# Download button
with open(nb_path, 'r') as f:
    nb_content = f.read()

col1, col2 = st.columns([3, 1])
with col2:
    st.download_button(
        "⬇️ Download .ipynb",
        data=nb_content,
        file_name=selected_file,
        mime="application/json",
        use_container_width=True,
    )
with col1:
    st.info("💡 This is a read-only view. Download the notebook and open it in Jupyter/Colab to run the code and see outputs.")

st.divider()

# Render
nb_data = load_notebook(nb_path)
render_notebook(nb_data)
