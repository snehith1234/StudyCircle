import streamlit as st
import json
import os
import sys
from io import StringIO
import traceback

st.set_page_config(page_title="📓 Notebooks", page_icon="📓", layout="wide")

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
.nb-output {
    background: #0d1117; border: 1px solid #2d3148; border-radius: 8px;
    padding: 0.8rem; margin: 0.3rem 0; font-family: 'Fira Code', monospace;
    font-size: 0.82rem; color: #c8cfe0; overflow-x: auto;
}
.nb-error {
    background: #2d1b1b; border: 1px solid #5a2a3a; border-radius: 8px;
    padding: 0.8rem; margin: 0.3rem 0; font-family: 'Fira Code', monospace;
    font-size: 0.82rem; color: #f45d6d; overflow-x: auto;
}
.cell-header {
    display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem;
}
.cell-num {
    display: inline-block; background: #252840; color: #7c6aff; font-size: 0.7rem;
    padding: 2px 8px; border-radius: 6px; font-weight: 600;
}
.run-status {
    display: inline-block; font-size: 0.7rem; padding: 2px 8px; border-radius: 6px; font-weight: 600;
}
.status-success { background: #22d3a722; color: #22d3a7; }
.status-error { background: #f45d6d22; color: #f45d6d; }
.status-pending { background: #f5b73122; color: #f5b731; }
</style>
""", unsafe_allow_html=True)

NOTEBOOKS_DIR = "notebooks"

# Phase notebooks
PHASE_NOTEBOOKS = {
    "Phase1_Statistics_Pizza": {
        "title": "📊 Phase 1: Statistics Fundamentals",
        "desc": "Complete statistics journey — descriptive stats, probability, distributions, hypothesis testing, correlation, and regression.",
        "topics": ["Descriptive Statistics", "Probability & Bayes", "Distributions", "Z-scores & T-tests", "Correlation", "Linear Regression"],
    },
    "Phase2_DS_Core_Pizza": {
        "title": "🔧 Phase 2: Data Science Core",
        "desc": "Data lifecycle, cleaning, feature engineering, and visualization.",
        "topics": ["Data Lifecycle", "Data Cleaning", "Feature Engineering", "Data Visualization"],
    },
    "Phase3_ML_Pizza": {
        "title": "🤖 Phase 3: Machine Learning",
        "desc": "Supervised learning, decision trees, random forests, XGBoost, and clustering.",
        "topics": ["Supervised Learning", "Decision Trees", "Random Forest", "XGBoost", "K-Means"],
    },
}

# Module notebooks
MODULE_NOTEBOOKS = {
    "pizza_store_descriptive_stats": ("🍕 M1: Descriptive Stats", "Mean, median, std dev, outliers"),
    "M2_probability": ("🎲 M2: Probability", "Bayes' theorem, conditional probability"),
    "M3_distributions": ("📈 M3: Distributions", "Normal, Poisson, binomial"),
    "M4_inferential_statistics": ("🧪 M4: Inferential Stats", "Z-scores, T-tests, P-values"),
    "M5_correlation": ("🔗 M5: Correlation", "Pearson, Spearman correlation"),
    "M6_regression": ("📉 M6: Regression", "Linear regression, R²"),
}


def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def execute_code(code, namespace):
    """Execute code and capture output."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    output = {"text": "", "figures": [], "error": None, "dataframes": []}
    
    try:
        # Execute the code
        exec(code, namespace)
        output["text"] = sys.stdout.getvalue()
        
        # Capture any matplotlib figures
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            output["figures"].append(fig)
        
        # Check for DataFrame in last expression
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # If last line is a variable name or expression, try to get its value
            if last_line and not last_line.startswith(('#', 'import', 'from', 'def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except', 'finally', 'return', 'yield', 'raise', 'assert', 'pass', 'break', 'continue')):
                if '=' not in last_line or last_line.count('=') == last_line.count('=='):
                    try:
                        result = eval(last_line, namespace)
                        if result is not None:
                            import pandas as pd
                            if isinstance(result, pd.DataFrame):
                                output["dataframes"].append(result)
                            elif isinstance(result, pd.Series):
                                output["dataframes"].append(result.to_frame())
                            elif not output["text"]:
                                output["text"] = repr(result)
                    except:
                        pass
                        
    except Exception as e:
        output["error"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    
    return output


def render_output(output):
    """Render execution output."""
    import matplotlib.pyplot as plt
    
    if output["error"]:
        st.markdown(f'<div class="nb-error">{output["error"]}</div>', unsafe_allow_html=True)
        return False
    
    # Show text output
    if output["text"].strip():
        st.code(output["text"], language="text")
    
    # Show DataFrames
    for df in output["dataframes"]:
        st.dataframe(df, use_container_width=True)
    
    # Show figures
    for fig in output["figures"]:
        st.pyplot(fig)
        plt.close(fig)
    
    return True


def render_saved_outputs(outputs):
    """Render pre-saved cell outputs from notebook file."""
    import base64
    for out in outputs:
        otype = out.get("output_type", "")

        if otype in ("stream", "execute_result", "display_data"):
            text = ""
            if "text" in out:
                text = "".join(out["text"])
            elif "data" in out and "text/plain" in out["data"]:
                text = "".join(out["data"]["text/plain"])

            if "data" in out and "text/html" in out["data"]:
                html = "".join(out["data"]["text/html"])
                st.markdown(html, unsafe_allow_html=True)
            elif text.strip():
                st.code(text, language="text")

            if "data" in out and "image/png" in out["data"]:
                img_data = out["data"]["image/png"]
                img_bytes = base64.b64decode(img_data)
                st.image(img_bytes)

        elif otype == "error":
            tb = "\n".join(out.get("traceback", []))
            st.error(f"```\n{tb}\n```")


# Initialize session state for execution
if 'nb_namespace' not in st.session_state:
    st.session_state.nb_namespace = {}
if 'nb_executed' not in st.session_state:
    st.session_state.nb_executed = {}
if 'nb_outputs' not in st.session_state:
    st.session_state.nb_outputs = {}
if 'current_notebook' not in st.session_state:
    st.session_state.current_notebook = None


# ── Sidebar ──
with st.sidebar:
    st.markdown("## 📓 Notebooks")
    st.caption("Run Jupyter notebooks directly in the browser")
    st.divider()
    
    category = st.radio("Category:", ["🎯 Phase Notebooks", "📚 Module Notebooks"], label_visibility="collapsed")
    
    st.divider()
    
    if category == "🎯 Phase Notebooks":
        st.markdown("### 🎯 Phase Notebooks")
        phase_keys = list(PHASE_NOTEBOOKS.keys())
        phase_names = [PHASE_NOTEBOOKS[k]["title"] for k in phase_keys]
        selected_idx = st.radio("Select:", range(len(phase_keys)),
                                format_func=lambda i: phase_names[i],
                                label_visibility="collapsed")
        selected_key = phase_keys[selected_idx]
        selected_file = f"{selected_key}.ipynb"
    else:
        st.markdown("### 📚 Module Notebooks")
        module_keys = list(MODULE_NOTEBOOKS.keys())
        module_names = [MODULE_NOTEBOOKS[k][0] for k in module_keys]
        selected_idx = st.radio("Select:", range(len(module_keys)),
                                format_func=lambda i: module_names[i],
                                label_visibility="collapsed")
        selected_key = module_keys[selected_idx]
        selected_file = f"{selected_key}.ipynb"
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Reset Notebook", use_container_width=True):
        st.session_state.nb_namespace = {}
        st.session_state.nb_executed = {}
        st.session_state.nb_outputs = {}
        st.rerun()


# Check if notebook changed
if st.session_state.current_notebook != selected_key:
    st.session_state.current_notebook = selected_key
    st.session_state.nb_namespace = {}
    st.session_state.nb_executed = {}
    st.session_state.nb_outputs = {}


# ── Main content ──
nb_path = os.path.join(NOTEBOOKS_DIR, selected_file)

if not os.path.exists(nb_path):
    st.error(f"Notebook not found: {selected_file}")
    st.stop()

nb_data = load_notebook(nb_path)
cells = nb_data.get("cells", [])

# Header
if selected_key in PHASE_NOTEBOOKS:
    info = PHASE_NOTEBOOKS[selected_key]
    st.markdown(f"# {info['title']}")
    st.caption(info['desc'])
    topics_str = " • ".join(info['topics'])
    st.markdown(f"**Topics:** {topics_str}")
else:
    title, desc = MODULE_NOTEBOOKS.get(selected_key, (selected_key, ""))
    st.markdown(f"# {title}")
    st.caption(desc)

# Controls
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.info("💡 Click **▶ Run** on each cell to execute, or **Run All** to execute the entire notebook.")
with col2:
    run_all = st.button("▶️ Run All Cells", use_container_width=True, type="primary")
with col3:
    with open(nb_path, 'r') as f:
        nb_content = f.read()
    st.download_button("⬇️ Download", data=nb_content, file_name=selected_file, 
                       mime="application/json", use_container_width=True)

st.divider()

# If Run All is clicked, execute all cells first before rendering
if run_all:
    with st.spinner("Running all cells..."):
        code_idx = 0
        for cell_idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))
            
            if not source.strip() or cell_type != "code":
                continue
            
            code_idx += 1
            cell_key = f"{selected_key}_{cell_idx}"
            
            # Execute if not already executed
            if not st.session_state.nb_executed.get(cell_key, False):
                output = execute_code(source, st.session_state.nb_namespace)
                st.session_state.nb_outputs[cell_key] = output
                st.session_state.nb_executed[cell_key] = True
    st.rerun()

# Render cells
code_count = 0
for cell_idx, cell in enumerate(cells):
    cell_type = cell.get("cell_type", "")
    source = "".join(cell.get("source", []))
    
    if not source.strip():
        continue
    
    if cell_type == "markdown":
        st.markdown(source, unsafe_allow_html=True)
    
    elif cell_type == "code":
        code_count += 1
        cell_key = f"{selected_key}_{cell_idx}"
        
        # Check execution status
        is_executed = st.session_state.nb_executed.get(cell_key, False)
        
        # Cell header with run button
        col_code, col_btn = st.columns([6, 1])
        
        with col_code:
            status = ""
            if is_executed:
                if st.session_state.nb_outputs.get(cell_key, {}).get("error"):
                    status = '<span class="run-status status-error">Error</span>'
                else:
                    status = '<span class="run-status status-success">✓ Done</span>'
            st.markdown(f'<span class="cell-num">In [{code_count}]</span> {status}', unsafe_allow_html=True)
        
        with col_btn:
            run_cell = st.button("▶ Run", key=f"run_{cell_key}", use_container_width=True)
        
        # Show code
        st.code(source, language="python")
        
        # Execute if individual run button clicked
        if run_cell:
            with st.spinner("Running..."):
                output = execute_code(source, st.session_state.nb_namespace)
                st.session_state.nb_outputs[cell_key] = output
                st.session_state.nb_executed[cell_key] = True
                st.rerun()
        
        # Show output
        if is_executed and cell_key in st.session_state.nb_outputs:
            output = st.session_state.nb_outputs[cell_key]
            render_output(output)
        elif cell.get("outputs"):
            # Show pre-saved outputs if available
            render_saved_outputs(cell.get("outputs", []))
        
        st.markdown("---")
