"""
Streamlit Dashboard for Results Visualization
"""

import os
import sys
import streamlit as st
import glob
import uuid
from streamlit_pdf_viewer import pdf_viewer
sys.path.append(os.path.abspath(os.path.join("src")))
from dataset import DOMAINS

st.title(":violet[Competition of Mechanisms]")
st.markdown("""
Visualization of Experiments
 - Logit Lens
 - Logit Attribution
 - Attention Pattern
""")

def to_snake_case(text):
    text = text.lower()
    text = text.split()
    text = "_".join(text)
    return text


DATASET = ["copyVSfact", "copyVSfactQnA", "copyVSfactDomain"]
EXPERIMENTS = {"Logit Lens": "residual_stream",
               "Logit Attribution": "logit_attribution",
               "Head Pattern": "heads_pattern"}
MODELS = ["gpt2", "pythia-6.9b", "Llama-3.2-1B", "Llama-3.1-8B"]
MODEL_FOLDER = "full"


## Sidebar Menu
with st.sidebar:
    st.title(":orange[Configuration]")
    model = st.selectbox("Model", MODELS)
    dataset = st.selectbox("Dataset", DATASET)
    experiment = st.selectbox("Experiment", list(EXPERIMENTS.keys()))
    domain, comparison_domain = None, None
    downsampled = st.selectbox("Downsampled?", [False, True])

    if dataset == "copyVSfactDomain":
        domain = st.selectbox("Domain", DOMAINS)

    st.title(":green[Comparison Configuration]")
    comparison = st.selectbox("Compare?", [False, True])

    if comparison:
        st.title(":orange[Comparison Configuration]")
        comparison_model = st.selectbox("Comparison Model", MODELS)
        comparison_dataset = st.selectbox("Comparison Dataset", DATASET)
        comparison_experiment = st.selectbox("Comparison Experiment", list(EXPERIMENTS.keys()))
        comparison_downsampled = st.selectbox("Comparison Downsampled?", [False, True])
        if comparison_dataset == "copyVSfactDomain":
            comparison_domain = st.selectbox("Comparison Domain", DOMAINS)

##  Main Window

def main_window(model, experiment, dataset, domain, downsampled):
    try:
        if domain:
            plots_folder = f"results/python_paper_plots/{model}_{dataset}_{EXPERIMENTS[experiment]}/{domain}"
        else:
            plots_folder = f"results/python_paper_plots/{model}_{dataset}_{EXPERIMENTS[experiment]}"
        if downsampled:
            plots_folder += f"_downsampled"

        plots = glob.glob(f"{plots_folder}/*.pdf")
        if not plots:
            st.error("No plots present for this configuration!")
        for plot in sorted(plots):
            if plot.endswith("multiplied_pattern.pdf"):
                continue
            st.markdown(f":blue-background[{os.path.basename(plot).capitalize()}]")
            pdf_viewer(plot, key=uuid.uuid4().hex)
    except Exception as e:
        st.error(e)


if comparison:
    col1, col2 = st.columns(2)

    with col1:
        st.header(f":orange[{experiment}]", divider=True)
        st.info(f"""
        **Model:** :violet[{model}]\n
        **Dataset:** :violet[{dataset}]\n
        **Domain:** :violet[{domain}]\n
        **Downsampled:** :violet[{downsampled}]\n
        """)
        main_window(model, experiment, dataset, domain, downsampled)
    with col2:
        st.header(f":orange[{comparison_experiment}]", divider=True)
        st.info(f"""
        **Model:** :violet[{comparison_model}]\n
        **Dataset:** :violet[{comparison_dataset}]\n
        **Domain:** :violet[{comparison_domain}]\n
        **Downsampled:** :violet[{comparison_downsampled}]\n
        """)
        main_window(comparison_model, comparison_experiment, comparison_dataset, comparison_domain, comparison_downsampled)

else:
    st.header(f":orange[{experiment}]", divider=True)
    main_window(model, experiment, dataset, domain, downsampled)