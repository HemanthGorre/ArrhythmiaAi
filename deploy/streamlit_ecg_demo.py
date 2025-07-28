# deploy/streamlit_ecg_demo.py
"""
Streamlit demo for MIT-BIH ConvGRU 6-class ECG arrhythmia classifier.

- Upload or paste an ECG window or batch (shape [720] or [N, 720])
- See predicted class, probabilities, and a saliency (input-gradient) plot
- Uses deploy/convgru_deployer.py (ConvGRUDeployer class)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deploy')))
from convgru_deployer import ConvGRUDeployer
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----- Settings -----
MODEL_PATH = 'models/convgru_6class_deploy.ptc'
LABEL_MAP_PATH = 'data/processed/labels_6class.json'
DEVICE = 'cpu'  # For demo; use 'cuda' if on a GPU server

# ----- Load Model -----
@st.cache_resource
def get_deployer():
    return ConvGRUDeployer(
        model_path=MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        device=DEVICE
    )

deployer = get_deployer()
st.title("MIT-BIH ECG Arrhythmia Demo: ConvGRU (6-class)")
st.write(f"**Class order:** {deployer.class_order}")
st.write("Upload an ECG window (.npy or .csv), batch of windows, or paste values below.")

# ----- Upload Option -----
uploaded = st.file_uploader("Upload ECG windows (.npy or .csv)", type=['npy', 'csv'])
X = None

if uploaded:
    if uploaded.name.endswith('.npy'):
        X = np.load(uploaded)
    else:
        X = np.loadtxt(uploaded, delimiter=',')
    if X.ndim == 1:
        X = X[None, :]
    st.write(f"**Loaded input shape:** {X.shape}")

    preds, pred_labels = deployer.predict(X)
    probs = deployer.predict_proba(X)
    st.write("### Predicted classes:", pred_labels)
    st.write("### Class probabilities:")
    st.write(probs)
    st.bar_chart(np.bincount(preds, minlength=len(deployer.class_order)))

    # --- For each window: plot ECG and saliency ---
    for i, (label, arr) in enumerate(zip(pred_labels, X)):
        st.markdown(f"---\n#### Sample {i}: Predicted **{label}**")
        st.line_chart(arr)
        # Saliency
        sal = deployer.saliency(arr)
        fig, ax1 = plt.subplots()
        ax1.plot(arr, color='b', label='ECG')
        ax2 = ax1.twinx()
        ax2.plot(sal, color='r', alpha=0.5, label='Saliency')
        ax1.set_title(f"ECG and Saliency (Pred: {pred_labels[0]})")
        ax1.set_xlabel("Sample index")
        ax1.set_ylabel("ECG amplitude")
        ax2.set_ylabel("Saliency (|grad|)")
        st.pyplot(fig)
    # ----- Sample Option -----

# ----- Paste Option -----
else:
    st.info("Or paste a single window (comma-separated, 720 values):")
    txt = st.text_area("ECG window (720 comma-separated numbers):")
    if txt.strip():
        arr = np.array([float(x) for x in txt.strip().split(",")])
        if arr.shape[0] != 720:
            st.error("Input must have exactly 720 values!")
        else:
            preds, pred_labels = deployer.predict(arr[None, :])
            probs = deployer.predict_proba(arr[None, :])
            st.write("### Predicted class:", pred_labels[0])
            st.write("### Class probabilities:", probs)
            st.line_chart(arr)
            # Saliency
            sal = deployer.saliency(arr)
            fig, ax1 = plt.subplots()
            ax1.plot(arr, color='b', label='ECG')
            ax2 = ax1.twinx()
            ax2.plot(sal, color='r', alpha=0.5, label='Saliency')
            ax1.set_title(f"ECG and Saliency (Pred: {pred_labels[0]})")
            ax1.set_xlabel("Sample index")
            ax1.set_ylabel("ECG amplitude")
            ax2.set_ylabel("Saliency (|grad|)")
            st.pyplot(fig)

st.markdown("---")
st.markdown(
    "🩺 *This demo predicts ECG arrhythmia type for a 2-second window (720 samples, 360Hz) using a deep ConvGRU model. "
    "Saliency highlights which parts of the signal most influenced the model’s prediction. "
    "For best results, use real test windows from your dataset or try each class using sample CSVs from `data/processed/`.*"
)
