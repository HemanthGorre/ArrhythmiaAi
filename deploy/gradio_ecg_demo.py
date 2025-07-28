# deploy/gradio_ecg_demo.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deploy')))
from convgru_deployer import ConvGRUDeployer
import gradio as gr
import numpy as np

deployer = ConvGRUDeployer(
    model_path='models/convgru_6class_deploy.ptc',
    label_map_path='data/processed/labels_6class.json',
    device='cpu'
)

def predict_ecg(window):
    arr = np.asarray(window).flatten()
    if arr.ndim == 1:
        arr = arr[None, :]
    preds, pred_labels = deployer.predict(arr)
    probs = deployer.predict_proba(arr)
    return {"Prediction": pred_labels[0], "Class probabilities": probs.tolist()[0]}

iface = gr.Interface(
    fn=predict_ecg,
    inputs=gr.inputs.Dataframe(type="numpy", shape=(None,)),
    outputs="json",
    title="MIT-BIH ConvGRU Arrhythmia Classifier",
    description="Paste or upload a single ECG window (array of values)."
)
iface.launch()
