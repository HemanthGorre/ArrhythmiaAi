# deploy/serve_model.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deploy')))
from convgru_deployer import ConvGRUDeployer
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()
deployer = ConvGRUDeployer(
    model_path='models/convgru_6class_deploy.ptc',
    label_map_path='data/processed/labels_6class.json',
    device='cpu'
)

class BatchInput(BaseModel):
    windows: list  # list of list of floats

@app.post('/predict')
def predict(batch: BatchInput):
    X = np.array(batch.windows)
    _, pred_labels = deployer.predict(X)
    return {"predicted_labels": pred_labels}
