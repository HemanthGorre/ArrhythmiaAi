# deploy/benchmark_inference.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deploy')))
from convgru_deployer import ConvGRUDeployer
import numpy as np
import time
import torch

test_data_path = 'data/processed/test_signals_6class.npy'
Xtest = np.load(test_data_path)
batch_sizes = [32, 128]

for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    deployer = ConvGRUDeployer(
        model_path='models/convgru_6class_deploy.ptc',
        label_map_path='data/processed/labels_6class.json',
        device=device
    )
    print(f"Benchmarking on {device.upper()}")
    for bs in batch_sizes:
        batch = Xtest[:bs]
        deployer.predict(batch)  # warmup
        times = []
        for _ in range(10):
            t0 = time.time()
            deployer.predict(batch)
            times.append(time.time() - t0)
        print(f"Batch {bs}: mean {np.mean(times)*1000:.2f} ms, max {np.max(times)*1000:.2f} ms")
