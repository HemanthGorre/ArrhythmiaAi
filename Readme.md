

````markdown
# ArrhythmiaAI: End-to-End Arrhythmia Detection

**ArrhythmiaAI** is a fully reproducible, production-ready deep learning pipeline for real-time arrhythmia detection using the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). The project covers the complete ML lifecycle from streaming data ingestion, preprocessing, robust label mapping, and model training (ConvGRU, LSTM, GRU), to deployment via TorchScript/ONNX, a FastAPI REST API, and a user-friendly Streamlit web demo. The codebase emphasizes transparency, explainability, and clinical relevance, making it ideal for both research and real-world prototyping.

---

## Key Features

- **End-to-End Deep Learning Pipeline** – From raw ECG to clinical predictions.
- **Real-Time Streaming Support** – Simulate or process live ECG data in sliding windows.
- **Multiple Model Architectures** – Efficient ConvGRU, LSTM, and GRU implementations.
- **Explainability Integration** – Saliency maps and visual overlays for each prediction.
- **Reproducibility & Auditability** – Rigorous experiment logging, versioning, and configuration tracking.
- **Deployment-Ready** – Export models to TorchScript/ONNX; robust REST API via FastAPI.
- **Interactive Web Demo** – Streamlit app for ECG upload, prediction, and explainability.
- **Comprehensive Documentation** – Clear code, comments, and reproducible workflows.
- **Safe and Responsible** – Designed for research, not for direct clinical use.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Training & Evaluation](#training--evaluation)
4. [Deployment](#deployment)
5. [Explainability](#explainability)
6. [Reproducibility & Logging](#reproducibility--logging)
7. [Directory Structure](#directory-structure)
8. [Screenshots](#screenshots)
9. [Citations](#citations)
10. [License and Contact](#license-and-contact)
11. [Disclaimer](#disclaimer)

---

## Getting Started

### Prerequisites

- Python 3.8+ (tested on 3.8/3.9)
- [PyTorch](https://pytorch.org/) (>=1.10, GPU recommended)
- pip
- [virtualenv](https://virtualenv.pypa.io/) or [conda](https://docs.conda.io/) (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arrhythmiaai.git
cd arrhythmiaai

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or, for development with extras
pip install -r requirements-dev.txt
````

---

## Data Preparation

### 1. **Download MIT-BIH Arrhythmia Database**

* Register and download from [PhysioNet MIT-BIH page](https://physionet.org/content/mitdb/1.0.0/).
* Place raw data in a folder, e.g., `data/raw/`.

### 2. **Preprocess Data**

* Use provided scripts in `src/` to preprocess, segment, and label the data:

```bash
# Example preprocessing command
python src/preprocess_mitbih.py --input data/raw/ --output data/processed/
```

* Outputs: NPY arrays (`train_signals_6class.npy`, etc.), label maps (`labels_6class.json`), and class distributions.

#### Example Processed Data Paths

* `data/processed/train_signals_6class.npy`
* `data/processed/train_labels_6class.npy`
* `data/processed/labels_6class.json`

---

## Training & Evaluation

### **Train a Model**

```bash
python src/train_convgru.py --config configs/convgru_6class.yaml
# or, for LSTM/GRU:
python src/train_lstm.py --config configs/lstm_6class.yaml
```

* Configurations control data paths, hyperparameters, and logging.
* Checkpoints and logs are saved in `models/` and `logs/`.

### **Evaluate a Model**

```bash
python src/eval_convgru.py --model models/convgru_best_6class.pt --data data/processed/test_signals_6class.npy --labels data/processed/test_labels_6class.npy
```

* Evaluation outputs confusion matrices, classification reports, and inference timing.

---

## Deployment

### **Export to TorchScript/ONNX**

```bash
python src/export_model.py --model models/convgru_best_6class.pt --format onnx --output models/convgru_best_6class.onnx
```

### **Run the FastAPI REST API**

```bash
cd deploy/
uvicorn api:app --reload
```

* The API will be available at `http://127.0.0.1:8000`.

#### Example API Request (Python)

```python
import requests
import numpy as np

signal = np.load('demo/sample_ecg_window.npy').tolist()  # Example shape: [1, 720]
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"signal": signal}
)
print(response.json())
```

#### Example API Request (curl)

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"signal": [[0.01, 0.02, ... ]] }'
```

### **Run the Streamlit Web Demo**

```bash
cd demo/
streamlit run app.py
```

* Open the local Streamlit address in your browser (typically [http://localhost:8501](http://localhost:8501)).
* Upload an ECG `.npy` or `.csv` file to visualize predictions and saliency maps.

---

## Explainability

* The API and web demo return both predicted classes and **saliency overlays** highlighting the most influential ECG signal regions.
* To generate or visualize explainability outputs programmatically:

  * Use the provided `explain.py` utilities in `src/` to compute and save saliency maps.
* Example:

```bash
python src/explain.py --model models/convgru_best_6class.pt --signal demo/sample_ecg_window.npy --output demo/saliency_overlay.png
```

---

## Reproducibility & Logging

* All experiments are version-controlled: configs, splits, logs, and model checkpoints are stored with unique IDs.
* Logs and experiment histories are saved in `logs/` (CSV, JSON, and plots).
* Random seeds, data splits, and model parameters are recorded in each config file and training log.
* To reproduce results:

  1. Use the provided config files and scripts.
  2. Check logs and checkpoints in `logs/` and `models/`.

---

## Directory Structure

```text
arrhythmiaai/
│
├── models/            # Model checkpoints, TorchScript, ONNX exports
├── data/
│   └── processed/     # Preprocessed NPY arrays, label maps
├── src/               # Training, evaluation, preprocessing, explainability scripts
├── deploy/            # FastAPI REST API, deployment scripts
├── demo/              # Streamlit web app and demo assets
├── logs/              # Experiment logs, metrics, training curves
├── configs/           # Example configuration files
├── README.md
└── requirements.txt
```

* **models/**: Stores trained models, exported files.
* **data/processed/**: Contains processed ECG signals, labels, and label maps.
* **src/**: Source code for all ML pipeline steps.
* **deploy/**: FastAPI code for API deployment.
* **demo/**: Streamlit UI for clinical/non-technical users.
* **logs/**: Training logs, evaluation metrics, visualizations.
* **configs/**: YAML or JSON configuration files.
* **requirements.txt**: Python dependencies.

---

## Screenshots

* **\[Add image here]**: End-to-End Pipeline Diagram
* **\[Add image here]**: Streamlit Web Demo (ECG upload & explainability)
* **\[Add image here]**: Confusion Matrix for 6-class ConvGRU
* **\[Add image here]**: Saliency Map Overlay Example

*(Add images in `demo/assets/` and reference here)*

---

## Citations

If you use ArrhythmiaAI or its codebase, please cite:

```bibtex
@report{gorrepati2025arrhythmiaai,
  title = {ArrhythmiaAI: End-to-End Arrhythmia Detection},
  author = {Hemanth Gorrepati},
  institution = {Georgia State University},
  year = {2025},
  note = {https://github.com/yourusername/arrhythmiaai}
}
```

* MIT-BIH Arrhythmia Database:
  G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.*, vol. 20, no. 3, pp. 45-50, 2001.

* Please cite any deep learning methods, frameworks, or third-party datasets as appropriate (see report’s References).

---

## License and Contact

This project is released under the **MIT License**.
See [LICENSE](LICENSE) for details.

**Contact:**
Hemanth Gorrepati
[hgorrepati1@student.gsu.edu](mailto:hgorrepati1@student.gsu.edu)

---

## Disclaimer

> **ArrhythmiaAI is a research tool and is NOT intended for clinical diagnosis or direct patient care. Use only for educational or research purposes.**

---

```

---

**Let me know if you want:**  
- Example config files, extra sample code, or more detailed API docs  
- Help generating/bundling demo images  
- A LICENSE file template  
- Adjustments for PyPI or Docker deployment  
- Or any section tailored for a specific audience!

**This README is ready to copy-paste—just update the GitHub repo URL and add images as needed.**
```
