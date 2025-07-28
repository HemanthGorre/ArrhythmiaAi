import os, pandas as pd, torch, numpy as np, time, sys
from collections import defaultdict

RESULTS_DIR = "results"
LOGS_DIR = "logs"
MODELS_DIR = "models"
DATA_DIR = "data/processed"
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from model import GRUClassifier, LSTMClassifier, ConvGRUClassifier
from config import WIN_SIZE

models = ["gru", "lstm", "convgru"]
versions = ["unfiltered", "6class", "5class"]
devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
batch_sizes = [32, 128]

summary = []
missing = []

def safe_read_csv(f, **kwargs):
    try: return pd.read_csv(f, **kwargs)
    except Exception as e:
        print(f"Failed to read {f}: {e}")
        return None

def get_num_classes(test_preds_file):
    df = safe_read_csv(test_preds_file)
    if df is not None:
        return len(set(df['true']))
    return None

def try_get_auc(roc_path):
    # Macro/Micro ROC-AUC from .txt or parse from eval script stdout (if available)
    txt_path = roc_path.replace(".png", "_metrics.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                if "Macro ROC AUC:" in line:
                    macro = float(line.strip().split(":")[1].split("|")[0].strip())
                    micro = float(line.strip().split("|")[1].replace("Micro ROC AUC:","").strip())
                    return macro, micro
    return None, None

for arch in models:
    for version in versions:
        run = f"{arch}_{version}"
        report_file = os.path.join(RESULTS_DIR, f"{run}_classification_report.csv")
        log_file = os.path.join(LOGS_DIR, f"train_{arch}_{version}_log.csv")
        test_preds_file = os.path.join(RESULTS_DIR, f"{run}_test_preds.csv")
        model_file = os.path.join(MODELS_DIR, f"{arch}_best_{version}.pt")
        roc_file = os.path.join(RESULTS_DIR, f"{run}_roc.png")
        # Sanity checks
        for p in [report_file, log_file, test_preds_file, model_file]:
            if not os.path.exists(p):
                missing.append(run)
                continue
        # Model size/params
        model_size = os.path.getsize(model_file) / (1024 * 1024)
        n_cls = get_num_classes(test_preds_file)
        if n_cls is None:
            print(f"Could not infer n_classes for {run}")
            continue
        model_cls = {'gru': GRUClassifier, 'lstm': LSTMClassifier, 'convgru': ConvGRUClassifier}[arch]
        model = model_cls(num_classes=n_cls, win_size=WIN_SIZE)
        ckpt = torch.load(model_file, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        n_params = sum(p.numel() for p in model.parameters())
        # Classification metrics
        report = safe_read_csv(report_file, index_col=0)
        log = safe_read_csv(log_file)
        if report is None or log is None: continue
        val_acc = log['val_acc'].max()
        try:
            val_macro_f1 = float(report.loc['macro avg', 'f1-score'])
            test_acc = float(report.loc['accuracy', 'precision'])
            test_macro_f1 = float(report.loc['macro avg', 'f1-score'])
            # Robustness: min F1 any class
            classwise_f1 = report.loc[[c for c in report.index if c not in ['accuracy','macro avg','weighted avg']], 'f1-score']
            min_class_f1 = classwise_f1.min()
            rare_classes = ','.join(classwise_f1[classwise_f1 == min_class_f1].index)
        except Exception as e:
            print(f"Metrics error {run}: {e}")
            continue
        # ROC-AUC from text or skip if not present
        macro_auc, micro_auc = try_get_auc(roc_file)
        # Inference latency: measure for both cpu/gpu if possible, both batch sizes
        latency = {}
        for device in devices:
            for bs in batch_sizes:
                X = np.load(f"{DATA_DIR}/test_signals_{version}.npy")
                X = torch.tensor(X[:bs], dtype=torch.float32).unsqueeze(1)
                X = X.to(device)
                model = model.to(device).eval()
                # Warmup
                with torch.no_grad():
                    for _ in range(5): model(X)
                    if device == "cuda": torch.cuda.synchronize()
                times = []
                with torch.no_grad():
                    for _ in range(10):
                        t0 = time.time()
                        model(X)
                        if device == "cuda": torch.cuda.synchronize()
                        times.append(time.time()-t0)
                latency[f"{device}_b{bs}"] = np.mean(times)*1000
        notes = f"Min F1={min_class_f1:.2f} (class: {rare_classes})"
        summary.append(dict(
            model=arch,
            data_version=version,
            val_acc=val_acc,
            test_acc=test_acc,
            val_macro_f1=val_macro_f1,
            test_macro_f1=test_macro_f1,
            macro_auc=macro_auc,
            micro_auc=micro_auc,
            model_size_MB=round(model_size,2),
            n_params=n_params,
            latency_cpu_b32=latency.get("cpu_b32",None),
            latency_cpu_b128=latency.get("cpu_b128",None),
            latency_gpu_b32=latency.get("cuda_b32",None),
            latency_gpu_b128=latency.get("cuda_b128",None),
            notes=notes
        ))

df = pd.DataFrame(summary)
out_csv = os.path.join(RESULTS_DIR, "deployment_model_metrics_summary.csv")
df.to_csv(out_csv, index=False)
print("\n==================== SUMMARY TABLE ====================")
print(df)
print(f"\nSaved full deployment metrics summary to {out_csv}\n")

if missing:
    print("\nMissing/incomplete runs:", missing)
else:
    print("\nAll runs found and analyzed!")
