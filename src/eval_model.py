import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import ConvGRUClassifier, GRUClassifier, LSTMClassifier
from config import WIN_SIZE

DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(split, suffix):
    X = np.load(os.path.join(DATA_DIR, f"{split}_signals{suffix}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"{split}_labels{suffix}.npy"))
    X = torch.tensor(X, dtype=torch.float32)
    if X.ndim == 2: X = X.unsqueeze(1)
    return X, np.array(y)

def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.colorbar()
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix: {out_path}")

def plot_roc(y_true, y_score, n_classes, out_path, label_names):
    plt.figure(figsize=(8, 6))
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label_names[i]} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved ROC curves: {out_path}")

def select_model(arch, num_classes):
    if arch == "convgru": return ConvGRUClassifier(num_classes=num_classes, win_size=WIN_SIZE)
    if arch == "gru":     return GRUClassifier(num_classes=num_classes, win_size=WIN_SIZE)
    if arch == "lstm":    return LSTMClassifier(num_classes=num_classes, win_size=WIN_SIZE)
    raise ValueError(f"Unknown architecture: {arch}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, choices=['convgru', 'gru', 'lstm'])
    parser.add_argument('--data_version', required=True, choices=['unfiltered', '6class', '5class'])
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    suffix = f"_{args.data_version}"
    label_map_path = f"labels_{args.data_version}.json"
    with open(os.path.join(DATA_DIR, label_map_path)) as f:
        label_map = json.load(f)
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    n_classes = len(label_names)
    print(f"Classes: {label_names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = select_model(args.arch, n_classes).to(device)
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"{args.arch}_best_{args.data_version}.pt"), map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    X_test, y_test = load_data('test', suffix)
    all_logits, all_preds = [], []
    with torch.no_grad():
        for i in range(0, X_test.shape[0], args.batch_size):
            xb = X_test[i:i+args.batch_size].to(device)
            logits, _ = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_logits.append(probs)
            all_preds.extend(np.argmax(probs, axis=1))
    all_logits = np.concatenate(all_logits, axis=0)
    y_pred = np.array(all_preds)

    # Save per-window predictions
    df = pd.DataFrame({
        "true": y_test,
        "pred": y_pred
    })
    df.to_csv(os.path.join(RESULTS_DIR, f"{args.arch}_{args.data_version}_test_preds.csv"), index=False)
    print(f"Saved per-window predictions: {os.path.join(RESULTS_DIR, f'{args.arch}_{args.data_version}_test_preds.csv')}")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=label_names, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(RESULTS_DIR, f"{args.arch}_{args.data_version}_classification_report.csv"))
    print(report_df)

    # Confusion matrix plot
    plot_confusion(y_test, y_pred, label_names, os.path.join(RESULTS_DIR, f"{args.arch}_{args.data_version}_confusion.png"))

    # ROC AUC micro/macro, and per-class ROC
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_test, classes=range(n_classes))
        macro_auc = roc_auc_score(y_true_bin, all_logits, average='macro')
        micro_auc = roc_auc_score(y_true_bin, all_logits, average='micro')
        print(f"Macro ROC AUC: {macro_auc:.4f} | Micro ROC AUC: {micro_auc:.4f}")
        plot_roc(y_test, all_logits, n_classes, os.path.join(RESULTS_DIR, f"{args.arch}_{args.data_version}_roc.png"), label_names)
    except Exception as e:
        print("ROC/AUC could not be computed (likely due to rare classes).", e)

if __name__ == "__main__":
    main()
