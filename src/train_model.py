import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import GRUClassifier, LSTMClassifier, ConvGRUClassifier
from losses import WeightedFocalLoss
from config import WIN_SIZE

DATA_DIR = "data/processed"
MODEL_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def load_data(split, suffix):
    X = np.load(os.path.join(DATA_DIR, f"{split}_signals{suffix}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"{split}_labels{suffix}.npy"))
    X = torch.tensor(X, dtype=torch.float32)
    if X.ndim == 2: X = X.unsqueeze(1)
    return X, torch.tensor(y, dtype=torch.long)

def select_model(arch, num_classes):
    if arch == "gru":       return GRUClassifier(num_classes=num_classes, win_size=WIN_SIZE)
    if arch == "lstm":      return LSTMClassifier(num_classes=num_classes, win_size=WIN_SIZE)
    if arch == "convgru":   return ConvGRUClassifier(num_classes=num_classes, win_size=WIN_SIZE)
    raise ValueError(f"Unknown arch {arch}")

def compute_class_weights(labels, num_classes):
    unique, counts = np.unique(labels, return_counts=True)
    beta = 0.999
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / eff_num
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32)
    return class_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, choices=['gru', 'lstm', 'convgru'])
    parser.add_argument('--data_version', required=True, choices=['unfiltered', '6class', '5class'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    suffix = f"_{args.data_version}"
    label_map_path = f"labels_{args.data_version}.json"
    with open(os.path.join(DATA_DIR, label_map_path)) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    X_train, y_train = load_data('train', suffix)
    X_val, y_val = load_data('val', suffix)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = select_model(args.arch, num_classes).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    class_weights = compute_class_weights(y_train.numpy(), num_classes).to(device)
    criterion = WeightedFocalLoss(weight=class_weights, gamma=2.0)
    scaler = GradScaler() if args.fp16 else None
    best_acc, patience, max_pat = 0, 0, 6
    log_path = os.path.join(LOGS_DIR, f"train_{args.arch}_{args.data_version}_log.csv")
    checkpoint_path = os.path.join(MODEL_DIR, f"{args.arch}_best_{args.data_version}.pt")
    with open(log_path, "w") as logf:
        logf.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for epoch in range(args.epochs):
            model.train(); tr_loss=tr_corr=tr_tot=0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                if args.fp16:
                    with autocast():
                        logits, _ = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                else:
                    logits, _ = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    opt.step()
                tr_loss += loss.item() * xb.size(0)
                tr_corr += (logits.argmax(1) == yb).sum().item()
                tr_tot  += xb.size(0)
            tr_loss /= tr_tot; tr_acc = tr_corr / tr_tot
            # Validation
            model.eval(); v_loss=v_corr=v_tot=0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, _ = model(xb)
                    loss = criterion(logits, yb)
                    v_loss += loss.item() * xb.size(0)
                    v_corr += (logits.argmax(1) == yb).sum().item()
                    v_tot  += xb.size(0)
            v_loss /= v_tot; v_acc = v_corr / v_tot
            # ... inside training loop
            logf.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
            logf.flush()
            print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, val_loss={v_loss:.4f}, val_acc={v_acc:.4f}")

            if v_acc > best_acc:
                best_acc = v_acc; patience = 0
                torch.save({'model': model.state_dict()}, checkpoint_path)
            else:
                patience += 1
            if patience > max_pat:
                print("Early stopping!"); break

if __name__ == "__main__":
    main()
