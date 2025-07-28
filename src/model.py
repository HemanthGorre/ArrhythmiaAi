from config import WIN_SIZE

# src/model.py
import torch
import torch.nn as nn

class ConvGRUClassifier(nn.Module):
    def __init__(self, input_dim=1, conv_channels=16, gru_hidden=32, num_classes=6, win_size=720):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=15, stride=1, padding=7)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(conv_channels, gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, num_classes)
        self.win_size = win_size

    def forward(self, x, h=None):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch, win, channels]
        if h is None:
            h = torch.zeros(1, x.size(0), self.gru.hidden_size, device=x.device, dtype=x.dtype)
        out, h = self.gru(x, h)
        out_last = out[:, -1, :]
        logits = self.fc(out_last)
        return logits, h

class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        num_classes=5,
        win_size=WIN_SIZE,
        dropout=0.2,
        bidirectional=True,
        pooling="mean"
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.pooling = pooling
        feature_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(feature_dim, num_classes)
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
    def forward(self, x, h=None):
        x = x.transpose(1, 2)
        if h is None:
            h = torch.zeros(
                self.gru.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.gru.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        out, h = self.gru(x, h)
        if self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "last":
            pooled = out[:, -1, :]
        elif self.pooling == "attention":
            scores = self.attn(out)
            attn_weights = torch.softmax(scores, dim=1)
            pooled = (out * attn_weights).sum(dim=1)
        else:
            pooled = out.mean(dim=1)
        logits = self.fc(pooled)
        return logits, h

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        num_classes=5,
        win_size=WIN_SIZE,
        dropout=0.2,
        bidirectional=True,
        pooling="mean"
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.pooling = pooling
        feature_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(feature_dim, num_classes)
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
    def forward(self, x, h=None):
        x = x.transpose(1, 2)
        if h is None:
            h = (
                torch.zeros(
                    self.lstm.num_layers * (2 if self.bidirectional else 1),
                    x.size(0),
                    self.lstm.hidden_size,
                    device=x.device,
                    dtype=x.dtype,
                ),
                torch.zeros(
                    self.lstm.num_layers * (2 if self.bidirectional else 1),
                    x.size(0),
                    self.lstm.hidden_size,
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
        out, (hn, cn) = self.lstm(x, h)
        if self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "last":
            pooled = out[:, -1, :]
        elif self.pooling == "attention":
            scores = self.attn(out)
            attn_weights = torch.softmax(scores, dim=1)
            pooled = (out * attn_weights).sum(dim=1)
        else:
            pooled = out.mean(dim=1)
        logits = self.fc(pooled)
        return logits, (hn, cn)
