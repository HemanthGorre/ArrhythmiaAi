# File: src/model.py  (or deploy/convgru_model.py)
# Purpose: Define ConvGRUClassifier for both training and deployment.

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

class ConvGRUClassifier(nn.Module):
    """
    ConvGRU hybrid for ECG window classification.
    Input shape: [batch, 1, win_size]
    """
    def __init__(self, input_dim=1, conv_channels=16, gru_hidden=32, num_classes=6, win_size=720):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=15, stride=1, padding=7)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(conv_channels, gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, num_classes)
        self.win_size = win_size

    def forward(self, x, h=None):
        # x: [batch, 1, win_size]
        x = self.conv1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch, win, channels]
        if h is None:
            h = torch.zeros(1, x.size(0), self.gru.hidden_size, device=x.device, dtype=x.dtype)
        out, h = self.gru(x, h)
        out_last = out[:, -1, :]
        logits = self.fc(out_last)
        return logits, h
