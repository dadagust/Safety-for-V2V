from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSpecCNN(nn.Module):
    def __init__(self, n_mels: int = 80, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
