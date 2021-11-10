import torch
from torch import nn


class ReviewClassifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout_p=0.5):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_features,
                      kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_features, out_channels=hidden_features,
                      kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x_in, apply_softmax=False):
        y_out = self.conv(x_in)
        y_out = self.fc1(y_out)

        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)
        return y_out