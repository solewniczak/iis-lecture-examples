import torch
from torch import nn


class ReviewClassifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features)
        )

    def forward(self, x_in, apply_softmax=False):
        y_out = self.model(x_in)

        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)
        return y_out