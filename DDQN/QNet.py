import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, n_actions, input_dims):
        super().__init__()

        self.qnet = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        output = self.qnet(x)
        return output