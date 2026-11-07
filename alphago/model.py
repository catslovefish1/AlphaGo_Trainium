"""Policy-value residual network."""


def build_model(size, channels, blocks):
    import torch.nn as nn
    import torch.nn.functional as F

    class Residual(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.c1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
            self.n1 = nn.GroupNorm(4, width)
            self.c2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
            self.n2 = nn.GroupNorm(4, width)

        def forward(self, x):
            y = F.relu(self.n1(self.c1(x)))
            y = self.n2(self.c2(y))
            return F.relu(x + y)

    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(4, channels, 3, padding=1, bias=False),
                nn.GroupNorm(4, channels),
                nn.ReLU(),
            )
            self.blocks = nn.Sequential(*[Residual(channels) for _ in range(blocks)])
            self.policy = nn.Sequential(
                nn.Conv2d(channels, 4, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4 * size * size, size * size + 1),
            )
            self.value = nn.Sequential(
                nn.Conv2d(channels, 2, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * size * size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            hidden = self.blocks(self.stem(x))
            return self.policy(hidden), self.value(hidden).squeeze(-1)

    return PolicyValueNet()
