import torch.nn as nn
import torch.nn.functional as F

class SimpleLinear(nn.Module):
    # Simple fully connected neural network with ReLU activations with a single hidden layer of size 200
    def __init__(self, h1=200):
        super().__init__()
        self.fc1 = nn.Linear(28*28, h1)
        self.fc2 = nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
