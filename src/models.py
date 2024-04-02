import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistNN(torch.nn.Module):
    
    """
    Class to define model for mnist 
    """
    
    def __init__(self, input_size=28*28, hidden_size=200, output_size=10):
        super(MnistNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleLinear(nn.Module):

    def __init__(self, h1=200):
        super().__init__()
        self.fc1 = nn.Linear(28*28, h1)
        self.fc2 = nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


