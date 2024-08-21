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


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding = 1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 4 * 4, 10)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)

    def flatten(self, x):
        return x.reshape(x.size()[0], -1)
    
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)