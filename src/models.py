import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinear(nn.Module):
    """ Fully connected neural network with a single hidden layer of default size 200 and ReLU activations"""
    
    def __init__(self, in_size, n_channels):
        
        """ Initialization function
        Arguments:
            h1: int
                Desired size of the hidden layer 
        """
        super().__init__()
        self.fc1 = nn.Linear(in_size*in_size,200)
        self.fc2 = nn.Linear(200, 10)
        self.in_size = in_size

    def forward(self, x: torch.Tensor):
        
        """ Forward pass function through the network
        
        Arguments:
            x : torch.Tensor
                input image of size in_size x in_size

        Returns: 
            log_softmax probabilities of the output layer
        """
        
        x = x.view(-1, self.in_size * self.in_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class SimpleConv(nn.Module):

    """ Convolutional neural network with 3 convolutional layers and one fully connected layer
    """

    def __init__(self,  in_size, n_channels):
        """ Initialization function
        """
        super(SimpleConv, self).__init__()
                
        self.conv1 = nn.Conv2d(n_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3,  padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3,  padding=1)
        
        self.img_final_size = int(in_size / 8)
        
        self.fc1 = nn.Linear(16 * self.img_final_size * self.img_final_size, 10)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(p=0.2)

    def flatten(self, x : torch.Tensor):
    
        """Function to flatten a layer
        
            Arguments: 
                x : torch.Tensor

            Returns:
                flattened Tensor
        """
    
        return x.reshape(x.size()[0], -1)
    
    def forward(self, x : torch.Tensor):
        """ Forward pass through the network which returns the softmax probabilities of the output layer

        Arguments:
            x : torch.Tensor
                input image to use for training
        """
        
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)