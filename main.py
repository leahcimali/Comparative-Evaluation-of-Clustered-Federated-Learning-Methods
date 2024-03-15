
from os import makedirs
import sys


import torch
import importlib
import contextlib
from pathlib import Path


import src.config
from src.models import MnistNN 
from src.fedclass import Client, Server
from src.utils_data import data_distribution, data_preparation
import numpy as np
seed = 42
torch.manual_seed(seed)

# CONFIG VARIABLE 
number_of_clients = 5
number_of_samples_by_clients = 10
clientdata = data_distribution(number_of_clients, number_of_samples_by_clients)
clientlist = []
for id in range(number_of_clients):
    clientlist.append(Client(id,clientdata[id]))
my_server = Server(MnistNN(),4)
print(type(clientlist[0].data['y']))
print(np.unique(clientlist[0].data['y']))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import time
import copy
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = MnistNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

data_preparation(clientlist[0],90)