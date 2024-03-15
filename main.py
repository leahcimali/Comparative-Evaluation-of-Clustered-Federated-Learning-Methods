
from os import makedirs
import sys


import torch
import importlib
import contextlib
from pathlib import Path


import src.config
from src.models import MnistNN 
from src.fedclass import Client, Server
from src.utils_data import data_distribution 
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import time
import copy
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from time import sleep
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load your data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for train and test data
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64)

# Instantiate the model
model = MnistNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    # Iterate over the training dataset
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item() * inputs.size(0)
    
    # Calculate average training loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    
    # Evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate over the test dataset
        for inputs, labels in test_loader:
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate accuracy on the test set
    accuracy = correct / total
    
    # Print the loss and accuracy for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2%}')