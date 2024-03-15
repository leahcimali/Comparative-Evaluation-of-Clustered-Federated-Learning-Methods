import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
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
        epoch_loss = running_loss / len(train_loader.dataset)

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
