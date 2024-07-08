import torch
import torch.nn as nn
import torch.optim as optim


def lr_schedule(epoch,lr):
    decay_factor = 0.1
    if epoch % 10 == 0 and epoch != 0:
        return lr * decay_factor
    else:
        return lr


def train_model(model, train_loader = None, test_loader = None, list_clients= None, num_epochs=10, learning_rate=0.001, lr_scheduler=None):
    
    if not train_loader:
        trained_obj = train_federated(model, list_clients = list_clients, rounds=3, epochs=num_epochs, lr = learning_rate)
        trained_model = trained_obj.model
    
    else:
        trained_model = train_central(model, train_loader, test_loader, num_epochs=num_epochs,
                                       learning_rate = learning_rate)
    
    return trained_model




def train_federated(my_server, list_clients, rounds=3, epochs=200,lr =0.001):
    """
    Controler function to launch federated learning

    Parameters
    ----------
    main_model:
        Define the central node model :

    data_dict : Dictionary
    Contains training and validation data for the different FL nodes

    rounds : int
        Number of federated learning rounds

    epoch : int
        Number of training epochs in each round

    model_path : str
        Define the path where to save the models

    """
    from src.utils_training import train_model
    from src.utils_fed import send_server_model_to_client, fedavg
    
    for _ in range(0, rounds):

        send_server_model_to_client(list_clients, my_server)
        for client in list_clients:
            client.model = train_central(client.model, client.data_loader['train'],client.data_loader['test'], epochs, lr)

        fedavg(my_server, list_clients)

    return my_server





def train_central(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, optimizer=optim.SGD, lr_scheduler=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=learning_rate) 
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Apply learning rate decay if lr_scheduler is provided
        if lr_scheduler is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scheduler(epoch, param_group['lr'])

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

    return model

def loss_calculation(model, train_loader): 
    import torch
    import torch.nn as nn

    # Assuming you have a PyTorch model named 'model' and its training data loader named 'train_loader'

    # Define your loss function
    criterion = nn.CrossEntropyLoss()  # Example, adjust based on your task

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to accumulate loss and total number of samples
    total_loss = 0.0
    total_samples = 0

    # Iterate through the training data loader
    with torch.no_grad():
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Accumulate the loss and the total number of samples
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    # Calculate the average loss
    average_loss = total_loss / total_samples

    return average_loss

def test_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track accuracy
    correct = 0
    total = 0
    test_loss = 0.0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate over the test dataset
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)

    # Calculate accuracy on the test set
    accuracy = correct / total

    # Print the test loss and accuracy
    return accuracy