import torch
import torch.nn as nn
import torch.optim as optim

lr = 0.01


def lr_schedule(epoch,lr):
    decay_factor = 0.1
    if epoch % 10 == 0 and epoch != 0:
        return lr * decay_factor
    else:
        return lr



def train_benchmark(list_clients, row_exp, i, main_model, training_type="centralized"):
        
        from src.utils_training import train_model
        from src.utils_data import centralize_data
        import copy


        train_loader, test_loader = centralize_data(list_clients)
    
        if "federated" in training_type:
            model_server = copy.deepcopy(main_model)
            model_trained = train_model(model_server, None, None, list_clients, row_exp)
        
        else:
            model_trained = train_model(main_model, train_loader, test_loader, list_clients, row_exp) 
        
        return model_trained, test_loader



def test_benchmark(model_trained, list_clients, test_loader):    
         
        from src.utils_training import test_model
        
        clients_accs = []
        model_tested = test_model(model_trained, test_loader)    

        for client in list_clients : 
            acc = test_model(model_trained, client.data_loader['test'])*100
            clients_accs.append(acc)

        return model_tested




def train_model(model_server, train_loader, test_loader, list_clients, row_exp):
#train_model(model, train_loader = None, test_loader = None, list_clients= None, num_epochs=10, learning_rate=0.001, lr_scheduler=None):
    
    if not train_loader:
        trained_obj = train_federated(model_server, list_clients, row_exp)
        trained_model = trained_obj.model
    
    else:
        trained_model = train_central(model_server, train_loader, test_loader, row_exp)
    
    return trained_model




def train_federated(main_model, list_clients, row_exp):
    """
    Controler function to launch federated learning

    Parameters
    ----------
    main_model:
        Define the central node model :
    """

    from src.utils_fed import send_server_model_to_client, fedavg
    
    for _ in range(0, row_exp['federated_rounds']):

        send_server_model_to_client(list_clients, main_model)

        for client in list_clients:
            client.model = train_central(client.model, client.data_loader['train'],client.data_loader['test'], row_exp)

        fedavg(main_model, list_clients)

    return main_model





def train_central(main_model, train_loader, test_loader, row_exp, lr_scheduler=None):

    criterion = nn.CrossEntropyLoss()
    optimizer=optim.SGD
    optimizer = optimizer(main_model.parameters(), lr=lr) 
    
    for epoch in range(row_exp['centralized_epochs']):
        main_model.train()  # Set the model to training mode
        running_loss = 0.0

        # Apply learning rate decay if lr_scheduler is provided
        if lr_scheduler is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scheduler(epoch, param_group['lr'])

        # Iterate over the training dataset
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = main_model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item() * inputs.size(0)

        # Calculate average training loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluate the model on the test set
        main_model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        # Disable gradient calculation for evaluation
        with torch.no_grad():
            # Iterate over the test dataset
            for inputs, labels in test_loader:
                outputs = main_model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy on the test set
        accuracy = correct / total

        # Print the loss and accuracy for each epoch
        print(f"Epoch [{epoch+1}/{row_exp['centralized_epochs']}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2%}")

    return main_model

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