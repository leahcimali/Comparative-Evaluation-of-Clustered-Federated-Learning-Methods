import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import pandas as pd

from src.models import ImageClassificationBase
from src.fedclass import Server



def run_cfl_server_side(model_server : Server, list_clients : list, row_exp : dict) -> pd.DataFrame:
    
    """ Driver function for server-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights (k-means).
    
    Arguments:

        main_model : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters

    Returns:

        df_results : dataframe with the experiment results
    """

    from src.utils_fed import k_means_clustering
    import copy
    import torch 

    torch.manual_seed(row_exp['seed'])

    model_server = train_federated(model_server, list_clients, row_exp, use_cluster_models = False)
    model_server.clusters_models= {cluster_id: copy.deepcopy(model_server.model) for cluster_id in range(row_exp['num_clusters'])}  
    setattr(model_server, 'num_clusters', row_exp['num_clusters'])

    k_means_clustering(list_clients, row_exp['num_clusters'], row_exp['seed'])

    model_server = train_federated(model_server, list_clients, row_exp, use_cluster_models = True)

    for client in list_clients :

        acc = test_model(model_server.clusters_models[client.cluster_id], client.data_loader['test'])    
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

    return df_results 


def run_cfl_client_side(model_server : Server, list_clients : list, row_exp : dict) -> pd.DataFrame:

    """ Driver function for client-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights (k-means).
    

    Arguments:

        main_model : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
    """

    from src.utils_fed import  set_client_cluster, fedavg
    import torch

    torch.manual_seed(row_exp['seed'])
    
    for _ in range(row_exp['federated_rounds']):

        for client in list_clients:

            client.model, _ = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)

        fedavg(model_server, list_clients)

        set_client_cluster(model_server, list_clients, row_exp)

    for client in list_clients :

        acc = test_model(model_server.clusters_models[client.cluster_id], client.data_loader['test'])
        setattr(client, 'accuracy', acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results


def run_benchmark(main_model : nn.Module, list_clients : list, row_exp : dict) -> pd.DataFrame:

    """ Benchmark function to calculate baseline FL results and ``optimal'' personalization results if clusters are known in advance

    Arguments:

        main_model : Type of Server model needed    
        list_clients : A list of Client Objects used as nodes in the FL protocol  
        row_exp : The current experiment's global parameters
    """

    import pandas as pd 
    import torch
    import copy

    from src.utils_data import centralize_data

    list_heterogeneities = list(dict.fromkeys([client.heterogeneity_class for client in list_clients]))
    
    torch.manual_seed(row_exp['seed'])
    torch.use_deterministic_algorithms(True)

    curr_model = main_model if row_exp['exp_type'] == 'global-federated' else main_model.model

    if row_exp['exp_type'] == 'pers-centralized':
        for heterogeneity_class in list_heterogeneities:
            list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
            train_loader, val_loader, test_loader = centralize_data(list_clients_filtered)
            model_trained, _ = train_central(curr_model, train_loader, val_loader, row_exp) 

            global_acc = test_model(model_trained, test_loader) 
                    
            for client in list_clients_filtered : 
    
                setattr(client, 'accuracy', global_acc)
    
    elif row_exp['exp_type'] == 'global-federated':
                
        model_server = copy.deepcopy(curr_model)
        model_trained = train_federated(model_server, list_clients, row_exp, use_cluster_models = False)

        _, _,test_loader = centralize_data(list_clients)
        global_acc = test_model(model_trained.model, test_loader) 
                    
        for client in list_clients : 
    
            setattr(client, 'accuracy', global_acc)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results


def train_federated(main_model, list_clients, row_exp, use_cluster_models = False):
    
    """Controler function to launch federated learning

    Arguments:

        main_model: Server model used in our experiment
        list_clients: A list of Client Objects used as nodes in the FL protocol  
        row_exp: The current experiment's global parameters
        use_cluster_models: Boolean to determine whether to use personalization by clustering
    """
    
    from src.utils_fed import send_server_model_to_client, send_cluster_models_to_clients, fedavg
    
    for i in range(0, row_exp['federated_rounds']):

        accs = []

        if use_cluster_models == False:
        
            send_server_model_to_client(list_clients, main_model)

        else:

            send_cluster_models_to_clients(list_clients, main_model)

        for client in list_clients:
            print(f"Training client {client.id} with dataset of size {client.data['x'].shape}")
            client.model, curr_acc = train_central(client.model, client.data_loader['train'], client.data_loader['val'], row_exp)
            accs.append(curr_acc)

        fedavg(main_model, list_clients)

    return main_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def evaluate(model : nn.Module, val_loader : DataLoader) -> dict:
    
    """ Returns a dict with loss and accuracy information"""
    model.to(device)
    model.eval()
    outputs =[]
    for batch in val_loader:
        # Move entire batch to the correct device
        batch = [item.to(device) for item in batch]
        
        # Call the validation step and append to outputs
        output = model.validation_step(batch,device)
        outputs.append(output)
    return model.validation_epoch_end(outputs)

def train_central(model: ImageClassificationBase, train_loader: DataLoader, val_loader: DataLoader, row_exp: dict):
    """ Main training function for centralized learning
    
    Arguments:
        model : Server model used in our experiment
        train_loader : DataLoader with the training dataset
        val_loader : Dataloader with the validation dataset
        row_exp : The current experiment's global parameters

    Returns:
        (model, history) : base model with trained weights / results at each training step
    """

    # Check if CUDA is available and set the device
    
    # Move the model to the appropriate device
    model.to(device)

    opt_func = torch.optim.SGD  # if row_exp['nn_model'] == "linear" else torch.optim.Adam
    lr = 0.001
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(row_exp['centralized_epochs']):
        
        model.train()
        train_losses = []
        
        for batch in train_loader:
            # Move batch to the same device as the model
            inputs, labels = [item.to(device) for item in batch]
    
            # Pass the unpacked inputs and labels to the model's training step
            loss = model.training_step((inputs, labels),device)            
            train_losses.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
                
        result = evaluate(model, val_loader)  # Ensure evaluate handles CUDA as needed
        result['train_loss'] = torch.stack(train_losses).mean().item()        
        
        model.epoch_end(epoch, result)
        
        history.append(result)
    
    return model, history

    

def test_model(model: nn.Module, test_loader: DataLoader) -> float:
    """ Calculates model accuracy (percentage) on the <test_loader> Dataset
    
    Arguments:
        model : the input server model
        test_loader : DataLoader with the dataset to use for testing
    """
    
    criterion = nn.CrossEntropyLoss()

    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the device
    model.to(device)

    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():  # No need to track gradients in evaluation

        for batch in test_loader:
            inputs, labels = [item.to(device) for item in batch]
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Calculate total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Average test loss over all examples
    test_loss = test_loss / len(test_loader.dataset)

    # Calculate accuracy percentage
    accuracy = (correct / total) * 100

    return accuracy
