import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd

from src.models import SimpleLinear
from src.fedclass import Server

lr = 0.01


def run_cfl_server_side(model_server : Server, list_clients : list, row_exp : dict, output_name : str) -> pd.DataFrame:
    
    """ Driver function for server-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights (k-means).
    
     Args:
        
        model_server : The nn.Module to save

        list_clients : A list of Client Objects used as nodes in the FL protocol
            
        row_exp : The current experiment's global parameters

        output_name : the name of the results csv files saved in results/

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

    list_clients = add_clients_accuracies(model_server, list_clients, row_exp)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results 



def run_cfl_client_side(model_server : Server, list_clients : list, row_exp : dict, output_name : str, init_cluster=True) -> pd.DataFrame:

    """ Driver function for client-side cluster FL algorithm. The algorithm personalize training by clusters obtained
    from model weights (k-means).
    
     Args:
        
        model_server : The nn.Module to save

        list_clients : A list of Client Objects used as nodes in the FL protocol

        row_exp : The current experiment's global parameters

        output_name : the name of the results csv files saved in results/

        init_clusters : boolean indicating whether cluster assignement is done before initial training

    """

    from src.utils_fed import init_server_cluster, set_client_cluster, fedavg
    import torch

    torch.manual_seed(row_exp['seed'])

    if init_cluster == True : 
        
        init_server_cluster(model_server, list_clients, row_exp, p_expert_opinion=0.0)
    
    for _ in range(row_exp['federated_rounds']):

        for client in list_clients:

            client.model, _ = train_central(client.model, client.data_loader['train'], row_exp)

        fedavg(model_server, list_clients)
        
        set_client_cluster(model_server, list_clients, row_exp)

    list_clients = add_clients_accuracies(model_server, list_clients, row_exp)

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results
    

def run_benchmark(list_clients : list, row_exp : dict, main_model : nn.Module, list_exps : list) -> pd.DataFrame:

    """ Benchmark function to calculate baseline FL results and ``optimal'' personalization results if clusters are known in advance

    Args:
        
        list_clients : A list of Client Objects used as nodes in the FL protocol  

        row_exp : The current experiment's global parameters

        main_model : Type of Server model needed (default to SimpleLinear())

        list_exps : list containing the experiment names to run (current supported options: 'global-federated' and 'pers-centralized')

    """

    import pandas as pd 
    import torch
    import copy

    from src.utils_data import centralize_data

    list_heterogeneities = list(dict.fromkeys([client.heterogeneity_class for client in list_clients]))
    
    torch.manual_seed(row_exp['seed'])
    torch.use_deterministic_algorithms(True)

    for training_type in list_exps: 
        
        curr_model = main_model if training_type == 'global-federated' else SimpleLinear()

        match training_type:
        
            case 'pers-centralized':

                for heterogeneity_class in list_heterogeneities:
                    
                    list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]

                    train_loader, test_loader = centralize_data(list_clients_filtered)

                    model_trained, _ = train_central(curr_model, train_loader, row_exp) 

                    test_benchmark(model_trained, list_clients_filtered, test_loader, row_exp)
        
            case 'global-federated':
                    
                model_server = copy.deepcopy(curr_model)

                model_trained = train_federated(model_server, list_clients, row_exp, use_cluster_models = False)
            
                _, test_loader = centralize_data(list_clients)

                test_benchmark(model_trained.model, list_clients, test_loader, row_exp)

        df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    return df_results


def test_benchmark(model_trained : nn.Module, list_clients : list, test_loader : DataLoader, row_exp : dict):

    """ Tests <model_trained> on test_loader (global) dataset and sets the attribute accuracy on each Client 

        Args:
                
            list_clients : A list of Client Objects used as nodes in the FL protocol  

            row_exp : The current experiment's global parameters

            main_model : Type of Server model needed

            training_type : a value frmo ['global-federated', 'pers-centralized'] 

    """       
         
    from src.utils_training import test_model
    
    global_acc = test_model(model_trained, test_loader, row_exp) 
                     
    for client in list_clients : 
        
        #client_acc = test_model(model_trained, client.data_loader['test'])*100

        setattr(client, 'accuracy', global_acc)
    
    return global_acc


def train_federated(main_model, list_clients, row_exp, use_cluster_models = False):
    
    """Controler function to launch federated learning

    Args:
        
        main_model : Server model used in our experiment
        
        list_clients : A list of Client Objects used as nodes in the FL protocol  

        row_exp : The current experiment's global parameters

        use_cluster_models : Boolean to determine whether to use personalization by clustering
    """

    from src.utils_fed import send_server_model_to_client, send_cluster_models_to_clients, fedavg
    
    
    for i in range(0, row_exp['federated_rounds']):

        accs = []

        if use_cluster_models == False:
        
            send_server_model_to_client(list_clients, main_model)

        else:

            send_cluster_models_to_clients(list_clients, main_model)

        for client in list_clients:

            client.model, curr_acc = train_central(client.model, client.data_loader['train'], row_exp)

            accs.append(curr_acc)

        fedavg(main_model, list_clients)

    return main_model


def train_central(main_model, train_loader, row_exp):

    """ Main training function for centralized learning
    
    Args:

        main_model : Server model used in our experiment
        
        train_loader : DataLoader with the dataset to use for training

        row_exp : The current experiment's global parameters

    """
    criterion = nn.CrossEntropyLoss()
    
    optimizer=optim.SGD
    optimizer = optimizer(main_model.parameters(), lr=0.01) 
   
    main_model.train()
    
    for epoch in range(row_exp['centralized_epochs']):
          
        running_loss = total = correct = 0

        for inputs, labels in train_loader:

            optimizer.zero_grad()  

            outputs = main_model(inputs)  

            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward() 

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    main_model.eval() 

    return main_model, accuracy



def loss_calculation(model : nn.modules, train_loader : DataLoader, row_exp : dict) -> float:

    """ Utility function to calculate average_loss across all samples <train_loader>

    Args:

        model : the input server model
        
        train_loader : DataLoader with the dataset to use for loss calculation

        row_exp : The current experiment's global parameters
    """ 
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()  

    model.eval()

    total_loss = 0.0
    total_samples = 0

    #torch.manual_seed(row_exp['seed'])

    with torch.no_grad():

        for inputs, targets in train_loader:

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples

    return average_loss


def test_model(model : nn.Module, test_loader : DataLoader, row_exp : dict) -> float:

    """ Calcualtes model accuracy (percentage) on the <test_loader> Dataset
    
    Args:

        model : the input server model
        
        test_loader : DataLoader with the dataset to use for testing

        row_exp : The current experiment's global parameters
    """
    
    criterion = nn.CrossEntropyLoss()

    #torch.manual_seed(row_exp['seed'])
    
    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0


    with torch.no_grad():

        for inputs, labels in test_loader:

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
           
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)

    accuracy = (correct / total) * 100

    return accuracy


def add_clients_accuracies(model_server : nn.Module, list_clients : list, row_exp : dict) -> list:

    """
    Evaluates the cluster's models saved in <model_server> on the relevant list of clients and sets the attribute accuracy.

    Args:
        model_server : Server object which contains the cluster models

        list_clients : list of Client objects which belong to the different clusters

        row_exp : The current experiment's global parameters
    """

    for client in list_clients :

        acc = test_model(model_server.clusters_models[client.cluster_id], client.data_loader['test'], row_exp)
        
        setattr(client, 'accuracy', acc)

    return list_clients
