
#from src.utils_data import my_data_loader, createLoaders


import pandas as pd
import torch
from src.fedclass import Client, Server
from src.utils_training import train_model
from src.utils_data import data_distribution,data_preparation
from sklearn.metrics import  mean_squared_error
import copy

import copy

def send_server_model_to_client(client_list, my_server): 
    if my_server.num_clusters == None: 
        for client in client_list:
            setattr(client, 'model', copy.deepcopy(my_server.model))
            
    else:
        for client in client_list:
            if client.cluster_id is None:
                setattr(client, 'model', copy.deepcopy(my_server.model))
            else:
                setattr(client, 'model', copy.deepcopy(my_server.clusters_models[client.cluster_id]))

import copy
'''
def fedavg(my_server, client_list):

    """
    Implementation of the FedAvg Algorithm as described in:
    McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data. 
    Artificial intelligence and statistics. PMLR, 2017
    TO DO: Need to add client samples ponderation in case of non-uniform clients data samples
    """
    if my_server.num_clusters == None:
        state_dict = client_list[0].model.state_dict()
        for name, param in client_list[0].model.named_parameters():
            for i in range(1, len(client_list)):
                state_dict[name] += client_list[i].model.state_dict()[name]
            state_dict[name] /= len(client_list)        
        my_server.model.load_state_dict(state_dict)
    else:
        for cluster_id in range(my_server.num_clusters):
            print('FedAVG on cluster {}!'.format(cluster_id))
            # Filter clients belonging to the current cluster
            cluster_client_list = [client for client in client_list if client.cluster_id == cluster_id]
            print('Number of clients in cluster {}: {}'.format(cluster_id, len(cluster_client_list)))
            
            # Initialize a dictionary to store parameter sums
            param_sums = {name: torch.zeros_like(param.data) for name, param in cluster_client_list[0].model.named_parameters()}
            
            # Sum up parameters from all clients
            total_clients = len(cluster_client_list)
            for client in cluster_client_list:
                for name, param in client.model.named_parameters():
                    param_sums[name] += param.data
            
            # Average parameters
            averaged_params = {name: param_sum / total_clients for name, param_sum in param_sums.items()}
            
            # Create a new model and load averaged parameters
            new_model = copy.deepcopy(cluster_client_list[0].model)  # Assuming all models have the same architecture
            for name, param in new_model.named_parameters():
                param.data = averaged_params[name]
            
            # Store the new model in the server's cluster models
            my_server.clusters_models[cluster_id] = new_model
'''           

def model_avg(client_list):
    new_model = copy.deepcopy(client_list[0].model)
        
    # Initialize a variable to store the total size of all local training datasets
    total_data_size = sum(len(client.data_loader['train'].dataset) for client in client_list)
    
    # Iterate over the parameters of the new model
    for name, param in new_model.named_parameters():
        # Initialize the weighted averaged parameter with zeros
        weighted_avg_param = torch.zeros_like(param)
        
        # Accumulate the parameters across all clients, ponderated by local data size
        for client in client_list:
            # Calculate the weight based on the local data size
            data_size = len(client.data_loader['train'].dataset)
            weight = data_size / total_data_size
            
            # Add the weighted parameters of the current client
            weighted_avg_param += client.model.state_dict()[name] * weight
        
        # Assign the weighted averaged parameter to the new model
        param.data = weighted_avg_param
        
        return new_model
    
def fedavg(my_server,client_list):
    """
    Perform a weighted average of model parameters across clients,
    where the weight is determined by the size of each client's
    local training dataset. Return a new model with the averaged parameters.

    Args:
        client_list (list): List of clients, each containing a PyTorch model and a data loader.

    Returns:
        torch.nn.Module: A new PyTorch model with the weighted averaged parameters.
    """
    if my_server.num_clusters == None:
        # Initialize a new model
        my_server.model = model_avg(client_list)
    else : 
         for cluster_id in range(my_server.num_clusters):
            print('FedAVG on cluster {}!'.format(cluster_id))
            # Filter clients belonging to the current cluster
            cluster_client_list = [client for client in client_list if client.cluster_id == cluster_id]
            if len(cluster_client_list)>0 :  
                print('Number of clients in cluster {}: {}'.format(cluster_id, len(cluster_client_list)))
                my_server.clusters_models[cluster_id] = model_avg(cluster_client_list)
            else : 
                print('No client in cluster ', cluster_id) 

def fed_training_plan(my_server, client_list,rounds=3, epoch=200,lr =0.001):
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
    for round in range(0, rounds):
        print('Init round {} :'.format(round+1))
        print('Sending Server model to clients !')
        send_server_model_to_client(client_list, my_server)
        for client in client_list:
            print('Training local model for client {} !'.format(client.id))
            client.model = train_model(client.model, client.data_loader['train'],client.data_loader['test'],epoch,lr)

        print('Aggregating local models with FedAVG !')
        fedavg(my_server, client_list)
        print('Communication round {} completed !'.format(round+1))


def model_weight_matrix(client_list):
    import numpy as np
    import pandas as pd
    
    """
    Create a weight matrix DataFrame using the weights of local federated models
    Parameters
    ----------
    model_dict: Dictionary
        Contains all the federated system models

    Returns
    -------
    pd.DataFrame
        DataFrame with weights of each model as rows
    """
    model_dict = {client.id : client.model for client in client_list}
    # Collect the shapes of the model parameters
    shapes = [param.data.numpy().shape for param in next(iter(model_dict.values())).parameters()]

    # Create an empty NumPy matrix to store the weights
    weight_matrix_np = np.empty((len(model_dict), sum(np.prod(shape) for shape in shapes)))

    # Iterate through the keys of model_dict
    for idx, (model_num, model) in enumerate(model_dict.items()):
        # Extract model weights and flatten
        model_weights = np.concatenate([param.data.numpy().flatten() for param in model.parameters()])
        weight_matrix_np[idx, :] = model_weights

    # Convert the NumPy matrix to a DataFrame
    weight_matrix = pd.DataFrame(weight_matrix_np, columns=[f'w_{i+1}' for i in range(weight_matrix_np.shape[1])])

    return weight_matrix

def k_means_cluster_id(weight_matrix, k): 
    
    from sklearn.cluster import KMeans
    
    """
    Define cluster identites using k-means
    ----------
    weight_matrix: DataFrame
        Weight matrix of all federated models
    k: Interger
        K-means parameter
        
    Returns
    -------
    pd.DataFrame
        Pandas Serie with cluster identity of each model
    """
    
    
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the model to the standardized data
    kmeans.fit(weight_matrix)

    # Add a new column to the DataFrame indicating the cluster assignment
    weight_matrix['cluster'] = kmeans.labels_

    clusters_identities = weight_matrix['cluster']
    return clusters_identities


def k_means_clustering(client_list,number_of_clusters): 
    weight_matrix = model_weight_matrix(client_list)
    clusters_identities = k_means_cluster_id(weight_matrix, number_of_clusters)
    for client in client_list : 
        setattr(client, 'cluster_id',clusters_identities[client.id])

        
def fed_training_plan_on_shot_k_means(my_server, client_list,rounds_before_clustering=3, round_after_clustering=3, epoch=10, number_of_clusters = 4,lr = 0.001):
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
    for round in range(0, rounds_before_clustering):
        print('Init round {} :'.format(round+1))
        print('Sending Server model to clients !')
        send_server_model_to_client(client_list, my_server)
        for client in client_list:
            print('Training local model for client {} !'.format(client.id))
            client.model = train_model(client.model, client.data_loader['train'], client.data_loader['test'],epoch, learning_rate=lr)
        print('Aggregating local models with FedAVG !')
        fedavg(my_server, client_list)
        print('Communication round {} completed !'.format(round+1))
    print('Starting clustering')
    setattr(my_server,'num_clusters',number_of_clusters)
    my_server.clusters_models= {cluster_id: copy.deepcopy(my_server.model) for cluster_id in range(number_of_clusters)}
    k_means_clustering(client_list,number_of_clusters)
    for rounds in range(rounds_before_clustering, rounds_before_clustering + round_after_clustering) :
        print('Init round {} :'.format(round+1))
        print('Sending Server models to clients !')
        send_server_model_to_client(client_list, my_server)
        for client in client_list:
            print('Training local model for client {} !'.format(client.id))
            client.model = train_model(client.model, client.data_loader['train'], client.data_loader['test'],epoch,learning_rate=lr)
        print('Aggregating local models with FedAVG !')
        fedavg(my_server, client_list)
        print('Communication round {} completed !'.format(round+1))
    '''
    if need of saving models 
    for cluster_id in range(4): 
        torch.save(my_server.clusters_models[cluster_id].state_dict(), 'model_{}.pth'.format(cluster_id))
    '''    
def calculate_cluster_id(my_server,client_list, number_of_clusters=4):
    from src.models import MnistNN, SimpleLinear
    from src.utils_training import loss_calculation
    import numpy as np
    for client in client_list:
        #print('Calculating all cluster model loss on local data for client {} !'.format(client.id))
        cluster_losses = []
        for cluster_id in range(number_of_clusters):
            cluster_loss = loss_calculation(my_server.clusters_models[cluster_id], client.data_loader['train'])
            cluster_losses.append(cluster_loss)
        index_of_min_loss = np.argmin(cluster_losses)
        client.cluster_id = index_of_min_loss
    listofcluster = [client.cluster_id for client in client_list]
    return listofcluster

def init_server_cluster(my_server,client_list, number_of_clusters=4, seed = 42):
    # Set client to random cluster for first round
    from src.models import MnistNN, SimpleLinear
    from src.utils_training import loss_calculation
    import numpy as np
    my_server.num_clusters = number_of_clusters
    torch.manual_seed(seed)
    my_server.clusters_models = {cluster_id: SimpleLinear(h1=200) for cluster_id in range(number_of_clusters)} 
    for client in client_list:
        client.cluster_id = np.random.randint(0,number_of_clusters)
        
    '''
        cluster_losses = []
        for cluster_id in range(number_of_clusters):
            cluster_loss = loss_calculation(my_server.clusters_models[cluster_id], client.data_loader['train'])
            cluster_losses.append(cluster_loss)
        index_of_min_loss = np.argmin(cluster_losses)
        client.cluster_id = index_of_min_loss
    listofcluster = [client.cluster_id for client in client_list]
    # testing if all cluster are choosen
    for cluster_id in range(number_of_clusters):
        while cluster_id not in listofcluster :
            print(listofcluster)
            print('restarting cluster centroid for cluster_id {}'.format(cluster_id))
            new_seed = np.random.randint(0,1000000)   
            torch.manual_seed(new_seed)
            my_server.clusters_models[cluster_id] = SimpleLinear(200)
            listofcluster = calculate_cluster_id(my_server,client_list, number_of_clusters=4)
            print('update', listofcluster)
    '''
            
def set_client_cluster(my_server,client_list,number_of_clusters=4,epochs=10):
    from src.utils_training import loss_calculation
    import numpy as np
    for client in client_list:
        print('Calculating all cluster model loss on local data for client {} !'.format(client.id))
        cluster_losses = []
        for cluster_id in range(number_of_clusters):
            cluster_loss = loss_calculation(my_server.clusters_models[cluster_id], client.data_loader['train'])
            cluster_losses.append(cluster_loss)
        index_of_min_loss = np.argmin(cluster_losses)
        print('Best loss with cluster model {}'.format(index_of_min_loss))
        client.model = copy.deepcopy(my_server.clusters_models[index_of_min_loss])
        client.cluster_id = index_of_min_loss

        
def fed_training_plan_client_side(my_server, client_list,rounds=3, epoch=10, number_of_clusters = 4,lr = 0.001,initcluster = True):
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

    model_path : str.
        Define the path where to save the models

    """
    from src.utils_training import train_model
    import numpy as np
    if initcluster == True : 
        init_server_cluster(my_server,client_list, number_of_clusters=number_of_clusters, seed = 42)
    for round in range(0, rounds):
        print('Init round {} :'.format(round+1))
        set_client_cluster(my_server, client_list, number_of_clusters=number_of_clusters, epochs=10)
        for client in client_list:
            print('Training local model for client {} !'.format(client.id))
            client.model = train_model(client.model, client.data_loader['train'], client.data_loader['test'],epoch,learning_rate=lr)
        print('Aggregating local models with FedAVG !')
        fedavg(my_server, client_list)
        print('Communication round {} completed !'.format(round+1))
    
    