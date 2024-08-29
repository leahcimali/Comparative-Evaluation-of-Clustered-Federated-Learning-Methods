from src.fedclass import Server
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

def send_server_model_to_client(list_clients : list, my_server : Server) -> None:
    
    """ Function to copy the Server model to client attributes in a FL protocol

    Args:
        list_clients : List of Client objects on which to set the parameter `model'
        my_server : Server object with the model to copy
    """

    import copy

    for client in list_clients:
        setattr(client, 'model', copy.deepcopy(my_server.model))

    return


def send_cluster_models_to_clients(list_clients : list , my_server : Server) -> None:
    """ Function to copy Server modelm to clients based on attribute client.cluster_id

    Args: 
        list_clients : List of Clients to update
        my_server : Server from which to fetch models
    """

    import copy

    for client in list_clients:
            if client.cluster_id is None:
                setattr(client, 'model', copy.deepcopy(my_server.model))
            else:
                setattr(client, 'model', copy.deepcopy(my_server.clusters_models[client.cluster_id]))
    return 


def model_avg(list_clients : list) -> nn.Module:
    
    """  Utility function for the fed_avg function which creates a new model
         with weights set to the weighted average of 
    
    Args:
        list_clients : List of Client whose models we want to use to perform the weighted average

    Returns:
        New model with weights equal to the weighted average of those in the input Clients list

    """
    
    import copy
    import torch

    new_model = copy.deepcopy(list_clients[0].model)

    total_data_size = sum(len(client.data_loader['train'].dataset) for client in list_clients)

    for name, param in new_model.named_parameters():

        weighted_avg_param = torch.zeros_like(param)
        
        for client in list_clients:

            data_size = len(client.data_loader['train'].dataset)

            weight = data_size / total_data_size
            
            weighted_avg_param += client.model.state_dict()[name] * weight

        param.data = weighted_avg_param #TODO: make more explicit
        
    return new_model
    
    
def fedavg(my_server : Server, list_clients : list) -> None:
    """
    Implementation of the (Clustered) federated aggregation algorithm with one model per cluster. 
    The code modifies the cluster models `my_server.cluster_models[i]'

    
    Args:
        my_server : Server model which contains the cluster models

        list_clients: List of clients, each containing a PyTorch model and a data loader.

    """
    if my_server.num_clusters == None:
        # Initialize a new model
        my_server.model = model_avg(list_clients)
    
    else : 
         
         for cluster_id in range(my_server.num_clusters):
          
            # Filter clients belonging to the current cluster
            
            cluster_clients_list = [client for client in list_clients if client.cluster_id == cluster_id]
            
            if len(cluster_clients_list)>0 :  
          
                my_server.clusters_models[cluster_id] = model_avg(cluster_clients_list)
    return


def model_weight_matrix(list_clients : list) -> pd.DataFrame:
   
    """ Create a weight matrix DataFrame using the weights of local federated models for use in the server-side CFL 

    Args :

    list_clients: List of Clients with respective models
         
    Returns
        DataFrame with weights of each model as rows
    """

    import numpy as np
    import pandas as pd
    

    model_dict = {client.id : client.model for client in list_clients}

    shapes = [param.data.numpy().shape for param in next(iter(model_dict.values())).parameters()]

    weight_matrix_np = np.empty((len(model_dict), sum(np.prod(shape) for shape in shapes)))

    for idx, (_, model) in enumerate(model_dict.items()):

        model_weights = np.concatenate([param.data.numpy().flatten() for param in model.parameters()])

        weight_matrix_np[idx, :] = model_weights

    weight_matrix = pd.DataFrame(weight_matrix_np, columns=[f'w_{i+1}' for i in range(weight_matrix_np.shape[1])])

    return weight_matrix


def k_means_cluster_id(weight_matrix : pd.DataFrame, k : int, seed : int) -> pd.Series: 
    
    """ Define cluster identites using k-means

    Args : 
        weight_matrix:  Weight matrix of all federated models
        k: K-means parameter
        seed : Random seed to allow reproducibility
        
    Returns:
        Pandas Serie with cluster identity of each model
    """
    
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=k, random_state=seed)

    kmeans.fit(weight_matrix)

    weight_matrix['cluster'] = kmeans.labels_

    clusters_identities = weight_matrix['cluster']

    return clusters_identities


def k_means_clustering(list_clients : list, num_clusters : int, seed : int) -> None:

    """ Performs a k-mean clustering and sets the cluser_id attribute to clients based on the result
    
    Args:
        list_clients : List of Clients on which to perform clustering
        num_clusters : Parameter to set the number of clusters needed
        seed : Random seed to allow reproducibility
        
    """ 

    weight_matrix = model_weight_matrix(list_clients)

    clusters_identities = k_means_cluster_id(weight_matrix, num_clusters, seed)

    for client in list_clients : 

        setattr(client, 'cluster_id',clusters_identities[client.id])
    
    return  



def init_server_cluster(my_server : Server, list_clients : list, row_exp : dict, p_expert_opinion : float = 0) -> None:
    
    """ Function to initialize cluster membership for client-side CFL (sets param cluster id) 
    using a given distribution or completely at random. 
    
    Args:
        my_server : Server model containing one model per cluster

        list_clients : List of Clients  whose model we want to initialize

        row_exp : Dictionary containing the different global experiement parameters

        p_expert_opintion : Parameter to avoid completly random assignment if neeed (default to 0)
    """
    
    from src.models import SimpleLinear
    import numpy as np
    import copy

    np.random.seed(row_exp['seed'])

    list_heterogeneities = list(dict.fromkeys([client.heterogeneity_class for client in list_clients]))

    if not p_expert_opinion or p_expert_opinion == 0:

        p_expert_opinion = 1 / row_exp['num_clusters']
        
    p_rest = (1 - p_expert_opinion) / (row_exp['num_clusters'] - 1)

    my_server.num_clusters = row_exp['num_clusters']
    
    my_server.clusters_models = {cluster_id: SimpleLinear(h1=200) for cluster_id in range(row_exp['num_clusters'])}
    
    
    for client in list_clients:
    
        probs = [p_rest if x != list_heterogeneities.index(client.heterogeneity_class) % row_exp['num_clusters']
                        else p_expert_opinion for x in range(row_exp['num_clusters'])] 

        client.cluster_id = np.random.choice(range(row_exp['num_clusters']), p = probs)

        client.model = copy.deepcopy(my_server.clusters_models[client.cluster_id])
    
    return 


def loss_calculation(model : nn.modules, train_loader : DataLoader) -> float:

    """ Utility function to calculate average_loss across all samples <train_loader>

    Args:

        model : the input server model
        
        train_loader : DataLoader with the dataset to use for loss calculation
    """ 
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()  

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for inputs, targets in train_loader:

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples

    return average_loss




def set_client_cluster(my_server : Server, list_clients : list, row_exp : dict) -> None:
    """ Function to calculate cluster membership for client-side CFL (sets param cluster id)
    
     Args:
        my_server : Server model containing one model per cluster

        list_clients : List of Clients  whose model we want to initialize

        row_exp : Dictionary containing the different global experiement parameters
    """

    import numpy as np
    import copy
    
    for client in list_clients:
        
        cluster_losses = []
        
        for cluster_id in range(row_exp['num_clusters']):
        
            cluster_loss = loss_calculation(my_server.clusters_models[cluster_id], client.data_loader['train'])
        
            cluster_losses.append(cluster_loss)
        
        index_of_min_loss = np.argmin(cluster_losses)
        
        #print(f"client {client.id} with heterogeneity {client.heterogeneity_class} cluster losses:", cluster_losses)

        client.model = copy.deepcopy(my_server.clusters_models[index_of_min_loss])
    
        client.cluster_id = index_of_min_loss
    