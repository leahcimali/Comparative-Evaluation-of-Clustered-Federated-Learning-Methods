
#from src.utils_data import my_data_loader, createLoaders


import torch
import copy
import copy

# STANDARD FL 

def send_server_model_to_client(client_list, my_server):
    # As title 
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

def model_avg(client_list):
    # Create a new model with the weight average of clients' weights
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


# FOR SERVER-SIDE CFL 
def model_weight_matrix(list_clients):
    import numpy as np
    import pandas as pd
    
    """
    Create a weight matrix DataFrame using the weights of local federated models
    Parameters
    ----------
    list_clients: List of Clients with respective models
         all the federated system models

    Returns
    -------
    pd.DataFrame
        DataFrame with weights of each model as rows
    """
    model_dict = {client.id : client.model for client in list_clients}
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
    import pandas as pd
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


def k_means_clustering(client_list,num_clusters): 
    import pickle
    weight_matrix = model_weight_matrix(client_list)
    clusters_identities = k_means_cluster_id(weight_matrix, num_clusters)
    for client in client_list : 
        setattr(client, 'cluster_id',clusters_identities[client.id])
        

        
def fed_training_plan_one_shot_k_means(model_server, list_clients, row_exp):

    from src.utils_training import train_central
    
    lr = 0.01

    for _ in range(0, row_exp['federated_rounds']):

        send_server_model_to_client(list_clients, model_server)

        for client in list_clients:

            client.model = train_central(client.model, client.data_loader['train'], client.data_loader['test'], row_exp)

        fedavg(model_server, list_clients)

    setattr(model_server,'num_clusters', row_exp['num_clusters'])
    
    model_server.clusters_models= {cluster_id: copy.deepcopy(model_server.model) for cluster_id in range(row_exp['num_clusters'])}
    
    k_means_clustering(list_clients, row_exp['num_clusters'])

    
    # Rounds after clustering
    send_server_model_to_client(list_clients, model_server)
    
    for client in list_clients:
    
        client.model = train_central(client.model, client.data_loader['train'], client.data_loader['test'], row_exp)
        
        fedavg(model_server, list_clients)
    


# FOR CLIENT-SIDE CFL 

def init_server_cluster(my_server,client_list, num_clusters, seed = 0):
    # Set client to random cluster for first round. Used for client_side CFL 
    from src.models import SimpleLinear
    from src.utils_training import loss_calculation
    import numpy as np
    my_server.num_clusters = num_clusters
    torch.manual_seed(seed)
    my_server.clusters_models = {cluster_id: SimpleLinear(h1=200) for cluster_id in range(num_clusters)} 
    for client in client_list:
        client.cluster_id = np.random.randint(0,num_clusters)
        
            
def set_client_cluster(my_server,client_list,num_clusters=4,epochs=10):
    # Use the loss to calculate the cluster membership for client-side CFL
    from src.utils_training import loss_calculation
    import numpy as np
    for client in client_list:
        print('Calculating all cluster model loss on local data for client {} !'.format(client.id))
        cluster_losses = []
        for cluster_id in range(num_clusters):
            cluster_loss = loss_calculation(my_server.clusters_models[cluster_id], client.data_loader['train'])
            cluster_losses.append(cluster_loss)
        index_of_min_loss = np.argmin(cluster_losses)
        print('Best loss with cluster model {}'.format(index_of_min_loss))
        client.model = copy.deepcopy(my_server.clusters_models[index_of_min_loss])
        client.cluster_id = index_of_min_loss

        
def fed_training_plan_client_side(model_server, list_clients, row_exp, init_cluster=True):

    from src.utils_training import train_central

    if init_cluster == True : 
        init_server_cluster(model_server, list_clients, row_exp['num_clusters'], row_exp['seed'])
    
    for _ in range(row_exp['federated_rounds']):

        set_client_cluster(model_server, list_clients, row_exp['num_clusters'], row_exp['federated_local_epochs'])
        
        for client in list_clients:

            client.model = train_central(client.model, client.data_loader['train'], client.data_loader['test'], row_exp)

        fedavg(model_server, list_clients)

    
    