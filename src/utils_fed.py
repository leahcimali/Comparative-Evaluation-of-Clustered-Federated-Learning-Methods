
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
            print('FedAVG on cluster {} ! '.format(cluster_id))
            # Define the client list of the current cluster
            cluster_client_list = [client for client in client_list if client.cluster_id == cluster_id]
            # Do fedavg 
            state_dict = cluster_client_list[0].model.state_dict()
            for name, param in cluster_client_list[0].model.named_parameters():
                for i in range(1, len(cluster_client_list)):
                    state_dict[name] += cluster_client_list[i].model.state_dict()[name]
                state_dict[name] /= len(cluster_client_list)      
            new_model = copy.deepcopy(my_server.model)
            new_model.load_state_dict(state_dict)
            my_server.clusters_models['cluster_id'] = new_model


def fed_training_plan(my_server, client_list,rounds=3, epoch=200 ):
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
            client.model = train_model(client.model, client.data_loader['train'],client.data_loader['test'],epoch)

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



def print_layer(model):
    first_layer_params = list(model.parameters())[2]
    # Extract the weight tensor from the parameters
    first_layer_weights = first_layer_params.data
    # Now you can use first_layer_weights as needed
    print(first_layer_weights)

def k_means_clustering(client_list,number_of_clusters): 
    weight_matrix = model_weight_matrix(client_list)
    clusters_identities = k_means_cluster_id(weight_matrix, number_of_clusters)
    for client in client_list : 
        setattr(client, 'cluster_id',clusters_identities[client.id])

def init_server_cluster(my_server,number_of_clusters=4, seed = 42):
    from src.models import MnistNN
    torch.manual_seed(seed)
    my_server.clusters_models = {cluster_id: MnistNN() for cluster_id in range(4)}   
 
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
        print('Best loss with cluster model {}'.format( index_of_min_loss))
        client.model = copy.deepcopy(my_server.clusters_models[index_of_min_loss])
        print('test print client layer')
        print_layer(client.model)
        print('test print cluster layer')
        print_layer(my_server.clusters_models[index_of_min_loss])
        
def client_side_clustering(my_server,client_list,number_of_clusters=4,epochs=10): 
    import numpy as np
    from src.utils_training import loss_calculation
    for client in client_list:
        print('Training all cluster model on local data for client {} !'.format(client.id))
        model_and_loss = [train_model(copy.deepcopy(my_server.clusters_models[cluster_id]), client.data_loader['train'], client.data_loader['test'],epochs, learning_rate=0.001) for cluster_id in range(4)]
        losses = [loss_calculation(model, client.data_loader['train']) for model, _ in model_and_loss]
        index_of_min_loss = np.argmin(losses)
        print('Best loss with cluster model {}'.format( index_of_min_loss))
        clientmodel, _ = model_and_loss[index_of_min_loss]
        client.model = copy.deepcopy(clientmodel)
        client.cluster_id = index_of_min_loss 
        
def fed_training_plan_on_shot_k_means(my_server, client_list,rounds_before_clustering=3, round_after_clustering=3, epoch=10, number_of_clusters = 4):
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
            client.model = train_model(client.model, client.data_loader['train'], client.data_loader['test'],epoch)
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
            client.model = train_model(client.model, client.data_loader['train'], client.data_loader['test'],epoch)
        print('Aggregating local models with FedAVG !')
        fedavg(my_server, client_list)
        print('Communication round {} completed !'.format(round+1))
        
def fed_training_plan_client_side(my_server, client_list,rounds=3, epoch=10, number_of_clusters = 4):
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
    for round in range(0, rounds):
        print('Init round {} :'.format(round+1))
        client_side_clustering(my_server,client_list,number_of_clusters,epoch)
        print('Aggregating local models with FedAVG !')
        fedavg(my_server, client_list)
        print('Communication round {} completed !'.format(round+1))
        