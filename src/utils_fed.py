
from src.utils_data import my_data_loader, createLoaders


import pandas as pd
import torch

from src.utils_training import train_model
from sklearn.metrics import  mean_squared_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import copy

def setup_models(n_nodes, main_model):
    """
    Initialize dictionary with n_nodes models

    n_nodes : int
        Number of nodes in the federated learning framework

    main_model: torch.nn.Module
        Torch neural network model

    Returns
    -------
    Dictionary with "n_nodes" models intialized

    """
    return {k: copy.deepcopy(main_model) for k in range(n_nodes)}
    
def send_model(main_model, model_dict, number_of_nodes):

    """
    Send federated model to nodes

    Parameters
    ----------
    main_model : torch.nn.Module

    model_dict : Dict
        Dictionary with the models of the different nodes

    number_of_nodes : int
        number of nodes in federation
    """

    with torch.no_grad():
        for i in range(number_of_nodes):
            state_dict = copy.deepcopy(main_model.state_dict())
            model_dict[i].load_state_dict(state_dict)
    return model_dict

def fedavg(main_model, model_dict, number_of_nodes):

    """
    Implementation of the FedAvg Algorithm as described in:
    McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data. 
    Artificial intelligence and statistics. PMLR, 2017
    """

    state_dict = model_dict[0].state_dict()
    for name, param in model_dict[0].named_parameters():
        for i in range(1, number_of_nodes):
            state_dict[name]=  state_dict[name] + model_dict[i].state_dict()[name]
        state_dict[name] = state_dict[name]/number_of_nodes
    new_model = copy.deepcopy(main_model)
    new_model.load_state_dict(state_dict)
    return new_model


def fed_training_plan(main_model, data_dict, rounds=3, epoch=200, model_path='./'):
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
    from src.utils_training import testmodel
    nodes = len(data_dict)

    model_dict = setup_models(nodes, main_model)
    best_node_loss = [float('inf') for i in range(nodes)]
    node_loss = [0 for i in range(nodes)]
    best_model_round = [0 for i in range(nodes)]
    for round in range(1, rounds + 1):

        print('Init round {} :'.format(round))

        model_dict = send_model(main_model, model_dict, nodes)

        for node in range(nodes):
            print('Training node {} for round {}'.format(node, round))
            model_dict[node], _ , _ = train_model(model_dict[node], data_dict[node]['train'], data_dict[node]['val'], f'{model_path}local{node}_round{round}.pth', epoch, remove = True)

        print('FedAVG for round {}:'.format(round))

        main_model = fedavg(main_model, model_dict, nodes)
        for node in range(nodes):
            y_true, y_pred = testmodel(main_model, data_dict[node]["val"])
            node_loss[node] = mean_squared_error(y_true.flatten(), y_pred.flatten())
            print(f"Node {node} Validation loss :{node_loss[node]:.4f}")
            if node_loss[node] < best_node_loss[node]:
                best_node_loss[node] = node_loss[node]
                best_model_round[node] = round
                print(f'Better model founded at round {round} for node {node}!')
                torch.save(main_model.state_dict(), model_path / f'bestmodel_node{node}.pth')
        print('Done')
        # torch.save(main_model.state_dict(), f'{model_path}model_round_{round}.pth')
    print("FedAvg All Rounds Complete !")
    for node in range(nodes):
        print(f"Best model for node {node} at round {best_model_round[node]}.")
    # return main_model



def model_weight_matrix(model_dict):
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
        DataFrame with weights of each model as rows
    """
    
    from sklearn.cluster import KMeans
    
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the model to the standardized data
    kmeans.fit(weight_matrix)

    # Add a new column to the DataFrame indicating the cluster assignment
    weight_matrix['cluster'] = kmeans.labels_

    clusters_identities = weight_matrix['cluster']
    return clusters_identities