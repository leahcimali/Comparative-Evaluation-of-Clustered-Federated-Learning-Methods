
def send_server_model_to_client(client_list, my_server):
    """
    Function to copy server model to clients in standard FL
    """

    import copy

    for client in client_list:
        setattr(client, 'model', copy.deepcopy(my_server.model))

    return


def send_cluster_models_to_clients(client_list , my_server):
    """
    Function to distribute cluster models to clients based on attribute client.cluster_id
    """
    import copy

    for client in client_list:
            if client.cluster_id is None:
                setattr(client, 'model', copy.deepcopy(my_server.model))
            else:
                setattr(client, 'model', copy.deepcopy(my_server.clusters_models[client.cluster_id]))
    return 


def model_avg(client_list):

    import copy
    import torch

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
          
            # Filter clients belonging to the current cluster
            
            cluster_client_list = [client for client in client_list if client.cluster_id == cluster_id]
            
            if len(cluster_client_list)>0 :  
          
                my_server.clusters_models[cluster_id] = model_avg(cluster_client_list)
            

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
    for idx, (_, model) in enumerate(model_dict.items()):
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


def k_means_clustering(client_list,num_clusters): 
    import pickle
    weight_matrix = model_weight_matrix(client_list)
    clusters_identities = k_means_cluster_id(weight_matrix, num_clusters)
    for client in client_list : 
        setattr(client, 'cluster_id',clusters_identities[client.id])
        


# FOR CLIENT-SIDE CFL 

def init_server_cluster(my_server, client_list, row_exp, p_expert_opinion=None):
    
    """
    Assign clients to initial clusters using a given distribution or completely at random. 
    """
    
    from src.models import SimpleLinear
    import numpy as np
    import copy
    import torch

    torch.manual_seed(row_exp['seed'])

    list_heterogeneities = list(set([c.heterogeneity_class for c in client_list]))

    if not p_expert_opinion:

        p_expert_opinion = 1 / row_exp['num_clusters']
        
    p_rest = (1 - p_expert_opinion) / (row_exp['num_clusters'] - 1)

    my_server.num_clusters = row_exp['num_clusters']
    
    my_server.clusters_models = {cluster_id: SimpleLinear(h1=200) for cluster_id in range(row_exp['num_clusters'])}
    
    
    for client in client_list:
    
        probs = [p_rest if x != list_heterogeneities.index(client.heterogeneity_class) % row_exp['num_clusters']
                        else p_expert_opinion for x in range(row_exp['num_clusters'])] 

        client.cluster_id = np.random.choice(range(row_exp['num_clusters']), p = probs)

        client.model = copy.deepcopy(my_server.clusters_models[client.cluster_id])
    


def set_client_cluster(my_server, client_list, row_exp):
    """
    Use the loss to calculate the cluster membership for client-side CFL
    """
    
    from src.utils_training import loss_calculation
    import numpy as np
    import copy
    
    for client in client_list:
        
        cluster_losses = []
        
        for cluster_id in range(row_exp['num_clusters']):
        
            cluster_loss = loss_calculation(my_server.clusters_models[cluster_id], client.data_loader['train'], row_exp)
        
            cluster_losses.append(cluster_loss)
        
        index_of_min_loss = np.argmin(cluster_losses)
        
        #print(f"client {client.id} with heterogeneity {client.heterogeneity_class} cluster losses:", cluster_losses)

        client.model = copy.deepcopy(my_server.clusters_models[index_of_min_loss])
    
        client.cluster_id = index_of_min_loss
    