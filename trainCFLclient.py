from os import makedirs
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import contextlib
from pathlib import Path
import torch
import pandas as pd

print(torch.__version__)
import src.config
from src.models import MnistNN, SimpleLinear
from src.fedclass import Client, Server
from src.utils_data import setup_experiment_rotation, setup_experiment_labelswap, setup_experiment_quantity_skew, setup_experiment_labels_skew, setup_experiment_features_skew
from src.utils_training import train_model, test_model
from src.utils_fed import model_weight_matrix
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
from src.metrics import report_CFL
lr = 0.1
# Load config from JSON file
with open('clientconfig.json') as config_file:
    config_data = json.load(config_file)

experiments = config_data['experiments']
for exp_id, experiment in enumerate(experiments):
    print(f"Running experiment {exp_id + 1}...")
    # Instead of directly iterating over experiment[key], we will check if it's a list or a single value
    for param_set in zip(*[experiment[key] if isinstance(experiment[key], list) else [experiment[key]] for key in experiment]):
        config = dict(zip(experiment.keys(), param_set))  
        seed = config['seed']
        number_of_clients = config['number_of_clients']
        number_of_samples_of_each_labels_by_clients = config['number_of_samples_of_each_labels_by_clients']
        number_of_clusters = config['number_of_clusters']
        rounds = config['federated_rounds']
        epochs = config['federated_local_epochs']
        output = config['output']
        heterogeneity = config["heterogeneity"]
        config['type'] = 'client_side'

        print(config)  # Print current configuration
        
        # Your experiment code here...
        # Make sure to use 'config' to access the parameters for the current experiment


    torch.manual_seed(seed)




    from src.utils_fed import fed_training_plan_client_side
    model = SimpleLinear()
    if heterogeneity == 'concept_shift_on_features':
        my_server, client_list  = setup_experiment_rotation(model,number_of_clients,number_of_samples_of_each_labels_by_clients)   
    elif heterogeneity == 'concept_shift_on_labels':
        my_server, client_list  = setup_experiment_labelswap(model,number_of_clients,number_of_samples_of_each_labels_by_clients,number_of_cluster=number_of_clusters,seed=seed)
    elif heterogeneity == 'labels_distribution_skew':
        my_server, client_list = setup_experiment_labels_skew(model,number_of_clients=number_of_clients, number_of_samples_by_clients=number_of_samples_of_each_labels_by_clients,seed =42)
    elif heterogeneity == 'labels_distribution_skew_balancing':
        my_server, client_list = setup_experiment_labels_skew(model,number_of_clients=number_of_clients, number_of_samples_by_clients=number_of_samples_of_each_labels_by_clients, 
                                                                skewlist=[[0,1,2,3,4],[5,6,7,8,9],[0,2,4,6,8],[1,3,5,7,9]], 
                                                                ratiolist = [[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1]],seed = 42)
    elif heterogeneity == 'labels_distribution_skew_upsampled':
        my_server, client_list = setup_experiment_labels_skew(model,number_of_clients=number_of_clients, number_of_samples_by_clients=number_of_samples_of_each_labels_by_clients,
                                                                skewlist=[[0,3,4,5,6,7,8,9],[0,1,2,5,6,7,8,9],[0,1,2,3,4,7,8,9],[0,1,2,3,4,5,6,9]], 
                                                                ratiolist = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]],seed = 42)
    elif heterogeneity == 'features_distribution_skew':
        my_server, client_list = setup_experiment_features_skew(model, number_of_clients, number_of_samples_of_each_labels_by_clients, seed)
    elif heterogeneity == 'quantity_skew':
        my_server, client_list  = setup_experiment_quantity_skew(model,number_of_client= number_of_clients, number_of_max_samples= number_of_samples_of_each_labels_by_clients,skewlist=[1, 0.5, 0.25, 0.1, 0.05],seed = seed)
        
    else : 
        print('Error no heterogeneity type defined')
    print(heterogeneity)
    fed_training_plan_client_side(my_server,client_list, rounds ,epochs,number_of_clusters=number_of_clusters,lr=lr,initcluster=True)
    for cluster_id in range(number_of_clusters): 
        torch.save(my_server.clusters_models[cluster_id].state_dict(), f'./results/{output}_client_model_cluster_{cluster_id}.pth')
    weight_matrix = model_weight_matrix(client_list)
    client_cluster = [client.cluster_id for client in client_list]
    weight_matrix['cluster'] = client_cluster
    import pickle
    weight_matrix.to_pickle(f'./results/{output}_client_weights.pkl')    
    with open("./results/{}.txt".format(output), 'w') as f:
        with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):
            report_CFL(my_server,client_list,config)