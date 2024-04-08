from os import makedirs
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import contextlib
from pathlib import Path
import torch

print(torch.__version__)
import src.config
from src.models import MnistNN, SimpleLinear 
from src.fedclass import Client, Server
from src.utils_data import setup_experiment_rotation, setup_experiment_labelswap, setup_experiment_quantity_skew, centralize_data
from src.utils_training import train_model, test_model
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
from src.metrics import report_CFL

# Load config from JSON file
lr = 0.1
with open('serverconfig.json') as config_file:
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
        cfl_before_cluster_rounds = config['cfl_before_cluster_rounds']
        cfl_after_cluster_rounds = config['cfl_after_cluster_rounds']
        cfl_local_epochs = config['cfl_local_epochs']
        number_of_clusters = config['number_of_clusters']
        output = config['output']
        heterogeneity = config["heterogeneity"]
        
        print(config)  # Print current configuration
        
        # Your experiment code here...
        # Make sure to use 'config' to access the parameters for the current experiment


    torch.manual_seed(seed)
    model = SimpleLinear()
    from src.utils_fed import fed_training_plan_on_shot_k_means
    if heterogeneity == 'concept_shift_on_features':
        my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model)   
    elif heterogeneity == 'concept_shift_on_labels':
        my_server, client_list  = setup_experiment_labelswap(number_of_clients,number_of_samples_of_each_labels_by_clients,model,number_of_cluster=number_of_clusters,seed=seed)
    elif heterogeneity == 'quantity_skew':
        my_server, client_list  = setup_experiment_quantity_skew(model,number_of_client= number_of_clients, number_of_max_samples= number_of_samples_of_each_labels_by_clients,seed = seed)
        
    fed_training_plan_on_shot_k_means(my_server,client_list, cfl_before_cluster_rounds , cfl_after_cluster_rounds , cfl_local_epochs,lr= lr, number_of_clusters=number_of_clusters)
    
    with open("./{}.txt".format(output), 'w') as f:
        with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):    
            print('server num clusters : ', my_server.num_clusters)
            for cluster_id in range(number_of_clusters):
                client_list_cluster = [client for client in client_list if client.cluster_id == cluster_id]
                _, test_loader = centralize_data(client_list_cluster)
                print('Federated Model Accuracy for cluster', cluster_id)
                print("Accuracy: {:.2f}%".format(test_model(my_server.clusters_models[cluster_id], test_loader)*100))
            print(config)
            #report_CFL(my_server,client_list)
