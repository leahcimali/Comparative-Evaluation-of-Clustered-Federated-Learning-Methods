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
from src.utils_data import setup_experiment_rotation, centralize_data
from src.utils_training import train_model, test_model
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
        rounds = config['federated_rounds']
        epochs = config['federated_local_epochs']
        output = config['output']

        print(config)  # Print current configuration
        
        # Your experiment code here...
        # Make sure to use 'config' to access the parameters for the current experiment


    torch.manual_seed(seed)




    from src.utils_fed import fed_training_plan_client_side
    model = SimpleLinear()
    my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model,number_of_cluster=4)
    number_of_clusters = 4
    fed_training_plan_client_side(my_server,client_list, rounds ,epochs,number_of_clusters=number_of_clusters,lr=lr,initcluster=True)
    with open("./{}.txt".format(output), 'w') as f:
        with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):
            print('server num clusters : ', my_server.num_clusters)
            for cluster_id in range(4):
                client_list_cluster = [client for client in client_list if client.cluster_id == cluster_id]
                print('Federated Model Accuracy for cluster', cluster_id)                       
            print(config)
            report_CFL(my_server,client_list)