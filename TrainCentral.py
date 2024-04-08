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

# Load config from JSON file
lr = 0.1
with open('config.json') as config_file:
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
        centralized_model_epochs = config['centralized_model_epochs']
        federated_rounds = config['federated_rounds']
        federated_local_epochs = config['federated_local_epochs']
        cfl_before_cluster_rounds = config['cfl_before_cluster_rounds']
        cfl_after_cluster_rounds = config['cfl_after_cluster_rounds']
        cfl_local_epochs = config['cfl_local_epochs']
        output = config['output']
        
        print(config)  # Print current configuration
        
        # Your experiment code here...
        # Make sure to use 'config' to access the parameters for the current experiment


    torch.manual_seed(seed)
    with open("./{}.txt".format(output), 'w') as f:
        with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):
            model = SimpleLinear()
            my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model)   
            for rotation in [0,90,180,270]:
                rotated_client= [client for client in client_list if client.rotation == rotation] 
                train_loader, test_loader = centralize_data(rotated_client)
                personalized_centralized_model = train_model(copy.deepcopy(model), train_loader, test_loader,centralized_model_epochs,learning_rate= lr ) 
                test_central = test_model(personalized_centralized_model, test_loader)
                print('personalized centralized model Accuracy with rotation ', rotation)
                print("Accuracy: {:.2f}%".format(test_central*100))
            