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
from src.models import MnistNN 
from src.fedclass import Client, Server
from src.utils_data import setup_experiment_rotation, centralize_data
from src.utils_training import train_model, test_model
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
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
    with open("./{}.txt".format(output), 'w') as f:
        with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):
            # CONFIG VARIABLE 
            model = MnistNN()
            my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model)   
            train_loader, test_loader = centralize_data(client_list)
            my_server

            #centralized_model = train_model(copy.deepcopy(model), train_loader, test_loader,centralized_model_epochs ) 

            #from src.utils_training import train_model

            #from src.utils_fed import fed_training_plan, send_server_model_to_client
            #fed_training_plan(my_server, client_list, federated_rounds, federated_local_epochs)
            #for client in client_list : 
            #    print('For client {} test data we have :'.format(client.id))
            #    print("Accuracy: {:.2f}%".format(test_model(my_server.model, client.data_loader['test'])*100))
            #test_fed  = test_model(my_server.model, test_loader)
            #test_central = test_model(centralized_model, test_loader)


            from src.utils_fed import fed_training_plan_client_side
            my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model)
            my_server.clusters_models = {cluster_id: MnistNN() for cluster_id in range(4)}   
            fed_training_plan_client_side(my_server,client_list, rounds ,epochs)
            print('server num clusters : ', my_server.num_clusters)
            for cluster_id in range(4):
                client_list_cluster = [client for client in client_list if client.cluster_id == cluster_id]
                _, test_loader = centralize_data(client_list_cluster)
                print('Federated Model Accuracy for cluster', cluster_id)
                print("Accuracy: {:.2f}%".format(test_model(my_server.clusters_models[cluster_id], test_loader)*100))            
            print(config)
