
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
with open('config.json') as config_file:
    config = json.load(config_file)
    
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

print(config)

torch.manual_seed(seed)
with open("./{}.txt".format(output), 'w') as f:
    with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):
        # CONFIG VARIABLE 
        model = MnistNN()
        my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model)   
        train_loader, test_loader = centralize_data(client_list)

        centralized_model = train_model(copy.deepcopy(model), train_loader, test_loader,centralized_model_epochs ) 

        from src.utils_training import train_model

        from src.utils_fed import fed_training_plan, send_server_model_to_client
        fed_training_plan(my_server, client_list, federated_rounds, federated_local_epochs)
        for client in client_list : 
            print('For client {} test data we have :'.format(client.id))
            print("Accuracy: {:.2f}%".format(test_model(my_server.model, client.data_loader['test'])*100))
        test_fed  = test_model(my_server.model, test_loader)
        test_central = test_model(centralized_model, test_loader)


        from src.utils_fed import fed_training_plan_on_shot_k_means
        my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model)   
        fed_training_plan_on_shot_k_means(my_server,client_list, cfl_before_cluster_rounds , cfl_after_cluster_rounds , cfl_local_epochs)
        print('server num clusters : ', my_server.num_clusters)
        print('Federated Model Accuracy')
        print("Accuracy: {:.2f}%".format(test_fed*100))
        print('centralized model Accuracy')
        print("Accuracy: {:.2f}%".format(test_central*100))
        for cluster_id in range(4):
            client_list_cluster = [client for client in client_list if client.cluster_id == cluster_id]
            _, test_loader = centralize_data(client_list_cluster)
            print('Federated Model Accuracy for cluster', cluster_id)
            print("Accuracy: {:.2f}%".format(test_model(my_server.clusters_models[cluster_id], test_loader)*100))
        
        print(config)
