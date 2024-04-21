from os import makedirs
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import contextlib
from pathlib import Path
import torch
import torch.optim as optim

print(torch.__version__)
import src.config
from src.models import MnistNN, SimpleLinear 
from src.fedclass import Client, Server
from src.utils_data import setup_experiment_rotation, setup_experiment_labels_skew,setup_experiment_labelswap,setup_experiment_quantity_skew, centralize_data, setup_experiment_features_skew
from src.utils_training import train_model, test_model
from src.utils_fed import fed_training_plan
import numpy as np
import matplotlib.pyplot as plt
import copy
import json

# Load config from JSON file
lr = 0.1
with open('centralconfig.json') as config_file:
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
        number_of_clusters = config["number_of_clusters"]
        output = config['output']
        heterogeneity = config['heterogeneity']
        config['type'] = 'central'
        results = config
        # Print current configuration
        
        # Your experiment code here...
        # Make sure to use 'config' to access the parameters for the current experiment


    torch.manual_seed(seed)
    with open("./results/{}.txt".format(output), 'w') as f:
        with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):
            model = SimpleLinear()
            if heterogeneity == 'concept_shift_on_features':
                my_server, client_list  = setup_experiment_rotation(number_of_clients,number_of_samples_of_each_labels_by_clients,model)
            elif heterogeneity == 'concept_shift_on_labels':
                my_server, client_list  = setup_experiment_labelswap(number_of_clients,number_of_samples_of_each_labels_by_clients,model,number_of_cluster=number_of_clusters,seed=seed)
            elif heterogeneity == 'labels_distribution_skew':
                my_server, client_list = setup_experiment_labels_skew(model,number_of_clients=number_of_clients, number_of_samples_by_clients=number_of_samples_of_each_labels_by_clients,seed =42)   
            elif heterogeneity == 'features_distribution_skew':
                lr = 0.001
                number_of_users = 9
                my_server, client_list = setup_experiment_features_skew(model,number_of_clients,number_of_users,number_of_samples_of_each_labels_by_clients,seed=seed)
            elif heterogeneity == 'quantity_skew':
                my_server, client_list  = setup_experiment_quantity_skew(model,number_of_client= number_of_clients, number_of_max_samples= number_of_samples_of_each_labels_by_clients,skewlist=[1, 0.5, 0.25, 0.1, 0.05],seed = seed)
            else : 
                print('Error no heterogeneity type defined')
            print(heterogeneity) 
            # Centralized results
            print("Centralized results")    
            centralized_model = SimpleLinear()
            train_loader, test_loader = centralize_data(client_list)
            train_model(centralized_model, train_loader, test_loader,centralized_model_epochs,learning_rate= lr)
            test_central = test_model(centralized_model, test_loader)
            print('centralized model accuracy')
            print("Accuracy: {:.2f}%".format(test_central*100))
            results['central'] = test_central*100
            # Centralized Personalized results
            heterogeneity_types = set(client.heterogeneity for client in client_list)
            model = SimpleLinear()
            for heterogeneity_type in heterogeneity_types :
                client_of_type = [client for client in client_list if client.heterogeneity == heterogeneity_type]
                train_loader, test_loader = centralize_data(client_of_type)
                personalized_centralized_model = train_model(copy.deepcopy(model), train_loader, test_loader,centralized_model_epochs,learning_rate= lr ) 
                test_central = test_model(personalized_centralized_model, test_loader)
                print('personalized centralized model Accuracy with rotation ', heterogeneity_type)
                print("Accuracy: {:.2f}%".format(test_central*100))
                results[f'personalized central ({heterogeneity_type})'] = test_central*100

            # federated
            fed_training_plan(my_server, client_list,federated_rounds, federated_local_epochs,lr)
            train_loader, test_loader = centralize_data(client_list)
            test_federatedmodel = test_model(my_server.model, test_loader)
            print('Federated model accuracy')
            print("Accuracy: {:.2f}%".format(test_federatedmodel*100))
            results['federated'] = test_federatedmodel*100
            clients_accs =[]
            for client in client_list : 
                acc = test_model(my_server.model, client.data_loader['test'])*100
                clients_accs.append(acc)
            results['federated std'] = np.std(clients_accs)
    
            print(results)
            
            with open('./results/{}.json'.format(config['output']), 'w') as json_file:
                json.dump(results, json_file, indent=4)