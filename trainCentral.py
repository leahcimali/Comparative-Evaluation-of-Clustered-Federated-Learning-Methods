import sys

import torch
import torch
import logging

print(torch.__version__)

from src.models import SimpleLinear 
from src.utils_data import setup_experiment, centralize_data
from src.utils_training import train_model, test_model
from src.utils_fed import fed_training_plan
import numpy as np
import copy

logging.basicConfig(filename="output.log", level=logging.INFO)
# Load config from JSON file

def main_driver():

    import pandas as pd

    df_experiments = pd.read_csv("exp_configs.csv")
  
    for i, row_exp in df_experiments.iterrows():
        
        torch.manual_seed(row_exp['seed'])

        with open("./results/{}.txt".format(row_exp['output']), 'w+') as f:
                
                model_server, list_clients  = setup_experiment(SimpleLinear(), row_exp)
                heterogeneity_types = set(client.heterogeneity_class for client in list_clients)
                
                run_experiment_central(list_clients, row_exp, df_experiments, i, n_epochs=10)
                run_experiment_federated(model_server, list_clients, row_exp, df_experiments, i , heterogeneity_class=None, training_type="federated")
                            
                for heterogeneity_class in heterogeneity_types :
                                    
                    run_experiment_central([client for client in list_clients if client.heterogeneity_class == heterogeneity_class],
                                                row_exp,
                                                df_experiments,
                                                i, heterogeneity_class,
                                                n_epochs=10,
                                                training_type = "personalized_centalized")
                    
                    run_experiment_federated(model_server, list_clients, row_exp, df_experiments, i , heterogeneity_class, training_type="federated_personalized")
        
    return df_experiments            
            


def run_experiment_central(list_clients, row_exp, df_experiments, i, heterogeneity_class=None, n_epochs = 10, training_type = "centralized"):
     
    model_central =  SimpleLinear()
    
    train_loader, test_loader = centralize_data(list_clients)

    model_trained = train_model(model_central, train_loader, test_loader, n_epochs,learning_rate= 0.1) 
    results_accuracy = test_model(model_trained, test_loader = test_loader)

    logging.info(f'personalized centralized model Accuracy with {heterogeneity_class}')
    logging.info("Accuracy: {:.2f}%".format(results_accuracy*100))

    torch.save(model_trained.state_dict(), f"./results/{row_exp['output']}_{training_type}_{heterogeneity_class}_.pth")
    df_experiments[i, f'result_{training_type}_{heterogeneity_class}'] = results_accuracy*100

    return df_experiments
    


def run_experiment_federated(model_server, list_clients, row_exp, df_experiments, i , heterogeneity_class=None, training_type="federated"):
    
    # Federated Learning by heterogeneity 
    logging.info(f'Federated Learning {heterogeneity_class}')

    model_server_het = copy.deepcopy(model_server)

    fed_training_plan(model_server_het, list_clients, row_exp['federated_rounds'], row_exp['federated_local_epochs'], lr=0.1)
    _, test_loader = centralize_data(list_clients)

    test_federatedmodel = test_model(model_server_het.model, test_loader)

    logging.info(f'Federated model {training_type} accuracy')
    logging.info("Accuracy: {:.2f}%".format(test_federatedmodel*100))

    df_experiments[i, f'result_federated_{training_type}']  = test_federatedmodel*100

    clients_accs =[]
    
    for client in list_clients : 
        acc = test_model(model_server.model, client.data_loader['test'])*100
        clients_accs.append(acc)

    df_experiments[i, f'result_federated_{training_type}_{heterogeneity_class}_std'] =  np.std(clients_accs)

    torch.save(model_server_het.model.state_dict(), f"./results/{row_exp['output']}_Federated_model_{training_type}_{heterogeneity_class}.pth")
    return df_experiments





if __name__ == "__main__":
     main_driver()