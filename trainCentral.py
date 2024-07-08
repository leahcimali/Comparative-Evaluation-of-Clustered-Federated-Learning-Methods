import sys

import torch
import torch
import logging

print(torch.__version__)

from src.models import SimpleLinear 
from src.utils_data import setup_experiment, centralize_data
from src.utils_training import train_model, test_model

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
                
                run_experiment(list_clients, row_exp, df_experiments, i, n_epochs=10, training_type="centralized")
                run_experiment(list_clients, row_exp, df_experiments, i, main_model=model_server, training_type="federated")
                            
                for heterogeneity_class in heterogeneity_types :
                    
                    run_experiment([client for client in list_clients if client.heterogeneity_class == heterogeneity_class],
                                   row_exp,
                                   df_experiments, 
                                   i, 
                                   heterogeneity_class, 
                                   n_epochs=10, 
                                   training_type="personalized_centralized")
                    
                    run_experiment(list_clients, row_exp, df_experiments, i , main_model=model_server,
                                   heterogeneity_class=heterogeneity_class, training_type="federated_personalized")
        
    return df_experiments            
            



def run_experiment(list_clients, row_exp, df_experiments, i, heterogeneity_class=None, main_model = SimpleLinear(), n_epochs = 10, training_type="centralized"):
     
    logging.info(f'Learning {training_type} {heterogeneity_class}:')
    clients_accs = []
    
    train_loader, test_loader = centralize_data(list_clients)
    
    if "federated" in training_type:
        model_server = copy.deepcopy(main_model)
        model_trained = train_model(model_server, None, None, list_clients=list_clients, num_epochs=row_exp['federated_local_epochs'])
        
    else:
        model_trained = train_model(main_model, train_loader, test_loader, n_epochs ,learning_rate= 0.1) 
        

    model_tested = test_model(model_trained, test_loader)

    logging.info("Accuracy: {:.2f}%".format(model_tested * 100))
    df_experiments[i, f'result_federated_{training_type}']  = model_tested *100

    for client in list_clients : 
        acc = test_model(model_trained, client.data_loader['test'])*100
        clients_accs.append(acc)

    df_experiments[i, f'result_{training_type}_{heterogeneity_class}_std'] =  np.std(clients_accs)

    torch.save(model_trained.state_dict(), f"./results/{row_exp['output']}__model_{training_type}_{heterogeneity_class}.pth")
    
    return df_experiments




if __name__ == "__main__":
     main_driver()