import sys

import torch
import torch

print(torch.__version__)

from src.models import SimpleLinear 
from src.utils_data import setup_experiment, centralize_data
from src.utils_training import train_model, test_model
from src.utils_fed import fed_training_plan
import numpy as np
import copy


# Load config from JSON file

def run_experiments():

    import pandas as pd

    df_experiments = pd.read_csv("exp_configs.csv")
    with open("out.log", "w") as sys.stdout:
         
        for i, row_exp in df_experiments.iterrows():
            
            torch.manual_seed(row_exp['seed'])

            with open("./results/{}.txt".format(row_exp['output']), 'w+') as f:
                    
                    model_server, list_clients  = setup_experiment(SimpleLinear(), row_exp)
                    heterogeneity_types = set(client.heterogeneity_class for client in list_clients)
                    
                    run_central(list_clients, row_exp, df_experiments, i)
                    run_federated(model_server, list_clients, row_exp, df_experiments, i)
                                
                    for heterogeneity_class in heterogeneity_types :
                                        
                        run_central_personalized_model(SimpleLinear(), list_clients, heterogeneity_class, row_exp, df_experiments,i)
                        run_federated_personalized_model(model_server, list_clients, heterogeneity_class, row_exp, df_experiments, i)
        
        return df_experiments            
            


def run_central_personalized_model(model, list_clients, heterogeneity_class, row_exp, df_experiments, i):
        

        n_epochs_centralized = 2
        output_name = row_exp['output']
        # Centralized by heterogeneity
        client_of_class = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
        train_loader, test_loader = centralize_data(client_of_class)

        model_centralized_personalized = train_model(copy.deepcopy(model), train_loader, test_loader, n_epochs_centralized,learning_rate= 0.1) 
        test_central_personalized = test_model(model_centralized_personalized, test_loader)

        print('personalized centralized model Accuracy with rotation ', heterogeneity_class)
        print("Accuracy: {:.2f}%".format(test_central_personalized*100))

        torch.save(model_centralized_personalized.state_dict(), f'./results/{output_name}_personalized_centralized_model_heterogeneity_{heterogeneity_class}_.pth')
        df_experiments[i, f'result_central_personalized_{heterogeneity_class}'] = test_central_personalized*100

        return  df_experiments


def run_federated_personalized_model(model_server, list_clients, heterogeneity_class, row_exp, df_experiments, i):

    # Federated Learning by heterogeneity 
    print('personalized Federated')
    
    model_server_het = copy.deepcopy(model_server)

    output_name = row_exp['output']

    fed_training_plan(model_server_het, list_clients, row_exp['federated_rounds'], row_exp['federated_local_epochs'], lr=0.1)
    _, test_loader = centralize_data(list_clients)

    test_federatedmodel = test_model(model_server_het.model, test_loader)
    
    print(f'Federated model heterogeneity {heterogeneity_class} accuracy')
    print("Accuracy: {:.2f}%".format(test_federatedmodel*100))

    df_experiments[i, f'result_federated_personalized_{heterogeneity_class}']  = test_federatedmodel*100
    torch.save(model_server_het.model.state_dict(), f'./results/{output_name}_Federated_model_heterogeneity_{heterogeneity_class}.pth')

    return df_experiments



def run_federated(model_server, list_clients, row_exp, df_experiments, i):
    
    output_name = row_exp['output']
    fed_training_plan(model_server, list_clients, row_exp['federated_rounds'], row_exp['federated_local_epochs'], lr=0.1)

    _, test_loader = centralize_data(list_clients)
    
    test_federatedmodel = test_model(model_server.model, test_loader)

    print('Federated model accuracy')
    print("Accuracy: {:.2f}%".format(test_federatedmodel*100))
    
    df_experiments[i, f'result_federated_no_personalization'] = test_federatedmodel*100
    
    clients_accs =[]
    
    for client in list_clients : 
        acc = test_model(model_server.model, client.data_loader['test'])*100
        clients_accs.append(acc)

    df_experiments[i, f'result_federated_std'] =  np.std(clients_accs)
    
    torch.save(model_server.model.state_dict(), f'./results/{output_name}_federated_model.pth')

    return df_experiments



def run_central(list_clients, row_exp, df_experiments, i):

    print("Centralized results") 
    output_name = row_exp['output']

    model_central =  SimpleLinear()
    train_loader, test_loader = centralize_data(list_clients)

    train_model(model_central, train_loader = train_loader, test_loader = test_loader, 
                num_epochs= 2, learning_rate= 0.1)
    
    test_central = test_model(model_central, test_loader = test_loader)


    print('centralized model accuracy')
    print("Accuracy: {:.2f}%".format(test_central*100))

    torch.save(model_central.state_dict(), f'./results/{output_name}_model_central.pth')
    df_experiments[i, 'result_central'] =  test_central*100 

    return df_experiments
    
    

if __name__ == "__main__":
     run_experiments()