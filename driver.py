
def main_driver():

    from src.utils_data import setup_experiment 
    import pandas as pd

    df_experiments = pd.read_csv("exp_configs.csv")
  
    for i, row_exp in df_experiments.iterrows():
               
        model_server, list_clients, list_heterogeneities = setup_experiment(row_exp)
                    
        launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, i)

    return          
            


def launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, i):
        
        from src.utils_fed import fed_training_plan_client_side, fed_training_plan_one_shot_k_means

        if row_exp['exp_type'] == "central":
              
            run_benchmark(list_clients, row_exp, i, training_type="centralized")
            run_benchmark(list_clients, row_exp, i, main_model=model_server, training_type="federated")
                        
            for heterogeneity_class in list_heterogeneities:
                
                list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
                
                run_benchmark(list_clients_filtered, row_exp, i, training_type="personalized_centralized")
                
                run_benchmark(list_clients_filtered, row_exp, i, main_model=model_server,
                                training_type="federated_personalized")
                
        elif row_exp['exp_type'] == "client":

            fed_training_plan_client_side(model_server, list_clients, row_exp)
            
        elif row_exp['exp_type'] == "server":

            fed_training_plan_one_shot_k_means(model_server, list_clients, row_exp)
            
        else:
        
            raise Exception(f"Unrecognized experiement type {row_exp['exp_type']}. Please check config file and try again.")
        
        return




def run_benchmark(list_clients, row_exp, i, main_model = None, training_type="centralized"):
    
    from src.models import SimpleLinear 
    from src.utils_training import train_benchmark, test_benchmark
    
    if not main_model:
         main_model = SimpleLinear()
    
    model_server, test_loader = train_benchmark(list_clients, row_exp, i, main_model, training_type)

    test_benchmark(model_server, list_clients, test_loader)

    #save_results(model_server, list_clients, row_exp)

    return



    
def save_results(model_server, list_clients, row_exp):
    
    from src.utils_fed import model_weight_matrix
    import torch

    if row_exp['exp_type'] == "client" or "server":
        for cluster_id in range(row_exp['num_clusters']): 
            torch.save(model_server.clusters_models[cluster_id].state_dict(), f"./results/{row_exp['output']}_{row_exp['exp_type']}_model_cluster_{cluster_id}.pth")


        weight_matrix = model_weight_matrix(list_clients)
        client_cluster = [client.cluster_id for client in list_clients]
        weight_matrix['cluster'] = client_cluster
        
        weight_matrix.to_pickle(f'./results/{row_exp['output']}_client_weights.pkl')  
    
        for cluster_id in range(row_exp['num_clusters']): 
            torch.save(model_server.clusters_models[cluster_id].state_dict(), f"./results/{row_exp['output']}_{row_exp['exp_type']}_model_cluster_{cluster_id}.pth")

    return 



if __name__ == "__main__":
    main_driver()