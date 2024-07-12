
def main_driver():

    from src.utils_data import setup_experiment 
    import pandas as pd

    df_experiments = pd.read_csv("exp_configs.csv")
  
    for i, row_exp in df_experiments.iterrows():
               
        model_server, list_clients, list_heterogeneities = setup_experiment(row_exp)
                    
        launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, i)

    return          
            


def launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, i):
        
        from src.utils_training import run_cfl_client_side, run_cfl_server_side
        from src.utils_training import run_benchmark

        if row_exp['exp_type'] == "benchmark":
              
            run_benchmark(list_clients, row_exp, i, training_type="centralized")
            run_benchmark(list_clients, row_exp, i, main_model=model_server, training_type="federated")
                        
            for heterogeneity_class in list_heterogeneities:
                
                list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
                
                run_benchmark(list_clients_filtered, row_exp, i, training_type="personalized_centralized")
                
                run_benchmark(list_clients_filtered, row_exp, i, main_model=model_server,
                                training_type="federated_personalized")
                
        elif row_exp['exp_type'] == "client":

            run_cfl_client_side(model_server, list_clients, row_exp)
            
        elif row_exp['exp_type'] == "server":

            run_cfl_server_side(model_server, list_clients, row_exp)
            
        else:
        
            raise Exception(f"Unrecognized experiement type {row_exp['exp_type']}. Please check config file and try again.")
        
        return


if __name__ == "__main__":
    main_driver()