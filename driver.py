
def main_driver():

    from src.utils_data import setup_experiment 
    import pandas as pd

    df_experiments = pd.read_csv("exp_configs.csv")
    df_results = pd.DataFrame(columns=['exp_type','training_type', 'heterogeneity_type', 'model_type', 'heterogeneity_class', 'accuracy'])

    for _, row_exp in df_experiments.iterrows():
               
        model_server, list_clients, list_heterogeneities = setup_experiment(row_exp)
                    
        launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, df_results)

    return          
            


def launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, df_results):
        
        from src.utils_training import run_cfl_client_side, run_cfl_server_side
        from src.utils_training import run_benchmark

        if row_exp['exp_type'] == "benchmark":
            
            print("Launching benchmark experiment with parameters:\n", row_exp)   

            df_results = run_benchmark(list_clients, row_exp, df_results = df_results, training_type="centralized")
            df_results = run_benchmark(list_clients, row_exp, df_results = df_results,
                                    main_model=model_server,
                                    training_type="federated")
                        
            for heterogeneity_class in list_heterogeneities:
                
                list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
                
                df_results = run_benchmark(list_clients_filtered, row_exp,
                              df_results= df_results,
                              training_type="personalized_centralized")
                
                df_results = run_benchmark(list_clients_filtered, row_exp,
                              df_results= df_results,
                              main_model=model_server,
                              training_type="personalized_federated",
                              write_results=True)
                
        elif row_exp['exp_type'] == "client":
            
            print("Launching client-side experiment with parameters:\n", row_exp)
            run_cfl_client_side(model_server, list_clients, row_exp)
            
        elif row_exp['exp_type'] == "server":

            print("Launching server-side experiment with parameters:\n", row_exp)
            run_cfl_server_side(model_server, list_clients, row_exp)
            
        else:
        
            raise Exception(f"Unrecognized experiement type {row_exp['exp_type']}. Please check config file and try again.")
        
        return


if __name__ == "__main__":
    main_driver()