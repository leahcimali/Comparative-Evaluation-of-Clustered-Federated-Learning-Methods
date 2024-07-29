
def main_driver():

    from pathlib import Path
    from inputimeout import inputimeout
    import pandas as pd

    from src.utils_logging import cprint, setup_logging
    from src.utils_data import setup_experiment, get_uid 

    setup_logging()

    df_experiments = pd.read_csv("exp_configs.csv")
    df_results = pd.DataFrame(columns=['exp_type','training_type', 'heterogeneity_type', 'model_type', 'heterogeneity_class', 'accuracy'])

    for _, row_exp in df_experiments.iterrows():
               
        model_server, list_clients, list_heterogeneities, output_name = setup_experiment(row_exp)
        
        hash_outputname = get_uid(output_name)

        pathlist = Path("results").rglob('*.json')
    
        for file_name in pathlist:
    
            if get_uid(str(file_name.stem)) == hash_outputname:

                cprint(f"Experiment {str(file_name.stem)} already executed in with results in \n {output_name}.json", lvl="warning")   

                try:
                    if inputimeout('Experiment already ran. Do You Want To Continue [y/n]? (defaults to [y] in 5 seconds): ', timeout=5) != 'y':
                        return
                
                except Exception:
                    continue
    
        launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, df_results, output_name)

    return          
            


def launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, df_results, output_name):
        
        from src.utils_training import run_cfl_client_side, run_cfl_server_side
        from src.utils_training import run_benchmark
        from src.utils_logging import cprint


        if row_exp['exp_type'] == "benchmark":
            
            cprint(f"Launching benchmark experiment with parameters:\n{':'.join(row_exp.to_string().replace('\n', '/').split())}", lvl="info")   

            df_results = run_benchmark(list_clients, row_exp, output_name,
                                       df_results = df_results, 
                                       training_type="centralized")
            
            df_results = run_benchmark(list_clients, row_exp, output_name,
                                       df_results = df_results,
                                       main_model=model_server,
                                       training_type="federated")
                        
            for heterogeneity_class in list_heterogeneities:
                
                list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]
                
                df_results = run_benchmark(list_clients_filtered, row_exp, 
                                           output_name,
                                           df_results= df_results,
                                           training_type="personalized_centralized")
                
                df_results = run_benchmark(list_clients_filtered, row_exp,
                                           output_name,
                                           df_results= df_results,
                                           main_model=model_server,
                                           training_type="personalized_federated",
                                           write_results=True)
                
        elif row_exp['exp_type'] == "client":
            
            cprint(f"Launching client-side experiment with parameters:\n {':'.join(row_exp.to_string().replace('\n', '/').split())}", lvl="info")

            run_cfl_client_side(model_server, list_clients, row_exp, output_name)
            
        elif row_exp['exp_type'] == "server":

            cprint(f"Launching server-side experiment with parameters:\n {':'.join(row_exp.to_string().replace('\n', '/').split())}", lvl="info")

            run_cfl_server_side(model_server, list_clients, row_exp, output_name)
            
        else:
        
            raise Exception(f"Unrecognized experiement type {row_exp['exp_type']}. Please check config file and try again.")
        
        return




if __name__ == "__main__":
    main_driver()