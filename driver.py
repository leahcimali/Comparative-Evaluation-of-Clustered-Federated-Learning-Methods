
import click

@click.command()
@click.option('--exp_type', help="The experiment type to run")
@click.option('--heterogeneity_type', help="The data heterogeneity to test (or dataset)")
@click.option('--num_clients', type=int)
@click.option('--num_samples_by_label', type=int)
@click.option('--num_clusters', type=int)
@click.option('--centralized_epochs', type=int, default=50)
@click.option('--federated_rounds', type=int, default=5)
@click.option('--federated_local_epochs', type=int, default=20)
@click.option('--seed', type=int, default=42)



def main_driver(exp_type, heterogeneity_type, num_clients, num_samples_by_label, num_clusters, centralized_epochs, federated_rounds, federated_local_epochs, seed):

    from pathlib import Path
    import pandas as pd

    from src.utils_logging import cprint, setup_logging
    from src.utils_data import setup_experiment, get_uid 

    setup_logging()

    df_results = pd.DataFrame(columns=['exp_type','training_type', 'heterogeneity_type', 'model_type', 'heterogeneity_class', 'accuracy'])

    row_exp = pd.Series({"exp_type": exp_type, "heterogeneity_type": heterogeneity_type, "num_clients": num_clients,
               "num_samples_by_label": num_samples_by_label, "num_clusters": num_clusters, "centralized_epochs": centralized_epochs,
               "federated_rounds": federated_rounds, "federated_local_epochs": federated_local_epochs, "seed": seed})
    

    output_name =  row_exp.to_string(header=False, index=False, name=False).replace(' ', "").replace('\n','_')

    hash_outputname = get_uid(output_name)

    pathlist = Path("results").rglob('*.json')

    for file_name in pathlist:

        if get_uid(str(file_name.stem)) == hash_outputname:

            cprint(f"Experiment {str(file_name.stem)} already executed in with results in \n {output_name}.json", lvl="warning")   
        
            return 
    try:
        
        model_server, list_clients, list_heterogeneities = setup_experiment(row_exp)
    
    except Exception as e:

        cprint(f"Could not run experiment with parameters {output_name}. Exception {e}")

        return 
    
    launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, df_results, output_name)

    return          
            


def launch_experiment(model_server, list_clients, list_heterogeneities, row_exp, df_results, output_name):
        
        from src.utils_training import run_cfl_client_side, run_cfl_server_side
        from src.utils_training import run_benchmark
        from src.utils_logging import cprint

        str_row_exp = ':'.join(row_exp.to_string().replace('\n', '/').split())

        if row_exp['exp_type'] == "benchmark":
            
            cprint(f"Launching benchmark experiment with parameters:\n{str_row_exp}", lvl="info")   

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
            
            cprint(f"Launching client-side experiment with parameters:\n {str_row_exp}", lvl="info")

            run_cfl_client_side(model_server, list_clients, row_exp, output_name)
            
        elif row_exp['exp_type'] == "server":

            cprint(f"Launching server-side experiment with parameters:\n {str_row_exp}", lvl="info")

            run_cfl_server_side(model_server, list_clients, row_exp, output_name)
            
        else:
        
            raise Exception(f"Unrecognized experiement type {row_exp['exp_type']}. Please check config file and try again.")
        
        return




if __name__ == "__main__":
    main_driver()
