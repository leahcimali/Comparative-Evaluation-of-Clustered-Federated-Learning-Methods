import os
import pytest

if os.getenv('_PYTEST_RAISE', "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value

def test_run_cfl_benchmark():

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pickle 

    from src.utils_data import setup_experiment    
    from src.utils_training import run_benchmark

    file_path = Path("tests/refs/benchmark-global-federated_fashion-mnist_features-distribution-skew_8_100_3_5_5_42.csv")
    
    with open (file_path, "r") as fp:
        
        keys = ['exp_type', 'dataset' , 'heterogeneity_type' , 'num_clients',
                'num_samples_by_label' , 'num_clusters', 'centralized_epochs',
                'federated_rounds', 'seed']
        

        parameters = file_path.stem.split('_')

        row_exp = dict(
            zip(keys,
                parameters[:3] + [int(x) for x in  parameters[3:]])
            )

        model_server, list_clients = setup_experiment(row_exp)

    df_results = run_benchmark(list_clients, row_exp, model_server, ['pers-centralized'])

    assert all(np.isclose(df_results['accuracy'], pd.read_csv(file_path)['accuracy'], rtol=0.01))


def test_run_cfl_client_side():

    return


def test_run_cfl_server_side():

    return 

if __name__ == "__main__":
    test_run_cfl_benchmark()