import os
import pytest

from pathlib import Path

if os.getenv('_PYTEST_RAISE', "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


def utils_extract_params(file_path: Path):
    """  Creates a dictionary row_exp with the parameters for the experiment given a well formated results file path
    """

    with open (file_path, "r") as fp:
        
        keys = ['exp_type', 'dataset', 'nn_model', 'heterogeneity_type' , 'num_clients',
                'num_samples_by_label' , 'num_clusters', 'centralized_epochs',
                'federated_rounds', 'seed']
        

        parameters = file_path.stem.split('_')

        row_exp = dict(
            zip(keys,
                parameters[:4] + [int(x) for x in  parameters[4:]])
            )
    
    return row_exp


def test_run_cfl_benchmark_oracle():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from src.utils_data import setup_experiment    
    from src.utils_training import run_benchmark

    file_path = Path("tests/refs/pers-centralized_fashion-mnist_linear_features-distribution-skew_8_100_3_5_5_42.csv")

    row_exp = utils_extract_params(file_path) 
   
    model_server, list_clients = setup_experiment(row_exp)

    df_results = run_benchmark(model_server, list_clients, row_exp)

    assert all(np.isclose(df_results['accuracy'], pd.read_csv(file_path)['accuracy'], rtol=0.01))


def test_run_cfl_benchmark_fl():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from src.utils_data import setup_experiment    
    from src.utils_training import run_benchmark

    file_path = Path("tests/refs/global-federated_fashion-mnist_linear_features-distribution-skew_8_100_3_5_5_42.csv")

    row_exp = utils_extract_params(file_path) 
   
    model_server, list_clients = setup_experiment(row_exp)

    df_results = run_benchmark(model_server, list_clients, row_exp)

    assert all(np.isclose(df_results['accuracy'], pd.read_csv(file_path)['accuracy'], rtol=0.01))


def test_run_cfl_client_side():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from src.utils_data import setup_experiment    
    from src.utils_training import run_cfl_client_side

    file_path = Path("tests/refs/client_fashion-mnist_linear_features-distribution-skew_8_100_3_5_5_42.csv")

    row_exp = utils_extract_params(file_path) 
   
    model_server, list_clients = setup_experiment(row_exp)

    df_results =  run_cfl_client_side(model_server, list_clients, row_exp)

    assert all(np.isclose(df_results['accuracy'], pd.read_csv(file_path)['accuracy'], rtol=0.01))


def test_run_cfl_server_side():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from src.utils_data import setup_experiment    
    from src.utils_training import run_cfl_server_side

    file_path = Path("tests/refs/server_fashion-mnist_linear_features-distribution-skew_8_100_3_5_5_42.csv")

    row_exp = utils_extract_params(file_path) 
   
    model_server, list_clients = setup_experiment(row_exp)

    df_results =  run_cfl_server_side(model_server, list_clients, row_exp)

    assert all(np.isclose(df_results['accuracy'], pd.read_csv(file_path)['accuracy'], rtol=0.01))


if __name__ == "__main__":
    test_run_cfl_client_side()
    test_run_cfl_server_side()
    test_run_cfl_benchmark_fl()
    test_run_cfl_benchmark_oracle()
    