

class Params():

    def __init__(self, config_file_path):

        import json
'''
        # Load the configuration file using the provided path
        with open(config_file_path) as f:
            config = json.load(f)
            self.seed = config['seed']
            self.number_of_clients = config['number_of_clients']
            self.number_of_samples_of_each_labels_by_clients = config['number_of_samples_of_each_labels_by_clients']
            self.centralized_model_epochs = config['centralized_model_epochs']
            self.federated_rounds = config['federated_rounds']
            self.federated_local_epochs = config['federated_local_epochs']
            self.cfl_before_cluster_rounds = config['cfl_before_cluster_rounds']
            self.cfl_after_cluster_rounds = config['cfl_after_cluster_rounds']
            self.cfl_local_epochs = config['cfl_local_epochs']
            self.output = config['output']
'''
            
class Tee:
    # Class so that the stdout output is saved in a txt file and print at the same time
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()  # Flush the buffer to ensure immediate writing

    def flush(self):
        for file in self.files:
            file.flush()

