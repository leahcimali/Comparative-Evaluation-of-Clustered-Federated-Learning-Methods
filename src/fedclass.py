class Client:
    # Define the client class
    def __init__(self, client_id, data):
        self.id = client_id
        self.data = data 
        self.model = None  # Initialize local model
        self.cluster_id = None # Initialize cluster ID


class Server:
    # Define the server class
    def __init__(self,model,num_clusters=None):
        self.model = model # initialize central server model
        self.num_clusters = num_clusters # number of clusters defined 
        self.clusters_models = {} # Dictionary of clusters models 
        