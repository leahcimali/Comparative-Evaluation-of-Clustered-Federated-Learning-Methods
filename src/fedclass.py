class Client:

    # Define the client class
    def __init__(self, client_id, data):
        
        import numpy as np
        
        self.id = client_id
        self.data = data 
        self.model = None  
        self.cluster_id = np.nan 
        self.heterogeneity_class = np.nan
        self.accuracy = 0

    def to_dict(self):
        return {
            'id': self.id, 
            'cluster_id': self.cluster_id,
            'heterogeneity_class': self.heterogeneity_class,
            'accuracy': self.accuracy
        }

class Server:
    # Define the server class
    def __init__(self,model,num_clusters=None):
        self.model = model # initialize central server model
        self.num_clusters = num_clusters # number of clusters defined 
        self.clusters_models = {} # Dictionary of clusters models 
        