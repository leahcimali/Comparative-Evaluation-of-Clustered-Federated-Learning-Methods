class Client:

    """ Client Object used in the Fedearated Learning protocol

    Attributes:
        client_id: unique client identifier
        data: client data in the form {'x': [], 'y' :[]) where x, and y are
            respectively the features and labels of the dataset 
    """

    def __init__(self, client_id: int, data: dict):
        
        """Initialize the Client object

        Args:
            id : int
                unique client identifier
            data : dict
                local data dict of the form {'x': [], 'y'[]}
            model : nn.Module
                The local nn model of the Client
            cluster_id : int 
                ID of the cluster the client belong to or None if not applicable
            heterogeneity_class: int
                The ID of heterogeneity class the client's data belong to or None if not applicable
            accuracy : float
                The current client's model's accuracy based on a test set
        """

        self.id = client_id
        self.data = data 
        self.model = None  
        self.cluster_id = None
        self.heterogeneity_class = None
        self.accuracy = 0

    def __eq__(self, value: object) -> bool:
        return (self.id == value.id and
                self.model == value.model and
                all((self.data['x'] == value.data['x']).flatten()) and
                all((self.data['y'] == value.data['y']).flatten()) and
                self.cluster_id == value.cluster_id and
                self.heterogeneity_class == value.heterogeneity_class) 
    
    
    def to_dict(self):

        """Return a dictionary with the attributes of the Client """ 

        return {
            'id': self.id, 
            'cluster_id': self.cluster_id,
            'heterogeneity_class': self.heterogeneity_class,
            'accuracy': self.accuracy
        }
    
    

class Server:

    """ Server Object used in the Fedearated Learning protocol

    Attributes:
        model: nn.Module
            The nn learing model the server is associated with
        num_clusters: int
            Number of clusters the server defines for a CFL protocol
    """

    def __init__(self,model,num_clusters: int=None):
        """Initialize a Server object with an empty dictionary of cluster_models

        Args:
        model: nn.Module
            The nn learing model the server is associated with
        num_clusters: int
            Number of clusters the server defines for a CFL protocol
        
        """

        self.model = model
        self.num_clusters = num_clusters  
        self.clusters_models = {}

    def __eq__(self, value: object) -> bool:
        return (str(self.model.state_dict()) == str(value.model.state_dict()) and
                self.num_clusters == value.num_clusters and
                self.clusters_models == value.clusters_models)