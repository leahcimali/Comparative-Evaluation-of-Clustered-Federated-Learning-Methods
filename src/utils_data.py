from src.fedclass import Client, Server
from torch.utils.data import DataLoader
from numpy import ndarray
from typing import Tuple

def shuffle_list(list_samples : int, seed : int) -> list: 
    
    """Function to shuffle the samples list

    Arguments:
        list_samples : A list of samples to shuffle
        seed : Randomization seed for reproducible results
    
    Returns:
        The shuffled list of samples 
    """

    import numpy as np

    np.random.seed(seed)

    shuffled_indices = np.arange(list_samples.shape[0])

    np.random.shuffle(shuffled_indices)

    shuffled_list = list_samples[shuffled_indices].copy()
    
    return shuffled_list


def create_label_dict(dataset : str, nn_model : str) -> dict:
    
    """Create a dictionary of dataset samples

    Arguments:
        dataset : The name of the dataset to use (e.g 'fashion-mnist', 'mnist', or 'cifar10')
        nn_model : the training model type ('linear' or 'convolutional') 

    Returns:
        label_dict : A dictionary of data of the form {'x': [], 'y': []}

    Raises:
        Error : if the dataset name is unrecognized
    """
    
    import sys
    import numpy as np
    import torchvision
   
    import torchvision.transforms as transforms

    if dataset == "fashion-mnist":
        fashion_mnist = torchvision.datasets.MNIST("datasets", download=True)
        (x_data, y_data) = fashion_mnist.data, fashion_mnist.targets
        x_data = x_data.unsqueeze(3)
    elif dataset == 'mnist':
        mnist = torchvision.datasets.MNIST("datasets", download=True)
        (x_data, y_data) = mnist.data, mnist.targets
        x_data = x_data.unsqueeze(3)
    elif dataset == "cifar10":
        cifar10 = torchvision.datasets.CIFAR10("datasets", download=True)
        (x_data, y_data) = cifar10.data, cifar10.targets
         
    elif dataset == 'kmnist':
        kmnist = torchvision.datasets.KMNIST("datasets", download=True)
        x_data = kmnist.data  # This gives you the images
        x_data = x_data.unsqueeze(3)
        y_data = kmnist.targets  # This gives you the labels  
         
    else:
        sys.exit("Unrecognized dataset. Please make sure you are using one of the following ['mnist', fashion-mnist', 'kmnist']")    

    label_dict = {}

    for label in range(10):
       
        label_indices = np.where(np.array(y_data) == label)[0]   
        label_samples_x = x_data[label_indices]
        label_dict[label] = label_samples_x
        
    return label_dict


def get_clients_data(num_clients : int, num_samples_by_label : int, dataset : str, nn_model : str) -> dict:
    
    """Distribute a dataset evenly accross num_clients clients. Works with datasets with 10 labels
    
    Arguments:
        num_clients : Number of clients of interest
        num_samples_by_label : Number of samples of each labels by client
        dataset: The name of the dataset to use (e.g 'fashion-mnist', 'mnist', or 'cifar10')
        nn_model : the training model type ('linear' or 'convolutional')

    Returns:
        client_dataset :  Dictionnary where each key correspond to a client index. The samples will be contained in the 'x' key and the target in 'y' key
    """
    
    import numpy as np 

    label_dict = create_label_dict(dataset, nn_model)

    clients_dictionary = {}
    client_dataset = {}

    for client in range(num_clients):
        
        clients_dictionary[client] = {}    
        
        for label in range(10):
        
            clients_dictionary[client][label]= label_dict[label][client*num_samples_by_label:(client+1)*num_samples_by_label]
    
    for client in range(num_clients):
    
        client_dataset[client] = {}    
    
        client_dataset[client]['x'] = np.concatenate([clients_dictionary[client][label] for label in range(10)], axis=0)
    
        client_dataset[client]['y'] = np.concatenate([[label]*len(clients_dictionary[client][label]) for label in range(10)], axis=0)
    
    return client_dataset



def rotate_images(client: Client, rotation: int) -> None:
    
    """ Rotate a Client's images, used for ``concept shift on features''
    
    Arguments:
        client : A Client object whose dataset images we want to rotate
        rotation : the rotation angle to apply  0 < angle < 360
    """
    
    import numpy as np

    images = client.data['x']

    if rotation > 0 :

        rotated_images = []
    
        for img in images:
    
            orig_shape = img.shape             
            rotated_img = np.rot90(img, k=rotation//90)  # Rotate image by specified angle 
            rotated_img = rotated_img.reshape(*orig_shape)
            rotated_images.append(rotated_img)   
    
        client.data['x'] = np.array(rotated_images)

    return

import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms


class AddGaussianNoise(object):
    
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        import torch
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddRandomJitter(object):

    def __init__(self, brightness =0.5, contrast = 1, saturation = 0.1, hue = 0.5):
        self.brightness = brightness,
        self.contrast = contrast, 
        self.saturation = saturation, 
        self.hue = hue
    
    def __call__(self, tensor):
        import torchvision.transforms as transforms
        transform = transforms.ColorJitter(brightness = self.brightness, contrast= self.contrast, 
                               saturation = self.saturation, hue = self.hue)
        return transform(tensor)

class CustomDataset(Dataset):
    
    def __init__(self, data, labels, transform=None):
        # Ensure data is in (N, H, W, C) format
        self.data = data  # Assume data is in (N, H, W, C) format
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # Shape (H, W, C)
        label = self.labels[idx]

        # Convert image to tensor and permute to (C, H, W)
        image = torch.tensor(image, dtype=torch.float)  # Convert to tensor
        image = image.permute(2, 0, 1)  # Change to (C, H, W)

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        return image, label

def data_preparation(client: Client, row_exp: dict) -> None:
    """Saves Dataloaders of train and test data in the Client attributes 
    
    Arguments:
        client : The client object to modify
        row_exp : The current experiment's global parameters
    """

    def to_device_tensor(data, device, data_dtype):
        data = torch.tensor(data, dtype=data_dtype)
        data = data.to(device)
        return data
    
    import torch 
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    import torchvision.transforms as transforms
    import numpy as np  # Import NumPy for transpose operation
    
    # Define data augmentation transforms
    if row_exp['dataset'] == 'cifar10': 
        train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Normalize if needed
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Transform for validation and test data (no augmentation, just normalization)
        test_val_transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize if needed
    ])
    else : 
        train_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),  # Normalize if needed
    ])
        
        # Transform for validation and test data (no augmentation, just normalization)
        test_val_transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),  # Normalize if needed
        ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split into train, validation, and test sets
    x_data, x_test, y_data, y_test = train_test_split(client.data['x'], client.data['y'], test_size=0.3, random_state=row_exp['seed'], stratify=client.data['y'])
    x_train, x_val, y_train, y_val  = train_test_split(x_data, y_data, test_size=0.25, random_state=42)


    # Create datasets with transformations
    train_dataset = CustomDataset(x_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(x_val, y_val, transform=test_val_transform)
    test_dataset = CustomDataset(x_test, y_test, transform=test_val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # Store DataLoaders in the client object
    setattr(client, 'data_loader', {'train': train_loader, 'val': validation_loader, 'test': test_loader})
    setattr(client, 'train_test', {'x_train': x_train, 'x_val': x_val, 'x_test': x_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test})

    return



def get_dataset_heterogeneities(heterogeneity_type: str) -> dict:

    """
    Retrieves the "skew" and "ratio" attributes of a given heterogeneity type

    Arguments:
        heterogeneity_type : The label of the heterogeneity scenario (labels-distribution-skew, concept-shift-on-labels, quantity-skew)
    Returns:
        dict_params: A dictionary of the form {<het>: []} where <het> is the applicable heterogeneity type 
    """
    dict_params = {}

    if heterogeneity_type == 'labels-distribution-skew':
        dict_params['skews'] = [[0,3,4,5,6,7,8,9], [0,1,2,5,6,7,8,9], [0,1,2,3,4,7,8,9], [0,1,2,3,4,5,6,9]]
        dict_params['ratios'] = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                               [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]
    
    elif heterogeneity_type == 'concept-shift-on-labels':
        dict_params['swaps'] = [(1,7),(2,7),(4,7),(3,8),(5,6),(7,9)]

    elif heterogeneity_type == 'quantity-skew':
        dict_params['skews'] = [0.1,0.2,0.6,1]

    return dict_params
    

def setup_experiment(row_exp: dict) -> Tuple[Server, list]:

    """ Setup function to create and personalize client's data 

    Arguments:
        row_exp : The current experiment's global parameters


    Returns: 
        model_server, list_clients: a nn model used the server in the FL protocol, a list of Client Objects used as nodes in the FL protocol

    """

    from src.models import GenericConvModel,GenericLinearModel
    from src.utils_fed import init_server_cluster
    import torch
    
    list_clients = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(row_exp['seed'])

    imgs_params = {'mnist': (28,1) , 'fashion-mnist': (28,1), 'kmnist': (28,1), 'cifar10': (32,3)}

    if row_exp['nn_model'] == "linear":
        
        model_server = Server(GenericLinearModel(in_size=imgs_params[row_exp['dataset']][0], n_channels=imgs_params[row_exp['dataset']][1])) 
    
    elif row_exp['nn_model'] == "convolutional": 
        
        model_server = Server(GenericConvModel(in_size=imgs_params[row_exp['dataset']][0], n_channels=imgs_params[row_exp['dataset']][1]))
       

    model_server.model.to(device)

    dict_clients = get_clients_data(row_exp['num_clients'],
                                    row_exp['num_samples_by_label'],
                                    row_exp['dataset'],
                                    row_exp['nn_model'])    
    
    for i in range(row_exp['num_clients']):

        list_clients.append(Client(i, dict_clients[i]))

    list_clients = add_clients_heterogeneity(list_clients, row_exp)
    
    if row_exp['exp_type'] == "client":

        init_server_cluster(model_server, list_clients, row_exp, imgs_params[row_exp['dataset']])

    return model_server, list_clients



def add_clients_heterogeneity(list_clients: list, row_exp: dict) -> list:
    """ Utility function to apply the relevant heterogeneity classes to each client
    
    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    Returns:
        The updated list of clients
    """

    dict_params = get_dataset_heterogeneities(row_exp['heterogeneity_type'])

    if row_exp['heterogeneity_type']  == "concept-shift-on-features": # rotations
        list_clients = apply_rotation(list_clients, row_exp)
    
    elif row_exp['heterogeneity_type'] == "concept-shift-on-labels": #label swaps
        list_clients = apply_label_swap(list_clients, row_exp, dict_params['swaps'])
    
    elif row_exp['heterogeneity_type'] == "quantity-skew": #less images altogether for certain clients
        list_clients = apply_quantity_skew(list_clients, row_exp, dict_params['skews']) 
    
    elif row_exp['heterogeneity_type'] == "labels-distribution-skew":
        list_clients = apply_labels_skew(list_clients, row_exp, dict_params['skews'], # less images of certain labels
                                          dict_params['ratios'])
    
    elif row_exp['heterogeneity_type'] == "features-distribution-skew": #change image qualities
        list_clients = apply_features_skew(list_clients, row_exp)
    elif row_exp['heterogeneity_type'] == "quantity-skew+concept-shift-on-features": #less images altogether for certain clients
        dict_clients = get_clients_data(5,
                                    int(row_exp['num_samples_by_label'] * 0.1),
                                    row_exp['dataset'],
                                    row_exp['nn_model'])
        for id in range(5) :
            client = Client(id,dict_clients[id])
            list_clients[id] = client 
        list_clients = apply_rotation(list_clients, row_exp)


    return list_clients



def apply_label_swap(list_clients : list, row_exp : dict, list_swaps : list) -> list:
    
    """ Utility function to apply label swaps on Client images

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
        list_swap : List containing the labels to swap by heterogeneity class
    Returns :
        Updated list of clients
    
    """
    n_swaps_types = len(list_swaps)
    
    n_clients_by_swaps_type = row_exp['num_clients'] // n_swaps_types
    
    for i in range(n_swaps_types):
        
        start_index = i * n_clients_by_swaps_type
        end_index = (i + 1) * n_clients_by_swaps_type

        list_clients_swapped = list_clients[start_index:end_index]

        for client in list_clients_swapped:
            
            client = swap_labels(list_swaps[i],client, str(i))
            
            data_preparation(client, row_exp)

        list_clients[start_index:end_index] = list_clients_swapped

    list_clients  = list_clients[:end_index]

    return list_clients




def apply_rotation(list_clients : list, row_exp : dict) -> list:

    """ Utility function to apply rotation 0,90,180 and 270 to 1/4 of Clients 

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    
    Returns:
        Updated list of clients
    """
    
    n_rotation_types = 4
    n_clients_by_rotation_type = row_exp['num_clients'] // n_rotation_types #TODO check edge cases where n_clients < n_rotation_types

    for i in range(n_rotation_types):
        
        start_index = i * n_clients_by_rotation_type
        end_index = (i + 1) * n_clients_by_rotation_type

        list_clients_rotated = list_clients[start_index:end_index]

        for client in list_clients_rotated:

            rotation_angle = (360 // n_rotation_types) * i

            rotate_images(client , rotation_angle)
            
            data_preparation(client, row_exp)
            
            setattr(client,'heterogeneity_class', f"rot_{rotation_angle}")

        list_clients[start_index:end_index] = list_clients_rotated

    list_clients  = list_clients[:end_index]

    return list_clients


def apply_labels_skew(list_clients : list, row_exp : dict, list_skews : list, list_ratios : list) -> list:
    
    """ Utility function to apply label skew to Clients' data 

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    
    Returns:
        Updated list of clients
    """

    n_skews = len(list_skews)
    n_clients_by_skew = row_exp['num_clients'] // n_skews 

    for i in range(n_skews):

        start_index = i * n_clients_by_skew
        end_index = (i + 1) * n_clients_by_skew

        list_clients_skewed = list_clients[start_index:end_index]

        for client in list_clients_skewed:
            
            unbalancing(client, list_skews[i], list_ratios[i])
            
            data_preparation(client, row_exp)

            setattr(client,'heterogeneity_class', f"lbl_skew_{str(i)}")

        list_clients[start_index:end_index] = list_clients_skewed
    
    list_clients = list_clients[:end_index]

    return list_clients



def apply_quantity_skew(list_clients : list, row_exp : dict, list_skews : list) -> list:
    
    """ Utility function to apply quantity skew to Clients' data 
     For each element in list_skews, apply the skew to an equal subset of Clients 


    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
        list_skew : List of float 0 < i < 1  with quantity skews to subsample data
    
    Returns:
        Updated list of clients
    """
    
    n_max_samples = 100 # TODO: parameterize by dataset

    n_skews = len(list_skews)
    n_clients_by_skew = row_exp['num_clients'] // n_skews  

    dict_clients = [get_clients_data(n_clients_by_skew,
                                    int(n_max_samples * skew),
                                    row_exp['dataset'],
                                    row_exp['nn_model']) 
                                    for skew in list_skews] 
    list_clients = []

    for c in range(n_clients_by_skew):

        for s in range(len(list_skews)):
            
            client = Client(c * len(list_skews)+ s, dict_clients[s][c])
            setattr(client,'heterogeneity_class', str(s))
            list_clients.append(client)

    for client in list_clients :

        data_preparation(client, row_exp)

    return list_clients



def apply_features_skew(list_clients : list, row_exp : dict) -> list :
    
    """ Utility function to apply features skew to Clients' data 

    Arguments:
        list_clients : List of Client Objects with specific heterogeneity_class 
        row_exp : The current experiment's global parameters
    
    Returns:
        Updated list of clients
    """
    
    n_skew_types = 3 #TODO parameterize
    
    n_clients_by_skew = row_exp['num_clients'] // n_skew_types  
    
    for i in range(n_skew_types):

        start_index = i * n_clients_by_skew
        end_index = (i + 1) * n_clients_by_skew

        list_clients_rotated = list_clients[start_index:end_index]

        for client in list_clients_rotated:
            if client.id % n_skew_types == 1:
                client.data['x'] = erode_images(client.data['x'])
                client.heterogeneity_class = 'erosion'

            elif client.id % n_skew_types == 2 :
                client.data['x'] = dilate_images(client.data['x'])
                client.heterogeneity_class = 'dilatation'

            else :
                client.heterogeneity_class = 'none'

            data_preparation(client, row_exp)

        list_clients[start_index:end_index] = list_clients_rotated
    
    list_clients = list_clients[:end_index]
    
    return list_clients



def swap_labels(labels : list, client : Client, heterogeneity_class : int) -> Client:

    """ Utility Function for label swapping used for concept shift on labels. Sets the attribute "heterogeneity class"
    
    Arguments:
        labels : Labels to swap
        client : The Client object whose data we want to apply the swap on
    Returns:
        Client with labels swapped
    """

    newlabellist = client.data['y'] 

    otherlabelindex = newlabellist==labels[1]

    newlabellist[newlabellist==labels[0]]=labels[1]

    newlabellist[otherlabelindex] = labels[0]

    client.data['y']= newlabellist

    setattr(client,'heterogeneity_class', heterogeneity_class)

    return client


def centralize_data(list_clients: list, row_exp: dict) -> Tuple[DataLoader, DataLoader]:
    """Centralize data of the federated learning setup for central model comparison

    Arguments:
        list_clients : The list of Client Objects
        row_exp : The current experiment's global parameters

        
    Returns:
        Train and test torch DataLoaders with data of all Clients
    """
    
    from torchvision import transforms
    import torch 
    from torch.utils.data import DataLoader,TensorDataset
    import numpy as np 
    
    if row_exp['dataset'] == 'cifar10': 
        train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Normalize if needed
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Transform for validation and test data (no augmentation, just normalization)
        test_val_transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    else : 
        train_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),  # Normalize if needed
    ])
        # Transform for validation and test data (no augmentation, just normalization)
        test_val_transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),  # Normalize if needed
        ])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Concatenate training data from all clients
    x_train = np.concatenate([list_clients[id].train_test['x_train'] for id in range(len(list_clients))], axis=0)
    y_train = np.concatenate([list_clients[id].train_test['y_train'] for id in range(len(list_clients))], axis=0)

    # Concatenate validation data from all clients
    x_val = np.concatenate([list_clients[id].train_test['x_val'] for id in range(len(list_clients))], axis=0)
    y_val = np.concatenate([list_clients[id].train_test['y_val'] for id in range(len(list_clients))], axis=0)

    # Concatenate test data from all clients
    x_test = np.concatenate([list_clients[id].train_test['x_test'] for id in range(len(list_clients))], axis=0)
    y_test = np.concatenate([list_clients[id].train_test['y_test'] for id in range(len(list_clients))], axis=0)

    # Create Custom Datasets
    train_dataset = CustomDataset(x_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(x_val, y_val, transform=test_val_transform)
    test_dataset = CustomDataset(x_test, y_test, transform=test_val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)  # Validation typically not shuffled
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)  # Test data typically not shuffled

    return train_loader, val_loader, test_loader





def unbalancing(client : Client ,labels_list : list ,ratio_list: list) -> Client :
    
    """ Downsample the dataset of a client with each elements of the labels_list will be downsampled by the corresponding ration of ratio_list

    Arguments: 
        client : Client whose dataset we want to downsample
        labels_list : Labels to downsample in the Client's dataset
        ratio_list : Ratios to use for downsampling the labels
    """
    
    import pandas as pd
    from imblearn.datasets import make_imbalance
    from math import prod

    def ratio_func(y, multiplier, minority_class):
    
        from collections import Counter
    
        target_stats = Counter(y)
        return {minority_class: int(multiplier * target_stats[minority_class])}


    x_train = client.data['x']
    y_train = client.data['y']
    
    orig_shape = x_train.shape
    
     # flatten the images 
    X_resampled = x_train.reshape(-1, prod(orig_shape[1:]))
    y_resampled = y_train
    
    for i in range(len(labels_list)):
    
        X = pd.DataFrame(X_resampled)
    
        X_resampled, y_resampled = make_imbalance(X,
                y_resampled,
                sampling_strategy=ratio_func,
                **{"multiplier": ratio_list[i], "minority_class": labels_list[i]})

    client.data['x'] = X_resampled.to_numpy().reshape(-1, *orig_shape[1:])
    client.data['y'] = y_resampled
    
    return client


def dilate_images(x_train : ndarray, kernel_size : tuple = (3, 3)) -> ndarray:
    
    """ Perform dilation operation on a batch of images using a given kernel.
    Make image 'bolder' for features distribution skew setup
    
    
    Arguments:
        x_train : Input batch of images (3D array with shape (n, height, width)).
        kernel_size : Size of the structuring element/kernel for dilation.

    Returns:
        ndarray Dilation results for all images in the batch.
    """
    
    import cv2
    import numpy as np 

    n = x_train.shape[0] 

    dilated_images = np.zeros_like(x_train, dtype=np.uint8)

    # Create the kernel for dilation
    kernel = np.ones(kernel_size, np.uint8)

    for i in range(n):
    
        dilated_image = cv2.dilate(x_train[i], kernel, iterations=1)
    
        dilated_images[i] = dilated_image

    return dilated_images


def erode_images(x_train : ndarray, kernel_size : tuple =(3, 3)) -> ndarray:
    """
    Perform erosion operation on a batch of images using a given kernel.
    Make image 'finner' for features distribution skew setup

    Arguments:
        x_train : Input batch of images (3D array with shape (n, height, width)).
        kernel_size :  Size of the structuring element/kernel for erosion.

    Returns:
        ndarray of Erosion results for all images in the batch.
    """
    
    import cv2
    import numpy as np 

    n = x_train.shape[0]  
    eroded_images = np.zeros_like(x_train, dtype=np.uint8)

    # Create the kernel for erosion
    kernel = np.ones(kernel_size, np.uint8)

    # Iterate over each image in the batch
    for i in range(n):
        # Perform erosion on the current image
        eroded_image = cv2.erode(x_train[i], kernel, iterations=1)
        # Store the eroded image in the results array
        eroded_images[i] = eroded_image

    return eroded_images


def get_uid(str_obj: str) -> str:
    """
    Generates an (almost) unique Identifier given a string object.
    Note: Collision probability is low enough to be functional for the use case desired which is to uniquely identify experiment parameters using an int
    """

    import hashlib
    hash = hashlib.sha1(str_obj.encode("UTF-8")).hexdigest()
    return hash

    
