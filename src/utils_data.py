import torch
import numpy as np

from collections import Counter
import pandas as pd
import numpy as np

from src.fedclass import Client, Server
from src.models import SimpleLinear 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def shuffle(array,seed=42): 
    # Function to shuffle the samples 
    # Generate a list of shuffled indices
    shuffled_indices = np.arange(array.shape[0])
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to reorder the array along the first axis
    shuffled_arr = array[shuffled_indices].copy()
    return shuffled_arr

def create_mnist_label_dict(seed = 42) :
    # Create a dictionary of mnist samples by labels 
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Seperating Mnist by labels
    label_dict = {}
    for label in range(10):
        label_indices = np.where(y_train == label)[0]
        label_samples_x = x_train[label_indices]
        # Dictionnary that contains all samples of the labels to associate key 
        label_dict[label] = shuffle(label_samples_x, seed)
        
    return label_dict

def get_clients_data(num_clients, num_samples_by_label, seed = 42):
    """
    Distribute Mnist Dataset evenly accross num_clients clients
    ----------
    num_clients : int
        number of client of interest
        
    num_samples_by_label : int
        number of samples of each labels by clients
    Returns
    -------
    client_dataset : Dictionnary
        Dictionnary where each key correspond to a client index. The samples will be contained in the 'x' key and the target in 'y' key
    """
    
    label_dict = create_mnist_label_dict(seed)
    # Initialize dictionary to store client data
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

def rotate_images(client, rotation):
    # Rotate images, used of concept shift on features
    images = client.data['x']
    
    if rotation >0 :
        rotated_images = []
        for img in images:
            rotated_img = np.rot90(img, k=rotation//90)  # Rotate image by specified angle
            rotated_images.append(rotated_img)   
        client.data['x'] = np.array(rotated_images)

def data_preparation(client):
    # Train test split of a client's data and create onf dataloaders for local model training
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    x_train, x_test, y_train, y_test = train_test_split(client.data['x'], client.data['y'], test_size=0.3, random_state=42,stratify=client.data['y'])
    x_train, x_test = x_train/255.0 , x_test/255.0
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)
    setattr(client, 'data_loader', {'train' : train_loader,'test': test_loader})
    setattr(client,'train_test', {'x_train': x_train,'x_test': x_test, 'y_train': y_train, 'y_test': y_test})
    


def mnist_dataset_heterogeneities(heterogeneity_type, exp_type):
    
    dict_params = {}

    if heterogeneity_type == "labels_distribution_skew":
        dict_params['skews'] = [[1,2],[3,4],[5,6],[7,8]]
        dict_params['ratios'] = [[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2]]

    elif heterogeneity_type  == "labels_distribution_skew_balancing":
        dict_params['skews'] = [[0,1,2,3,4],[5,6,7,8,9],[0,2,4,6,8],[1,3,5,7,9]]
        dict_params['ratios'] = [[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1]]
        
    elif heterogeneity_type == 'labels_distribution_skew_downsampled':
        dict_params['skews'] = [[0,3,4,5,6,7,8,9], [0,1,2,5,6,7,8,9], [0,1,2,3,4,7,8,9], [0,1,2,3,4,5,6,9]]
        dict_params['ratios'] = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                               [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]

    elif heterogeneity_type == 'concept_shift_on_labels':
        dict_params['swaps'] = [(1,7),(2,7),(4,7),(3,8),(5,6),(7,9)]

    elif heterogeneity_type == 'quantity_skew':
        dict_params['skews'] = [0.1,0.2,0.6,1]

    return dict_params




def setup_experiment(row_exp, model=SimpleLinear()):

    list_clients = []
    model_server = Server(model)

    dict_clients = get_clients_data(row_exp['num_clients'],
                                    row_exp['num_samples_by_label'],
                                    row_exp['seed'])    
    
    for i in range(row_exp['num_clients']):
        list_clients.append(Client(i, dict_clients[i]))
    
    list_clients = add_clients_heterogeneity(list_clients, row_exp)
    
    list_heterogeneities = list(set(client.heterogeneity_class for client in list_clients))

    return model_server, list_clients, list_heterogeneities



def add_clients_heterogeneity(list_clients, row_exp):
    
    dict_params = mnist_dataset_heterogeneities(row_exp['heterogeneity_type'], row_exp['exp_type'])

    if row_exp['heterogeneity_type']  == "concept_shift_on_features": # rotations?
        list_clients = apply_rotation(list_clients, row_exp)
    
    elif row_exp['heterogeneity_type'] == "concept_shift_on_labels": #label swaps
        list_clients = apply_label_swap(list_clients, row_exp, dict_params['swaps'])
    
    elif row_exp['heterogeneity_type'] == "quantity_skew":
        list_clients = apply_quantity_skew(list_clients, row_exp, dict_params['skews'])
    
    elif "labels_distribution_skew" in row_exp['heterogeneity_type']:
        list_clients = apply_labels_skew(list_clients, row_exp, dict_params['skews'],
                                          dict_params['ratios'])
    
    elif row_exp['heterogeneity_type'] == "features_distribution_skew":
        list_clients = apply_features_skew(list_clients, row_exp)
    
    return list_clients



def apply_label_swap(list_clients, row_exp, list_swaps):
   
    n_swaps_types = len(list_swaps)
    n_clients_by_swaps_type = row_exp['num_clients'] // n_swaps_types
    
    for i in range(n_swaps_types):
        
        start_index = i * n_clients_by_swaps_type
        end_index = (i + 1) * n_clients_by_swaps_type

        list_clients_swapped = list_clients[start_index:end_index]

        for client in list_clients_swapped:
            
            client = swap_labels(list_swaps[i],client, str(i))
            data_preparation(client)

        list_clients[start_index:end_index] = list_clients_swapped
    
    return list_clients




def apply_rotation(list_clients, row_exp):

    # Apply rotation 0,90,180 and 270 to 1/4 of clients each
    n_rotation_types = 4
    n_clients_by_rotation_type = row_exp['num_clients'] // n_rotation_types #TODO check edge cases where n_clients < n_rotation_types

    for i in range(n_rotation_types):
        
        start_index = i * n_clients_by_rotation_type
        end_index = (i + 1) * n_clients_by_rotation_type

        list_clients_rotated = list_clients[start_index:end_index]

        for client in list_clients_rotated:

            rotation_angle = (360 // n_rotation_types) * i

            rotate_images(client , rotation_angle)
            
            data_preparation(client)
            
            setattr(client,'heterogeneity_class', f"rot_{rotation_angle}")

        list_clients[start_index:end_index] = list_clients_rotated

    list_clients  = list_clients[:end_index]
    return list_clients


def apply_labels_skew(list_clients, row_exp, list_skews, list_ratios):
    
    n_skews = len(list_skews)
    n_clients_by_skew = row_exp['num_clients'] // n_skews 

    for i in range(n_skews):

        start_index = i * n_clients_by_skew
        end_index = (i + 1) * n_clients_by_skew

        list_clients_skewed = list_clients[start_index:end_index]

        for client in list_clients_skewed:
            
            unbalancing(client, list_skews[i], list_ratios[i])
            
            data_preparation(client)

            setattr(client,'heterogeneity_class', f"lbl_skew_{str(i)}")

        list_clients[start_index:end_index] = list_clients_skewed
    
    list_clients = list_clients[:end_index]

    return list_clients



def apply_quantity_skew(list_clients, row_exp, list_skews):
    
    # Setup server and clients for quantity skew experiment
    # Skew list create for each element an equal subset of clients with the corresponding percentage of the client data
    
    n_max_samples = 100 # TODO: parameterize by dataset

    n_skews = len(list_skews)
    n_clients_by_skew = row_exp['num_clients'] // n_skews  

    dict_clients = [get_clients_data(n_clients_by_skew,
                                    int(n_max_samples * skew)) 
                                    for skew in list_skews] 
           
    list_clients = []

    for c in range(n_clients_by_skew):

        for s in range(len(list_skews)):
            
            client = Client(c * len(list_skews)+ s, dict_clients[s][c])
            setattr(client,'heterogeneity_class', str(s))
            list_clients.append(client)

    for client in list_clients :

        data_preparation(client)

    
    return list_clients



def apply_features_skew(list_clients, row_exp) :
    # Setup server and clients for features distribution skew experiments
    
    n_skew_types = 3
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

            data_preparation(client)

        list_clients[start_index:end_index] = list_clients_rotated
    
    list_clients = list_clients[:end_index]

    return list_clients



def swap_labels(labels, client, heterogeneity_class):

    # Function for label swapping use for concept shift on labels
    # labels : tuple of labels to swap
    newlabellist = client.data['y'] 
    otherlabelindex = newlabellist==labels[1]
    newlabellist[newlabellist==labels[0]]=labels[1]
    newlabellist[otherlabelindex] = labels[0]
    client.data['y']= newlabellist
    setattr(client,'heterogeneity_class', heterogeneity_class)
    return client

def centralize_data(clientlist):
    # Centralize data of the federated learning setup for central model comparison
    from torch.utils.data import DataLoader,TensorDataset
    x_train = np.concatenate([clientlist[id].train_test['x_train'] for id in range(len(clientlist))],axis = 0)
    x_test = np.concatenate([clientlist[id].train_test['x_test'] for id in range(len(clientlist))],axis = 0)
    y_train = np.concatenate([clientlist[id].train_test['y_train'] for id in range(len(clientlist))],axis = 0)
    y_test = np.concatenate([clientlist[id].train_test['y_test'] for id in range(len(clientlist))],axis = 0)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader



def ratio_func(y, multiplier, minority_class):
    # downsample a label by multiplier
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}

def unbalancing(client,labels_list ,ratio_list):
    # downsample the dataset of a client with each elements of the labels_list will be downsample with the corresponding ration of ratio_list
    from imblearn.datasets import make_imbalance
    x_train = client.data['x']
    y_train = client.data['y']
    (nsamples, i_dim,j_dim) = x_train.shape
    X_resampled = x_train.reshape(-1, i_dim * j_dim) # flatten the images 
    y_resampled = y_train
    
    for i in range(len(labels_list)):
        X = pd.DataFrame(X_resampled)
        X_resampled, y_resampled = make_imbalance(X,
                y_resampled,
                sampling_strategy=ratio_func,
                **{"multiplier": ratio_list[i], "minority_class": labels_list[i]})

    ### unflatten the images 
    client.data['x'] = X_resampled.to_numpy().reshape(-1, i_dim, j_dim)
    client.data['y'] = y_resampled
    
    return client


def load_users_data(directory, number_of_users, seed = 42):
    import os
    import json
    import random

    """
    Loads n = number_of_users random JSON users_datas from the specified directory.

    Parameters
    ----------
    directory : str
        The directory path where JSON users_datas are located.
    seed : int
        A random seed to ensure reproducibility of random selection.
    number_of_users : int
        The number of JSON files to load randomly. Correspond to the number of users

    Returns
    -------
    users_data : list
        A list containing the JSON data loaded from the randomly selected files.
    Each element of the list is a user with its own sample. 
    It is a dictionnary of the form : 
    #{'users':[], 'num_samples':{['user_data : val]: }, 'user_data':[json_data['users'][0] : {'x': features, 'y': labels}]} 
    """
    
    # Set the random seed
    random.seed(seed)
    
    # Get a list of JSON files in the directory
    json_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]
    
    # Sample n files without replacement
    selected_files = random.sample(json_files, min(number_of_users, len(json_files)))
    
    # Load the selected JSON files into a list
    users_data = []
    for filename in selected_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            json_data = json.load(file)
            users_data.append(json_data)
    
    return users_data

def dilate_images(x_train, kernel_size=(3, 3)):
    import cv2
    """
    Perform dilation operation on a batch of images using a given kernel.
    Make image 'bolder' for features distribution skew setup
    Parameters:
        x_train (ndarray): Input batch of images (3D array with shape (n, height, width)).
        kernel_size (tuple): Size of the structuring element/kernel for dilation.

    Returns:
        ndarray: Dilation results for all images in the batch.
    """
    import cv2
    n = x_train.shape[0]  # Number of images in the batch
    dilated_images = np.zeros_like(x_train, dtype=np.uint8)

    # Create the kernel for dilation
    kernel = np.ones(kernel_size, np.uint8)

    # Iterate over each image in the batch
    for i in range(n):
        # Perform dilation on the current image
        dilated_image = cv2.dilate(x_train[i], kernel, iterations=1)
        # Store the dilated image in the results array
        dilated_images[i] = dilated_image

    return dilated_images

def erode_images(x_train, kernel_size=(3, 3)):
    """
    Perform erosion operation on a batch of images using a given kernel.
    Make image 'finner' for features distribution skew setup

    Parameters:
        x_train (ndarray): Input batch of images (3D array with shape (n, height, width)).
        kernel_size (tuple): Size of the structuring element/kernel for erosion.

    Returns:
        ndarray: Erosion results for all images in the batch.
    """
    import cv2
    n = x_train.shape[0]  # Number of images in the batch
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



def save_results(model_server, row_exp):
    
    import torch

    if row_exp['exp_type'] == "client" or "server":
        for cluster_id in range(row_exp['num_clusters']): 
            torch.save(model_server.clusters_models[cluster_id].state_dict(), f"./results/{row_exp['output']}_{row_exp['exp_type']}_model_cluster_{cluster_id}.pth")

    return 
