import torch
from sklearn.datasets import load_digits
import numpy as np
from tensorflow.keras.datasets import mnist
from src.fedclass import Client, Server
from torch.utils.data import ConcatDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def shuffle (array,seed=42): 
    # Assuming arr is your numpy array of shape (n, 28, 28)  # Example array

# Generate a list of shuffled indices
    shuffled_indices = np.arange(array.shape[0])
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to reorder the array along the first axis
    shuffled_arr = array[shuffled_indices].copy()
    return shuffled_arr

def create_mnist_label_dict(seed = 42) : 
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

def data_distribution(number_of_clients, samples_by_client_of_each_labels,seed = 42):
    """
    Distribute Mnist Dataset evenly accross number_of_clients clients
    ----------
    number_of_clients : int
        number of client of interest
        
    samples_by_client_of_each_labels : int
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
    for client in range(number_of_clients):
        clients_dictionary[client] = {}    
        for label in range(10):
            clients_dictionary[client][label]= label_dict[label][client*samples_by_client_of_each_labels:(client+1)*samples_by_client_of_each_labels]
    for client in range(number_of_clients):
        client_dataset[client] = {}    
        client_dataset[client]['x'] = np.concatenate([clients_dictionary[client][label] for label in range(10)], axis=0)
        client_dataset[client]['y'] = np.concatenate([[label]*len(clients_dictionary[client][label]) for label in range(10)], axis=0)
    return client_dataset

def rotate_images(client,rotation):
    images = client.data['x']
    setattr(client,'heterogeneity',str(rotation))
    if rotation >0 :
        rotated_images = []
        for img in images:
            rotated_img = np.rot90(img, k=rotation//90)  # Rotate image by specified angle
            rotated_images.append(rotated_img)   
        client.data['x'] = np.array(rotated_images)

def data_preparation(client):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset,TensorDataset
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
    

def setup_experiment_rotation(number_of_clients,number_of_samples_by_clients, model,number_of_cluster=1,seed =42) :
    clientdata = data_distribution(number_of_clients, number_of_samples_by_clients,seed)
    clientlist = []
    for id in range(number_of_clients):
        clientlist.append(Client(id,clientdata[id]))
    my_server = Server(model)
    # Apply rotation 0,90,180 and 270 to 1/4 of clients each
    n = number_of_clients//4
    for i in range(4):
        start_index = i * n
        end_index = (i + 1) * n
        clientlistrotated = clientlist[start_index:end_index]
        for client in clientlistrotated:
            rotate_images(client,90*i)
            data_preparation(client)
        clientlist[start_index:end_index] = clientlistrotated
    return my_server, clientlist

def label_swap(labels, client):
    # labels : tuple of labels to swap
    newlabellist = client.data['y'] 
    otherlabelindex = newlabellist==labels[1]
    newlabellist[newlabellist==labels[0]]=labels[1]
    newlabellist[otherlabelindex] = labels[0]
    client.data['y']= newlabellist
    setattr(client,'heterogeneity', str(labels))
    
def setup_experiment_labelswap(number_of_clients,number_of_samples_by_clients, model,swaplist=[(1,7),(2,7),(4,7),(3,8),(5,6),(7,9)],number_of_cluster=1,seed =42):
    clientdata = data_distribution(number_of_clients, number_of_samples_by_clients,seed)
    clientlist = []
    for id in range(number_of_clients):
        clientlist.append(Client(id,clientdata[id]))
    my_server = Server(model)
    # Apply rotation 0,90,180 and 270 to 1/4 of clients each
    n = number_of_clients // len(swaplist)
    for i in range(number_of_cluster):
        start_index = i * n
        end_index = (i + 1) * n
        clientlistswap = clientlist[start_index:end_index]
        for client in clientlistswap:
            label_swap(swaplist[i],client)
            data_preparation(client)
        clientlist[start_index:end_index] = clientlistswap
    return my_server, clientlist

def setup_experiment_quantity_skew(model,number_of_client=200,number_of_max_samples=100,skewlist=[1,0.5,0.25,0.1,0.05], seed = 42):
    number_of_skew = len(skewlist)
    number_of_client_by_skew = number_of_client//number_of_skew 
    clientdata = [data_distribution(number_of_client_by_skew,int(number_of_max_samples*skew),seed) for skew in skewlist]        
    clientlist = []
    for id in range(number_of_client_by_skew):
        for skew_id in range(len(skewlist)):
            client = Client(id*len(skewlist)+ skew_id,clientdata[skew_id][id])
            setattr(client,'heterogeneity',str(skewlist[skew_id]))
            clientlist.append(client)
    for client in clientlist : 
        data_preparation(client)
    my_server = Server(model)
    return my_server, clientlist
    
def centralize_data(clientlist):
    from torch.utils.data import DataLoader,Dataset,TensorDataset
    x_train = np.concatenate([clientlist[id].train_test['x_train'] for id in range(len(clientlist))],axis = 0)
    x_test = np.concatenate([clientlist[id].train_test['x_test'] for id in range(len(clientlist))],axis = 0)
    y_train = np.concatenate([clientlist[id].train_test['y_train'] for id in range(len(clientlist))],axis = 0)
    y_test = np.concatenate([clientlist[id].train_test['y_test'] for id in range(len(clientlist))],axis = 0)
    print(x_train.shape)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader

from collections import Counter
import pandas as pd
import numpy as np
from imblearn.datasets import make_imbalance
import matplotlib.pyplot as plt

def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}

def unbalancing(client,labels_list ,ratio_list, plot = False):
    from imblearn.datasets import make_imbalance
    x_train = client.data['x']
    y_train = client.data['y']
    X_resampled = x_train.reshape(-1, 784) # flatten the images 
    y_resampled = y_train
    for i in range(len(labels_list)):
        X = pd.DataFrame(X_resampled)
        X_resampled, y_resampled = make_imbalance(X,
                y_resampled,
                sampling_strategy=ratio_func,
                **{"multiplier": ratio_list[i], "minority_class": labels_list[i]})
    if plot == True : 
        plt.hist(y_resampled, bins=np.arange(min(y), 11), align='left', rwidth=1)
        plt.title("Ratio ")
        plt.show()
    ### unflatten the images 
    client.data['x'] = X_resampled.to_numpy().reshape(-1,28,28)
    client.data['y'] = y_resampled
    setattr(client,'heterogeneity',str((labels_list,ratio_list)))


def setup_experiment_labels_skew(model,number_of_clients=48,number_of_samples_by_clients=50,skewlist=[[1,2],[3,4],[5,6],[7,8],[9,0]], ratiolist = [[0.5,0.5],[0.4,0.4],[0.3,0.3],[0.2,0.2],[0.1,0.1]],seed = 42):
    clientdata = data_distribution(number_of_clients, number_of_samples_by_clients,seed)
    clientlist = []
    for id in range(number_of_clients):
        clientlist.append(Client(id,clientdata[id]))
    my_server = Server(model)
    # Apply rotation 0,90,180 and 270 to 1/4 of clients each
    n = number_of_clients // len(skewlist)
    for i in range(len(skewlist)):
        start_index = i * n
        end_index = (i + 1) * n
        clientlistskew = clientlist[start_index:end_index]
        for client in clientlistskew:
            unbalancing(client,skewlist[i],ratiolist[i])
            data_preparation(client)
        clientlist[start_index:end_index] = clientlistskew
    return my_server, clientlist

