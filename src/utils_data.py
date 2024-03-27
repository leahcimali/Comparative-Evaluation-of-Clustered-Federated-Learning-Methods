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

def rotate_images(images, angle):
    rotated_images = []
    for img in images:
        rotated_img = np.rot90(img, k=angle//90)  # Rotate image by specified angle
        rotated_images.append(rotated_img)
    return np.array(rotated_images)

def data_preparation(client,rotation=0):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset,TensorDataset

    if rotation > 0 :
        client.data['x'] = rotate_images(client.data['x'],rotation)
    x_train, x_test, y_train, y_test = train_test_split(client.data['x'], client.data['y'], test_size=0.3, random_state=42,stratify=client.data['y'])
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
        elements = clientlist[start_index:end_index]
        for element in elements:
            data_preparation(element,90*i)
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

