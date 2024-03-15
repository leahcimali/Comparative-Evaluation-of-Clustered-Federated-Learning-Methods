import torch
from sklearn.datasets import load_digits
import numpy as np
from tensorflow.keras.datasets import mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_distribution(number_of_clients, samples_by_client_of_each_labels):
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
    
    # Load the Mnist dataset
     
    (x_train, y_train), _ = mnist.load_data()

# Seperating Mnist by labels
    label_dict = {}
    for label in range(10):
        label_indices = np.where(y_train == label)[0]
        label_samples_x = x_train[label_indices]
        # Dictionnary that contains all samples of the labels to associate key 
        label_dict[label] = label_samples_x
        
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

def data_preparation(Client,rotation=0):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset,TensorDataset

    if rotation > 0 :
        Client.data['x'] = rotate_images(Client.data['x'],rotation)
    x_train, x_test, y_train, y_test = train_test_split(Client.data['x'], Client.data['y'], test_size=0.2, random_state=42,stratify=Client.data['y'])
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64)
    Client.dataloader = {'train_loader' : train_loader,'data_loader': test_loader}
