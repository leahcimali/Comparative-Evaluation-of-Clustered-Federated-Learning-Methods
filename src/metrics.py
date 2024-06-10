from src.utils_training import test_model
from src.utils_fed import model_weight_matrix
import numpy as np 
import pandas as pd
import json
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.spatial import distance

def average_intracluster_distance(X, labels):
    try:
        total_distance = 0
        count = 0
        for label in np.unique(labels):
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                total_distance += np.mean(pairwise_distances(cluster_points))
                count += 1
        if count == 0:
            return None  # No valid clusters found
        return total_distance / count
    except ValueError:
        return None  # Catching ValueError
        
def intracluster_distance_variance(X, labels):
    try:
        distances = []
        for label in np.unique(labels):
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                distances.extend(pairwise_distances(cluster_points).flatten())
        return np.var(distances)
    except ValueError:
        return None
    
def dunn_index(X, labels):
    try:
        min_intercluster_distance = np.inf
        max_intracluster_distance = -np.inf
        for label1 in np.unique(labels):
            for label2 in np.unique(labels):
                if label1 != label2:
                    intercluster_distance = np.min(pairwise_distances(X[labels == label1], X[labels == label2]))
                    min_intercluster_distance = min(min_intercluster_distance, intercluster_distance)
            max_intracluster_distance = max(max_intracluster_distance, np.max(pairwise_distances(X[labels == label1])))
        return min_intercluster_distance / max_intracluster_distance
    except ValueError:
        return None
    
def davies_bouldin_index(X, labels):
    try:
        k = len(np.unique(labels))
        centroids = [np.mean(X[labels == label], axis=0) for label in np.unique(labels)]
        average_distances = np.zeros(k)
        for i in range(k):
            intracluster_distances = pairwise_distances(X[labels == i], [centroids[i]])
            average_distances[i] = np.mean(intracluster_distances)
        db_index = 0
        for i in range(k):
            max_similarity = -np.inf
            for j in range(k):
                if i != j:
                    similarity = (average_distances[i] + average_distances[j]) / distance.euclidean(centroids[i], centroids[j])
                    max_similarity = max(max_similarity, similarity)
            db_index += max_similarity
        return db_index / k
    except ValueError:
        return None

def calinski_harabasz_index(X, labels):
    try:
        k = len(np.unique(labels))
        centroids = [np.mean(X[labels == label], axis=0) for label in np.unique(labels)]
        within_cluster_dispersion = sum([np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k)])
        between_cluster_dispersion = sum([np.sum((centroids[i] - np.mean(X, axis=0)) ** 2) * len(X[labels == i]) for i in range(k)])
        return between_cluster_dispersion / within_cluster_dispersion * (len(X) - k) / (k - 1)
    except ValueError:
        return None

# Example usage:
# Assuming X is your data and labels are cluster labels assigned to each data point
# You can compute the metrics like this:
# avg_intra_dist = average_intracluster_distance(X, labels)
# intra_dist_var = intracluster_distance_variance(X, labels)
# dunn_idx = dunn_index(X, labels)
# db_idx = davies_bouldin_index(X, labels)
# ch_idx = calinski_harabasz_index(X, labels)


def report_CFL(my_server, client_list, config):
    # function that create an experiment report and save it as a json
    from sklearn.metrics import silhouette_score
    import json
    import yaml   
    weight_matrix = model_weight_matrix(client_list)
    clusters_identities = {client.id : client.cluster_id for client in client_list}
    cluster_id = pd.DataFrame.from_dict(clusters_identities, orient='index', columns=['cluster_id'])
    X = weight_matrix 
    labels= cluster_id.values
    try : 
        silhouette_scores = silhouette_score(X, labels, metric='euclidean')
    except ValueError:
        silhouette_scores = None
        
    avg_intra_dist = average_intracluster_distance(X, labels)
    intra_dist_var = intracluster_distance_variance(X, labels)
    dunn_idx = dunn_index(X, labels)
    db_idx = davies_bouldin_index(X, labels) 
    config['silhouette'] = silhouette_scores
    config['avg_intra_dist'] = intra_dist_var
    config['duhn_index'] = dunn_idx
    config['davies_bouldin_index'] = db_idx
    print('Number of clusters : ', my_server.num_clusters)
    print ('AVG intra cluster dist : ', avg_intra_dist) 
    print('Intracluster dist variance : ', intra_dist_var) 
    print('Duhn index : ', dunn_idx)
    print( 'davies_bouldin_index : ', db_idx)
    print('silhouette_scores : ', silhouette_scores)
    
    for cluster_id in range(my_server.num_clusters):
        print('For cluster ', cluster_id, ' : ')
        client_cluster_list = [client for client in client_list if client.cluster_id == cluster_id]
        clients_accs = []
        print('Number of cluster members : ', len(client_cluster_list))
        if len(client_cluster_list) > 0 :
            client_heterogeneity = []
            for client in client_cluster_list : 
                acc = test_model(my_server.clusters_models[cluster_id], client.data_loader['test'])*100
                clients_accs.append(acc)
            print("Cluster accuracy : ", np.mean(clients_accs))
            print("Cluster accuracy std : ", np.std(clients_accs))
            cluster_heterogeneity = [client.heterogeneity for client in client_cluster_list]
            values, counts = np.unique(cluster_heterogeneity, return_counts = True)
            hetero_dict= {str(values[i]) : int(counts[i]) for i in range(len(values))}
            for key, value in hetero_dict.items() : 
                print('Cluster heterogeneity : ', key , ' , Quantity :', value)
            config[f'Cluster {cluster_id}'] = {'num_members' : len(client_cluster_list), 'accuracy' : np.mean(clients_accs), 'std' : np.std(clients_accs), 'members_heterogeneity':hetero_dict}
        else: 
            print('Cluster with no members !')
        with open('./results/{}.json'.format(config['output']), 'w') as json_file:
            json.dump(config, json_file, indent=4)


def plot_mnist(image,label):
    # Function to plot the mnist image
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray')
    plt.title(f'MNIST Digit: {label}')  # Add the label as the title
    plt.axis('off')  # Turn off axis
    plt.show()
    
import matplotlib.pyplot as plt

def plot_weights_heatmap(model, v_value=0.1):
    # Function that plot the heatmap of model's weight
    
    # Get the weights of the first and second layers
    first_layer_weights = model.fc1.weight.data.numpy()
    second_layer_weights = model.fc2.weight.data.numpy()

    # Create custom colormap
    cmap = 'coolwarm'

    # Plot the input weights of the first layer
    plt.figure(figsize=(10, 12))

    plt.subplot(2, 1, 1)
    plt.imshow(first_layer_weights, cmap=cmap, aspect='auto', vmin=-v_value, vmax=v_value)
    plt.title('Input Weights of the Model (fc1)')
    plt.xlabel('Input Pixel')
    plt.ylabel('Neuron in First Layer')
    plt.colorbar(label='Weight Value')

    # Plot the output weights of the second layer
    plt.subplot(2, 1, 2)
    plt.imshow(second_layer_weights, cmap=cmap, aspect='auto', vmin=-v_value, vmax=v_value)
    plt.title('Output Weights of the Model (fc2)')
    plt.xlabel('Neuron in First Layer')
    plt.ylabel('Output Neuron')
    plt.colorbar(label='Weight Value')

    plt.tight_layout()
    plt.show()
    
import torch
from sklearn.metrics.pairwise import cosine_similarity

def flatten_weights(model):
    # Flatten the model's weights and create a weight's vector (used for cosine similarity calculation)
    flattened_weights = []
    for param in model.parameters():
        flattened_weights.extend(param.data.flatten().numpy())
    return flattened_weights

def calculate_cosine_similarity(model1, model2):
    # Example usage:
    # Assuming you have two models named 'model1' and 'model2'
    # similarity = calculate_cosine_similarity(model1, model2)
    
    # Flatten the weights of both models
    weights1 = flatten_weights(model1)
    weights2 = flatten_weights(model2)
    
    # Calculate the cosine similarity between the flattened weights
    similarity = cosine_similarity([weights1], [weights2])[0][0]
    return similarity

