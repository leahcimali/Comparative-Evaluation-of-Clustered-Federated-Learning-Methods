import numpy as np 
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
            intracluster_distances = pairwise_distances(X[labels == np.array(i)], [centroids[i]])
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
        return
    

def calinski_harabasz_index(X, labels):
    try:
        k = len(np.unique(labels))
        centroids = [np.mean(X[labels == label], axis=0) for label in np.unique(labels)]
        within_cluster_dispersion = sum([np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k)])
        between_cluster_dispersion = sum([np.sum((centroids[i] - np.mean(X, axis=0)) ** 2) * len(X[labels == i]) for i in range(k)])
        return between_cluster_dispersion / within_cluster_dispersion * (len(X) - k) / (k - 1)
    except ValueError:
        return None

def calc_global_metrics(labels_true, labels_pred):

    """
    Calculate global metrics based on model weights
    """
    from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure, adjusted_mutual_info_score

    homogeneity_score, completness_score, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)

    ARI_score = adjusted_rand_score = adjusted_rand_score(labels_true, labels_pred)

    AMI_score = adjusted_mutual_info_score(labels_true, labels_pred) 
    
    dict_metrics = {"ARI": ARI_score, "AMI": AMI_score, "homogeneity": homogeneity_score, "completness": completness_score, "v_measure": v_measure}
    
    return dict_metrics



def report_CFL(list_clients, output_name):
    """
    Save results as a csv
    """
    import pandas as pd

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    df_results.to_csv("results/" + output_name + ".csv")

    return


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

