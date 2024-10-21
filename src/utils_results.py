
from pandas import DataFrame
from pathlib import Path
from torch import tensor


def save_histograms() -> None:

    """Read csv files found in 'results/' and generates and saves histogram plots of clients assignemnts

    Raises : 
        Warning when the csv file is not of the expected format (code generated results csv)
    """

    import pandas as pd
    
    pathlist = Path("results/").rglob('*.csv') 
    
    for file_path in pathlist:

        if 'benchmark' not in str(file_path):
            
            try:

                df_results = pd.read_csv(file_path)

                plot_histogram_clusters(df_results, file_path.stem)
    
            except Exception as e:
        
                print(f"Error: Unable to open result file {file_path}.",e)
            
                continue
    return



def get_clusters(df_results : DataFrame) -> list:
    
    """ Function to returns a list of clusters ranging from 0 to max_cluster (uses: append_empty_clusters())
    """

    list_clusters = list(df_results['cluster_id'].unique())

    list_clusters = append_empty_clusters(list_clusters)

    return list_clusters


def append_empty_clusters(list_clusters : list) -> list:
    """
    Utility function for ``get_clusters'' to handle the situation where some clusters are empty by appending the clusters ID
    
    Arguments:
        list_clusters: List of clusters with clients

    Returns:
        List of clusters with or without clients
    """

    list_clusters_int = [int(x) for x in list_clusters]
    
    max_clusters = max(list_clusters_int)
    
    for i in range(max_clusters + 1):
        
        if i not in list_clusters_int:
            
            list_clusters.append(str(i))

    return list_clusters



def get_z_nclients(df_results : dict, x_het : list, y_clust : list, labels_heterogeneity : list) -> list:
    
    """ Returns the number of clients associated with a given heterogeneity class for each cluster"""

    z_nclients = [0]* len(x_het)

    for i in range(len(z_nclients)):
        
        z_nclients[i] = len(df_results[(df_results['cluster_id'] == y_clust[i]) &
                                       (df_results['heterogeneity_class'] == labels_heterogeneity[x_het[i]])])

    return z_nclients



def plot_img(img : tensor) -> None:

    """Utility function to plot an image of any shape"""

    from torchvision import transforms
    import matplotlib.pyplot as plt

    plt.imshow(transforms.ToPILImage()(img))



def plot_histogram_clusters(df_results: DataFrame, title : str) -> None:
    
    """ Function to create 3D Histograms of clients to cluster assignments showing client's heterogeneity class 

    Arguments:
        
        df_results : DataFrame containing all parameters from the resulting csv files
        
        title : The plot title. The image is saved in results/plots/histogram_' + title + '.png'
    """
    
    import matplotlib.pyplot as plt
    import numpy as np 
        
    labels_heterogeneity = list(df_results['heterogeneity_class'].unique())

    bar_width = bar_depth = 0.5

    n_clusters =  len(get_clusters(df_results))
    n_heterogeneities = len(labels_heterogeneity)

    # bar coordinates lists
    x_heterogeneities = np.repeat(list(range(n_heterogeneities)), n_clusters)  
    y_clusters = [int(x) for x in get_clusters(df_results)] * n_heterogeneities   
    z = [0]*len(x_heterogeneities) 

    # dimensions for each bar (note we use the z dimension for the number of clients)
    dx = [bar_width] * len(x_heterogeneities) 
    dy = [bar_depth] * len(y_clusters) 
    dz_nclients = get_z_nclients(df_results, x_heterogeneities, y_clusters, labels_heterogeneity) 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    list_clusters = [x for x in 'abcdefghijklmnopqrstuvwxyz'][:n_clusters]

    ticksy = np.arange(0.25, len(list_clusters), 1)
    ticksx = np.arange(0.25, len(labels_heterogeneity), 1)

    plt.xticks(ticksx, labels_heterogeneity)
    plt.yticks(ticksy, list_clusters)


    plt.ylabel('Cluster ID')
    plt.xlabel('Heterogeneity Class')
    
    ax.set_zlabel('Number of Clients')
    
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, len(x_heterogeneities))]

    ax.bar3d(x_heterogeneities,y_clusters,z,dx,dy,dz_nclients, color=[colors[i] for i in x_heterogeneities])
    
    plt.title(title, fontdict=None, loc='center', pad=None)
    
    plt.savefig('results/plots/histogram_' + title + '.png')
    
    plt.close()

    return


def normalize_results(results_accuracy : float, results_std : float) -> int:

    """Utility function to convert float accuracy and std to percentage """
    
    if results_accuracy < 1:
        
        results_accuracy = results_accuracy * 100
    
        results_std = results_std * 100

    return results_accuracy, results_std


def summarize_results() -> None:

    """ Creates results summary of all the results files under "results/summarized_results.csv"""
        
    from pathlib import Path
    import pandas as pd
    from numpy import mean, std

    from metrics import calc_global_metrics


    pathlist = Path("results/").rglob('*.csv') 
    
    list_results = []

    for path in pathlist:
        
        if 'summarized_results' not in str(path):
            
            df_exp_results = pd.read_csv(path)

            results_accuracy = mean(list(df_exp_results['accuracy'])) 
            results_std = std(list(df_exp_results['accuracy']))

            results_accuracy, results_std = normalize_results(results_accuracy, results_std)

            accuracy =  "{:.2f}".format(results_accuracy) + " \\pm " +   "{:.2f}".format(results_std)

            list_params = path.stem.split('_')      

            dict_exp_results = {"exp_type" : list_params[0], "dataset": list_params[1], "nn_model" : list_params[2], "dataset_type": list_params[3], "number_of_clients": list_params[4],
                                    "samples by_client": list_params[5], "num_clusters": list_params[6], "centralized_epochs": list_params[7],
                                    "federated_rounds": list_params[8],"accuracy": accuracy}

            try:
                
                labels_true = list(df_exp_results['heterogeneity_class'])
                labels_pred = list(df_exp_results['cluster_id'])
                
                dict_metrics = calc_global_metrics(labels_true=labels_true, labels_pred=labels_pred)
            
                dict_exp_results.update(dict_metrics)
                
            
            except:

                print(f"Warning: Could not calculate cluster metrics for file {path}")
                
    
            list_results.append(dict_exp_results)
            
    df_results = pd.DataFrame(list_results)
    
    df_results.sort_values(['dataset_type',  'dataset', 'exp_type', 'number_of_clients'], inplace=True)
    
    df_results = df_results[['exp_type', 'dataset', 'num_clusters', 'dataset_type', "accuracy", "ARI", "AMI", "hom", "cmplt", "vm"]]
    
    df_results.to_csv("results/summarized_results.csv", float_format='%.2f', index=False, na_rep="n/a")

    return




if __name__ == "__main__":
    
    save_histograms()

    summarize_results()