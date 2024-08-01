
from pandas import DataFrame
from pathlib import Path
   

def load_data():

    """
    Read results file and save as DataFrame 
    """
    import sys
    import pandas as pd
    
    try:    
        file_path = Path("results/") / Path(sys.argv[1])
        df_results = pd.read_csv(file_path)
    
    except Exception as e:
        
        exit("Error: Unable to open result file. Please make sure that the correct path is provided as argument and that the file is not corrupted.")
 

    return df_results, sys.argv[1]



def get_clusters(df_results):
    
    list_clusters = list(df_results['cluster_id'].unique())

    list_clusters = append_empty_clusters(list_clusters)

    return list_clusters


def append_empty_clusters(list_clusters):
    """
    Handle the situation where some clusters are empty by appending the clusters ID
    """

    list_clusters_int = [int(x) for x in list_clusters]
    
    max_clusters = max(list_clusters_int)
    
    for i in range(max_clusters + 1):
        
        if i not in list_clusters_int:
            
            list_clusters.append(str(i))

    return list_clusters



def get_z_nclients(x_het, y_clust, labels_heterogeneity):
    
    z_nclients = [0]* len(x_het)

    for i in range(len(z_nclients)):
        
        z_nclients[i] = len(df_results[(df_results['cluster_id'] == y_clust[i]) &
                                       (df_results['heterogeneity_class'] == labels_heterogeneity[x_het[i]])])

    return z_nclients



def plot_histogram_clusters(df_results: DataFrame, title):
    
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
    dz_nclients = get_z_nclients(x_heterogeneities, y_clusters, labels_heterogeneity) 

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
    ax.set_zticks(list(range(0,max(dz_nclients)+1,1)))
    
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(x_heterogeneities))]

    ax.bar3d(x_heterogeneities,y_clusters,z,dx,dy,dz_nclients, color=[colors[i] for i in x_heterogeneities])
    
    plt.title(title, fontdict=None, loc='center', pad=None)
    
    plt.savefig('results/plots/histogram_' + title + '.png')



def summarize_results(overwrite=False):

    from pathlib import Path

    pathlist = Path("results/").rglob('*.csv')
    
    for path in pathlist:
        list_params = path.stem.split('_')   
        
    dict_results = {"exp_type" : list_params[0], "dataset_type": list_params[1], "number_of_clients": list_params[2], "samples by_client": list_params[3]}
    print(dict_results)
    #TODO add metrics to the dictionary and save as csv


if __name__ == "__main__":
    
    df_results, filename = load_data()

    plot_histogram_clusters(df_results, filename)

    summarize_results()