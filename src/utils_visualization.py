
import pandas as pd
from pandas import DataFrame
import json
from pathlib import Path

def get_results(file_path: Path) -> DataFrame:

    """
    Read json results files and convert to Dataframe
    """

    with open(file_path, 'r') as f:

        global_metrics = ['silhouette', 'avg_intra_dist', 'intra_dist_var', 'duhn_index', 'davies_bouldin_index']
        
        dict_results = json.load(f)
    
        dict_global_metrics = {key: dict_results[key] for key in global_metrics}
        
        dict_clusters = {key: dict_results[key] for key in dict_results.keys() if key not in global_metrics}

        df_global_metrics = pd.json_normalize(dict_global_metrics)

        df_clusters = reduce_columns_clusters_(pd.json_normalize(dict_clusters))

        df_clusters = reduce_columns_heterogeneities(df_clusters)

        df_clusters.loc[:, 'filename'] = file_path.stem

    return df_global_metrics, df_clusters


def reduce_columns_clusters_(df_clusters: DataFrame) -> DataFrame:
    """
    Returns a DataFrame with one row per cluster
    """
    
    list_clusters = get_clusters(df_clusters)

    list_dfs = []

    for c in list_clusters:

        df_tmp = df_clusters[[col for col in df_clusters.columns if 'Cluster ' + c + '.' in col]]
        
        df_tmp = df_tmp.rename(columns= dict(zip(df_tmp.columns, df_tmp.columns.str.replace('Cluster ' + c + '.', '', regex=False))))

        df_tmp = df_tmp.assign(cluster = pd.Series(c))
    
        list_dfs.append(df_tmp) 
 
    df_clusters = pd.concat(list_dfs)

    df_clusters.drop_duplicates(inplace=True)

    return df_clusters 


def reduce_columns_heterogeneities(df_clusters):

    columns_heterogeneity = [col for col in df_clusters.columns if 'members_heterogeneity.' in col]

    labels_heterogeneity = [lbl.replace('members_heterogeneity.', '') for lbl in columns_heterogeneity]
   
    df_temp = df_clusters[columns_heterogeneity].copy().fillna(0)
   
    df_clusters.loc[:,'members'] = df_temp.astype(str).agg(' '.join, axis=1).apply(lambda x: [int(float(val)) for val in x.split(' ')])

    df_clusters.loc[:, 'members'] = df_clusters['members'].apply(lambda x: sum([x[i]*[labels_heterogeneity[i]] for i,_ in enumerate(x) ],[]))

    return df_clusters    
    
    

def load_data(results_dir: Path):

    """
    Read results files and execute analysis on each 
    """

    df_results = []

    df_experiments = pd.read_csv("exp_configs.csv")
    
    for _, row_exp in df_experiments.iterrows():

        file_path = results_dir / Path(row_exp['output'] + ".json")
        
        if file_path.exists():
            
            df_gloabl_metrics, df_clusters = get_results(file_path)

            df_results.append(df_clusters)
    
    return df_results
            # execute analyses

def get_clusters(df):
    
    list_clusters = []

    if 'cluster' in df.columns:
        
        list_clusters = df['cluster'].unique()
    
    else:

        for col_name in df.columns:
            
            if "Cluster" in col_name:

                list_clusters.append(col_name.split('.')[0])
        
        list_clusters = list(set(list_clusters))
        list_clusters = [x.split(' ')[1] for x in list_clusters]

    return list_clusters


def replace_by_occurence(labels_list, ref_list):
    new_list = []
    
    for i,val in enumerate(ref_list):
        for _ in range(val):
            new_list.append(labels_list[i])
    return new_list


def get_z_nclients(x, y, labels_heterogeneity):
    
    z_nclients = []
    for i in range(len(x)):
        y_i = y[i]
        x_i = x[i]
        
        z_i = df_clusters[df_clusters['cluster']== str(y_i)]['members'][0].count(labels_heterogeneity[x_i])
        z_nclients.append(z_i)
    return z_nclients



def histogram_clusters_dist(df_clusters: DataFrame):
    
    import matplotlib.pyplot as plt
    import numpy as np 
        
    columns_heterogeneity = [col for col in df_clusters.columns if 'members_heterogeneity.' in col]
    labels_heterogeneity = [lbl.replace('members_heterogeneity.', '') for lbl in columns_heterogeneity]

    bar_width = bar_depth = 0.5

    n_clusters = len(get_clusters(df_clusters))
    n_heterogeneities = len(labels_heterogeneity)

    # bar coordinates lists
    x_heterogeneities = np.repeat(list(range(n_heterogeneities)), n_clusters)  
    y_clusters = [int(x) for x in get_clusters(df_clusters)] * n_heterogeneities   
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

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(x_heterogeneities))]

    ax.bar3d(x_heterogeneities,y_clusters,z,dx,dy,dz_nclients, color=[colors[i] for i in x_heterogeneities])
    plt.show()


if __name__ == "__main__":
    
    df_results = load_data(Path("./results"))
    
    for df_clusters in df_results:

        histogram_clusters_dist(df_clusters)
