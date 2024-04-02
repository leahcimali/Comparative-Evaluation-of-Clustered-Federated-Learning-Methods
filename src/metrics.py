from src.utils_training import test_model
import numpy as np 
def report_CFL(my_server, client_list): 
    for cluster_id in range(my_server.num_clusters):
        print('For cluster ', cluster_id, ' : ')
        client_cluster_list = [client for client in client_list if client.cluster_id == cluster_id]
        clients_accs = []
        for client in client_cluster_list : 
            print('Client ID : ', client.id)
            print('Client rotation : ', client.rotation)
            print('Cluster model accuracy on client test data')
            acc = test_model(my_server.clusters_models[cluster_id], client.data_loader['test'])*100
            clients_accs.append(acc)
            print(acc)
        print("Cluster accuracy : ", np.mean(clients_accs))
        print("Cluster accuracy std : ", np.std(clients_accs))
        cluster_rot = [client.rotation for client in client_cluster_list]
        values, counts = np.unique(cluster_rot, return_counts = True)
        print('Cluster rotations : ' , {values[i] : counts[i] for i in range(len(values))} )