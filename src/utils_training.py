import torch
import torch.nn as nn
import torch.optim as optim

lr = 0.01


def lr_schedule(epoch,lr):
    decay_factor = 0.1
    if epoch % 10 == 0 and epoch != 0:
        return lr * decay_factor
    else:
        return lr


def run_cfl_server_side(model_server, list_clients, row_exp, output_name):

    from src.utils_fed import k_means_clustering
    from src.metrics import report_CFL
    from src.utils_logging import cprint
    import copy
    import torch 

    torch.manual_seed(row_exp['seed'])

    model_server = train_federated(model_server, list_clients, row_exp)
    
    model_server.clusters_models= {cluster_id: copy.deepcopy(model_server.model) for cluster_id in range(row_exp['num_clusters'])}
    
    setattr(model_server, 'num_clusters', row_exp['num_clusters'])

    cprint(f"Preparing to cluster with {len(list_clients)} clients")

    k_means_clustering(list_clients, row_exp['num_clusters'])

    cprint("Finished clustering")
    
    model_server = train_federated(model_server, list_clients, row_exp)

    cprint('Finished server-side CFL')

    list_clients = add_clients_accuracies(model_server, list_clients, row_exp)

    results = report_CFL(list_clients, output_name)

    return results

    
def run_cfl_client_side(model_server, list_clients, row_exp, output_name, init_cluster=True):

    from src.utils_fed import init_server_cluster, set_client_cluster, fedavg
    from src.metrics import report_CFL
    from src.utils_logging import cprint
    import torch

    torch.manual_seed(row_exp['seed'])

    if init_cluster == True : 
        
        init_server_cluster(model_server, list_clients, row_exp, p_expert_opinion=0.8)

    
    #print({c:{h:n
    #        for h in list(set([fc.heterogeneity_class for fc in list_clients])) 
    #          for n in [len([x for x in [fc for fc in list_clients if fc.cluster_id == c and fc.heterogeneity_class == h]])]}
    #            for c in range(row_exp['num_clusters'])}, lvl = "info")    
    
    for round in range(row_exp['federated_rounds']):

        for client in list_clients:

            client.model, _ = train_central(client.model, client.data_loader['train'], row_exp)

        fedavg(model_server, list_clients)
        
        set_client_cluster(model_server, list_clients, row_exp)

        #print([c.cluster_id for c in list_clients])
        #print(f"Round {round} clusters distributions")
        #print({c:{h:n
        #    for h in list(set([fc.heterogeneity_class for fc in list_clients])) 
        #      for n in [len([x for x in [fc for fc in list_clients if fc.cluster_id == c and fc.heterogeneity_class == h]])]}
        #        for c in range(row_exp['num_clusters'])})

    cprint("Finished client-side CFL")

    list_clients = add_clients_accuracies(model_server, list_clients, row_exp)

    results = report_CFL(list_clients, output_name)
    return results
    

def run_benchmark(list_clients, row_exp, output_name, main_model):
    
    import pandas as pd    
    from src.models import SimpleLinear, SimpleConv

    main_model = SimpleLinear() if 'mnist' in row_exp['dataset'] else SimpleConv()
    
    list_exps = ['global-centralized', 'global-federated', 'pers-centralized', 'pers-federated'] 
    list_heterogeneities = list(set(client.heterogeneity_class for client in list_clients))

  
    for training_type in list_exps: 
        
        curr_model = main_model if 'federated' in training_type else main_model

        if 'pers' in training_type:

            for heterogeneity_class in list_heterogeneities:
                
                list_clients_filtered = [client for client in list_clients if client.heterogeneity_class == heterogeneity_class]

                model_server, test_loader = train_benchmark(list_clients_filtered, row_exp, curr_model, training_type)

                test_benchmark(model_server, list_clients_filtered, test_loader, row_exp)
       
        elif 'global' in training_type:
                
            model_server, test_loader = train_benchmark(list_clients, row_exp, curr_model, training_type)

            test_benchmark(model_server, list_clients, test_loader, row_exp)

        df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])

        df_results.to_csv(path_or_buf=  "results/" + output_name.replace("benchmark", "benchmark-" + training_type) + ".csv")

    return

def train_benchmark(list_clients, row_exp, main_model, training_type="centralized"):
        
        from src.utils_training import train_model
        from src.utils_data import centralize_data
        import copy

        train_loader, test_loader = centralize_data(list_clients)
    
        if "federated" in training_type:
            model_server = copy.deepcopy(main_model)
            model_trained = train_model(model_server, None, list_clients, row_exp)
        
        else:
            model_trained = train_model(main_model, train_loader, list_clients, row_exp) 
        
        return model_trained, test_loader



def test_benchmark(model_trained, list_clients, test_loader, row_exp):    
         
    from src.utils_training import test_model
    
    global_acc = test_model(model_trained, test_loader, row_exp) 
                     
    for client in list_clients : 
        
        #client_acc = test_model(model_trained, client.data_loader['test'])*100

        setattr(client, 'accuracy', global_acc)
    
    return global_acc



def train_model(model_server, train_loader, list_clients, row_exp):
    
    if not train_loader:
        trained_obj = train_federated(model_server, list_clients, row_exp)
        trained_model = trained_obj.model
    
    else:
        trained_model, _ = train_central(model_server, train_loader, row_exp)
    
    return trained_model



def train_federated(main_model, list_clients, row_exp):
    """
    Controler function to launch federated learning

    Parameters
    ----------
    main_model:
        Define the central node model :
    """
    import numpy as np
    from src.utils_fed import send_server_model_to_client, fedavg
    
    
    for i in range(0, row_exp['federated_rounds']):

        accs = []

        send_server_model_to_client(list_clients, main_model)

        for client in list_clients:

            client.model, curr_acc = train_central(client.model, client.data_loader['train'], row_exp)

            accs.append(curr_acc)

        print(f"accuracy at round {i}:", np.mean(accs))

        fedavg(main_model, list_clients)

    return main_model


def train_central(main_model, train_loader, row_exp, lr_scheduler=None):

    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam
    optimizer = optimizer(main_model.parameters(), lr=0.01) 
    torch.manual_seed(row_exp['seed'])

    main_model.train()
    for epoch in range(row_exp['centralized_epochs']):
          
        running_loss = total = correct = 0

        # Apply learning rate decay if lr_scheduler is provided
        if lr_scheduler is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scheduler(epoch, param_group['lr'])

        # Iterate over the training dataset
        for inputs, labels in train_loader:

            if (row_exp['dataset'] == "cifar10"):
                inputs = inputs.permute(0,3,1,2)

            optimizer.zero_grad()  # Zero the gradients
            outputs = main_model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * inputs.size(0)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy on the test set
    accuracy = correct / total

    main_model.eval()  # Set the model to evaluation mode        

    return main_model, accuracy

def loss_calculation(model, train_loader, row_exp): 
    import torch
    import torch.nn as nn

    # Assuming you have a PyTorch model named 'model' and its training data loader named 'train_loader'

    # Define your loss function
    criterion = nn.CrossEntropyLoss()  # Example, adjust based on your task

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to accumulate loss and total number of samples
    total_loss = 0.0
    total_samples = 0

    torch.manual_seed(row_exp['seed'])
    # Iterate through the training data loader
    with torch.no_grad():
        for inputs, targets in train_loader:

            if row_exp['dataset'] == "cifar10":
                inputs = inputs.permute(0,3,1,2)

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Accumulate the loss and the total number of samples
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    # Calculate the average loss
    average_loss = total_loss / total_samples

    return average_loss

def test_model(model, test_loader, row_exp):
    criterion = nn.CrossEntropyLoss()

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track accuracy
    correct = 0
    total = 0
    test_loss = 0.0


    torch.manual_seed(row_exp['seed'])
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate over the test dataset
        for inputs, labels in test_loader:

            if row_exp['dataset'] == "cifar10":
                inputs = inputs.permute(0,3,1,2)

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)

    # Calculate accuracy on the test set
    accuracy = correct / total

    # Print the test loss and accuracy
    return accuracy


def add_clients_accuracies(model_server, list_clients, row_exp):
    

    for client in list_clients : 
        acc = test_model(model_server.clusters_models[client.cluster_id], client.data_loader['test'], row_exp)*100
        setattr(client, 'accuracy', acc)

    return list_clients
