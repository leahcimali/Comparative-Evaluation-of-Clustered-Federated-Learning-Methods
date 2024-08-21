
def calc_global_metrics(labels_true, labels_pred):

    """
    Calculate global metrics based on model weights
    """
    from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure, adjusted_mutual_info_score

    homogeneity_score, completness_score, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)

    ARI_score = adjusted_rand_score = adjusted_rand_score(labels_true, labels_pred)

    AMI_score = adjusted_mutual_info_score(labels_true, labels_pred) 
    
    dict_metrics = {"ARI": ARI_score, "AMI": AMI_score, "hom": homogeneity_score, "cmplt": completness_score, "vm": v_measure}
    
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
    


