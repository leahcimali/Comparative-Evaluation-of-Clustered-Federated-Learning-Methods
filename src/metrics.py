
def calc_global_metrics(labels_true: list, labels_pred: list) -> dict:

    """ Calculate global metrics based on model weights

    Args:
        labels_true : list
            list of ground truth labels
        labels_pred : list
            list of predicted labels to compare with ground truth
    Returns:
        a dictionary containing the following metrics:
         'ARI', 'AMI', 'hom': homogeneity_score, 'cmpt': completness score, 'vm': v-measure
    """

    from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure, adjusted_mutual_info_score

    homogeneity_score, completness_score, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)

    ARI_score = adjusted_rand_score = adjusted_rand_score(labels_true, labels_pred)

    AMI_score = adjusted_mutual_info_score(labels_true, labels_pred) 
    
    dict_metrics = {"ARI": ARI_score, "AMI": AMI_score, "hom": homogeneity_score, "cmplt": completness_score, "vm": v_measure}
    
    return dict_metrics



def report_CFL(list_clients: list, output_name: str) -> None:
    """ Save dataframe of results in results/<output_name>.csv

    Args:
        list_clients: list
            the list of Client objects to save to file
        output_name : str
            the desired name of the output file
    """

    import pandas as pd

    df_results = pd.DataFrame.from_records([c.to_dict() for c in list_clients])
    
    df_results.to_csv("results/" + output_name + ".csv")

    return