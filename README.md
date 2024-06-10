Code for the paper : 'A Comparison of Data Heterogeneities in Clustered Federated Learning' 
submited to 'The 2nd IEEE International Conference on Federated Learning Technologies and Applications (FLTA24), VALENCIA, SPAIN' 

Link for the results use in the article : https://drive.google.com/file/d/1dqdgmPpZFEFa0Cx7Fg03Z-S_XCZRdNOV/view?usp=sharing 

Extract the content of 'results.rar' inside a directory named 'results'. 

Figure of results can by found by heterogeneities inside the Notebook directory. 
Not all figure where use in the article.

Each training setup have a different training file : 
- TrainCentral.py : Will trained a centralized model, a federated model and both personalization by heterogeneity of centralized and federated models (refere to article for more details)
- trainCFLclient.py : will trained client-side CFL models 
- trainCFLserver.py : will trained server-side CFL models

Each setup come with its own json config file where each dictionary inside the json correspond to one experimentation that will be trained iteratively.

*You can change the heterogeneity by the following to simulate the different case of heterogeneity : 

"concept_shift_on_features", "concept_shift_on_labels", "labels_distribution_skew", "labels_distribution_skew_balancing", "labels_distribution_skew_upsampled", "features_distribution_skew", "quantity_skew"

Each experiment will output:  
- a .txt file that details the training process
- a .json file with each setup accuracy
- .pth files of trained models used to visualize cluster and T-SNE of model weight (see Notebooks)

  
