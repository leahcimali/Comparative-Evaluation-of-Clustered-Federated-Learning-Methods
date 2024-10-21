#### Code for the paper: *Comparative Evaluation of Clustered Federated Learning Methods*

##### Submited to 'The 2nd IEEE International Conference on Federated Learning Technologies and Applications (FLTA24), VALENCIA, SPAIN' 

1. To reproduce the results in the paper run `driver.py` with the parameters in `exp_configs.csv`

2. Each experiment will output a `.csv` file with the resuting metrics

3. Histogram plots and a summary table of various experiments can be obtained running `src/utils_results.py`
  
To use driver.py use the following parameters : 

`python driver.py --exp_type --dataset --heterogeneity_type  --num_clients --num_samples_by_label --num_clusters --centralized_epochs --federated_rounds --seed ` 

To run all experiments in exp_config.csv user `run_exp.py`. 

Once all experiments are done, to get results run `src/utils_results.src`.