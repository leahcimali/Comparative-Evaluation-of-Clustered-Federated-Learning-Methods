import csv
import subprocess

# Path to your CSV file
csv_file = "exp_configs.csv"

# Read the second line from the CSV file
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Skip the header (if any) and the first row
    next(reader)  # Skipping the header
    row = next(reader)  # Reading the second row

    # Assigning CSV values to variables
    exp_type, dataset, nn_model, heterogeneity_type, num_clients, num_samples_by_label, num_clusters, centralized_epochs, federated_rounds, seed = row

    # Building the command
    command = [
        "python", "driver.py",
        "--exp_type", exp_type,
        "--dataset", dataset,

        "--nn_model", nn_model,
        "--heterogeneity_type", heterogeneity_type,
        "--num_clients", num_clients,
        "--num_samples_by_label", num_samples_by_label,
        "--num_clusters", num_clusters,
        "--centralized_epochs", centralized_epochs,
        "--federated_rounds", federated_rounds,
        "--seed", seed]

    # Run the command
    subprocess.run(command)
