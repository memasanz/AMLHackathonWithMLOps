
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse


# Parse input arguments
# parser = argparse.ArgumentParser("Score Inferencing Data")
# parser.add_argument('--user_param', type=str, required=True)


# args, _ = parser.parse_known_args()
# user = args.user_param



# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Get default datastore
ds = ws.get_default_datastore()

# Get inferencing dataset
scored_dataset = current_run.input_datasets['diabetes_scored_data']
scored_data_df = scored_dataset.to_pandas_dataframe()

# Save dataset to ./outputs dir
os.makedirs('./outputs', exist_ok=True)
scored_data_df.to_csv(os.path.join('outputs', 'scored_data.csv'), index=False)


# # Save dataset to ./outputs dir
# os.makedirs('./outputs', exist_ok=True)
# scored_data_df.to_parquet(os.path.join('outputs', 'scored_data.parquet'), index=False)

