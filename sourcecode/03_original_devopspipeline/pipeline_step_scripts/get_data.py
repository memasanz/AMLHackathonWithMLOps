
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
from sklearn import preprocessing
import numpy as np

#Parse input arguments
parser = argparse.ArgumentParser("Get data from and register in AML workspace")
parser.add_argument('--exp_raw_data', dest='exp_raw_data', required=True)

args, _ = parser.parse_known_args()
exp_raw_dataset = args.exp_raw_data

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

#Connect to default data store
ds = ws.get_default_datastore()

tab_data_set = Dataset.Tabular.from_delimited_files(path=(ds, 'diabetes-data/*.csv'))


raw_df = tab_data_set.to_pandas_dataframe()

#Make directory on mounted storage
os.makedirs(exp_raw_dataset, exist_ok=True)

#Upload modified dataframe
raw_df.to_csv(os.path.join(exp_raw_dataset, 'exp_raw_data.csv'), index=False)
