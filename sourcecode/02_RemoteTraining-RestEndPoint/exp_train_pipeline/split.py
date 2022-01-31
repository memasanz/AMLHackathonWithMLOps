
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from numpy.random import seed

#Parse input arguments
parser = argparse.ArgumentParser("Split raw data into train/test and scale appropriately")

parser.add_argument('--exp_training_data', dest='exp_training_data', required=True)
parser.add_argument('--exp_testing_data', dest='exp_testing_data', required=True)


args, _ = parser.parse_known_args()
exp_training_data = args.exp_training_data
exp_testing_data = args.exp_testing_data


#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
raw_datset = current_run.input_datasets['Exp_Raw_Data']
raw_df = raw_datset.to_pandas_dataframe()


for col in raw_df.columns:
    missing = raw_df[col].isnull()
    num_missing = np.sum(missing)
    if num_missing > 0:  
        raw_df['quality_{}_ismissing'.format(col)] = missing


print(raw_df.columns)

#Split data into training set and test set
df_train, df_test = train_test_split(raw_df, test_size=0.3, random_state=0)





# Save train data to both train and test (reflects the usage pattern in this sample. Note: test/train sets are typically distinct data).
os.makedirs(exp_training_data, exist_ok=True)
os.makedirs(exp_testing_data, exist_ok=True)

df_train.to_csv(os.path.join(exp_training_data, 'exp_training_data.csv'), index=False)
df_test.to_csv(os.path.join(exp_testing_data, 'exp_testing_data.csv'), index=False)
