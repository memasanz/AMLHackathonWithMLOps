
  
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse


#Parse input arguments
parser = argparse.ArgumentParser("Get Inferencing Data")
parser.add_argument('--inference_data_location', type=str, required=True)
parser.add_argument('--get_data_param_2', type=str, required=True)
parser.add_argument('--get_data_param_3', type=str, required=True)
parser.add_argument('--inferencing_dataset', dest='inferencing_dataset', required=True)

# Note: the get_data_param args below are included only as an example of argument passing.
# They are not consumed in the code sample shown here.
args, _ = parser.parse_known_args()

inference_data_location = args.inference_data_location
get_data_param_2 = args.get_data_param_2
get_data_param_3 = args.get_data_param_3
inferencing_dataset = args.inferencing_dataset

print(str(type(inferencing_dataset)))
print(inferencing_dataset)

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

# Get the default datastore
default_ds = ws.get_default_datastore()

tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, inference_data_location + '/*.csv'))

# Register the tabular dataset
try:
    tab_data_set = tab_data_set.register(workspace=ws, 
                                name= 'diabetes-tabular-dataset-raw',
                                description='diabetes data',
                                tags = {'format':'csv'},
                                create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
        print(ex)
        
df = tab_data_set.to_pandas_dataframe()

print('dataset shape = ' + str(df.shape))
print('saving inferencing data: ' + inferencing_dataset)

# Save dataset for consumption in next pipeline step
os.makedirs(inferencing_dataset, exist_ok=True)
df.to_csv(os.path.join(inferencing_dataset, 'inferencing_data.csv'), index=False)




