
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath

import os
import argparse
import shutil

parser = argparse.ArgumentParser("Evaluate model and register if more performant")
parser.add_argument('--exp_trained_model_pipeline_data', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--build_number', type=str, required=True)



args, _ = parser.parse_known_args()

exp_trained_model_pipeline_data = args.exp_trained_model_pipeline_data

model_name = args.model_name
build_number = args.build_number


#Get current run
run = Run.get_context()

#Get associated AML workspace
ws = run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

#Get metrics associated with current parent run
metrics = run.get_metrics()

print('current run metrics')
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')


print('parent run metrics')
#Get metrics associated with current parent run
metrics = run.parent.get_metrics()

for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')

current_model_AUC = float(metrics['AUC'])
current_model_accuracy = float(metrics['Accuracy'])

# Get current model from workspace
#model_name = 'diabetes_model'
model_description = 'Diabetes model'
model_list = Model.list(ws, name=model_name, latest=True)
first_registration = len(model_list)==0

build_id = os.getenv("BUILD_BUILDID", default='1')

print('build_id =' + build_id)
print('build_number = ' +  build_number)

updated_tags = {'AUC': current_model_AUC, 'build_number' : build_number}

print('updated tags')
print(updated_tags)

# Copy autoencoder training outputs to relative path for registration
relative_model_path = 'model_files'
run.upload_folder(name=relative_model_path, path=exp_trained_model_pipeline_data)


#If no model exists register the current model
if first_registration:
    print('First model registration.')
    #model = run.register_model(model_name, model_path='model_files', description=model_description, model_framework='sklearn', model_framework_version=tf.__version__, tags=updated_tags, datasets=formatted_datasets, sample_input_dataset = training_dataset)
    run.register_model(model_path=relative_model_path, model_name=model_name,
                   tags=updated_tags,
                   properties={'AUC': current_model_AUC, 'BUILD_ID' : build_id, 'BUILD_NUMBER' : build_number})
else:
    #If a model has been registered previously, check to see if current model 
    #performs better. If so, register it.
    print(dir(model_list[0]))
    if float(model_list[0].tags['AUC']) < current_model_AUC:
        print('New model performs better than existing model. Register it.')
        #model = run.register_model(model_name, model_path='model_files', description=model_description, model_framework='Tensorflow/Keras', model_framework_version=tf.__version__, tags=updated_tags, datasets=formatted_datasets, sample_input_dataset = training_dataset)
        run.register_model(model_path=relative_model_path, model_name='diabetes_model',
                   tags=updated_tags,
                   properties={'AUC': current_model_AUC, 'Accuracy': current_model_accuracy, 'BUILD_ID' : build_id, 'BUILD_NUMBER' : build_number})
                   
 
        
        
    else:
        print('New model does not perform better than existing model. Cancel run.')
        run.cancel()
