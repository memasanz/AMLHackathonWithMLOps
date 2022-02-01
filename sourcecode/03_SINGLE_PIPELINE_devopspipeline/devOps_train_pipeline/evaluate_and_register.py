
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath

import joblib
import os
import argparse
import shutil
import pandas as pd

from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient
from azureml.interpret.scoring.scoring_explainer import LinearScoringExplainer, save

from azureml.core.model import InferenceConfig
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import Webservice, AksWebservice


parser = argparse.ArgumentParser("Evaluate model and register if more performant")

parser.add_argument('--model_file', type=str, required=True)
parser.add_argument('--deploy_file_output', type=str, help='File passing in pipeline to deploy')

args, _ = parser.parse_known_args()

deploy_file = args.deploy_file_output
model_file = args.model_file

def converttypes(df):
    cols = df.columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors = 'coerce')

    print('data types')
    print(df.dtypes)
    return df

def model_explain():
    #load trinning data
    X_train_dataset = run.input_datasets['Exp_Training_Data'].to_pandas_dataframe()
    X_test_dataset = run.input_datasets['Exp_Testing_Data'].to_pandas_dataframe()
    
    X_test_dataset = converttypes(X_test_dataset)
    X_train_dataset = converttypes(X_train_dataset)
    
    X_train, y_train = X_train_dataset[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, X_train_dataset['Diabetic'].values
    X_test, y_test   = X_test_dataset[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, X_test_dataset['Diabetic'].values

    
    #load the model
    model_list = Model.list(ws, name=model_name, latest=True)
    model_path = model_list[0].download(exist_ok=True)
    model = joblib.load(model_path)

    #https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/explain-model/azure-integration/scoring-time/train_explain.py
    # create an explanation client to store the explanation (contrib API)
    client = ExplanationClient.from_run(run)

    # create an explainer to validate or debug the model
    tabular_explainer = TabularExplainer(model,
                                         initialization_examples=X_train,
                                         features=['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age'],
                                         classes=[0, 1])
                                         #transformations=transformations)

    # explain overall model predictions (global explanation)
    # passing in test dataset for evaluation examples - note it must be a representative sample of the original data
    # more data (e.g. x_train) will likely lead to higher accuracy, but at a time cost
    global_explanation = tabular_explainer.explain_global(X_test)

    # uploading model explanation data for storage or visualization
    comment = 'Global explanation on classification model trained'
    client.upload_model_explanation(global_explanation, comment=comment, model_id=model_reg.id)



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
model_name = 'diabetes_model_remote'
model_description = 'Diabetes model remote'
model_list = Model.list(ws, name=model_name, latest=True)
first_registration = len(model_list)==0

updated_tags = {'AUC': current_model_AUC}

print('updated tags')
print(updated_tags)

# Copy  training outputs to relative path for registration



relative_model_path = 'outputs'
run.upload_folder(name=relative_model_path, path=model_file)



#If no model exists register the current model
if first_registration:
    print('First model registration.')
    model_reg = run.register_model(model_path='outputs/diabetes_model_remote.pkl', model_name=model_name,
                   tags=updated_tags,
                   properties={'AUC': current_model_AUC})

    #model_explain()
else:
    #If a model has been registered previously, check to see if current model 
    #performs better. If so, register it.
    print(dir(model_list[0]))
    if float(model_list[0].tags['AUC']) < current_model_AUC:
        print('New model performs better than existing model. Register it.')

        model_reg = run.register_model(model_path='outputs/diabetes_model_remote.pkl', model_name=model_name,
                   tags=updated_tags,
                   properties={'AUC': current_model_AUC, 'Accuracy': current_model_accuracy})

        #model_explain()
        
        # Output accuracy to file
        with open(deploy_file, 'w+') as f:
            f.write(('deploy'))
    
    else:
        print('New model does not perform better than existing model. Cancel run.')
        
        with open(deploy_file, 'w+') as f:
            f.write(('no deployment'))
            
        run.cancel()
