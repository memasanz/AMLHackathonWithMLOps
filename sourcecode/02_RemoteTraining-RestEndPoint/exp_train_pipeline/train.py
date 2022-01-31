
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse
import shutil

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


import matplotlib.pyplot as plt
import joblib
from numpy.random import seed


#Parse input arguments
parser = argparse.ArgumentParser("Train Logistic Regression model")
parser.add_argument('--model_file_output', dest='model_file_output', required=True)


args, _ = parser.parse_known_args()
model_file_output = args.model_file_output

def converttypes(df):
    cols = df.columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors = 'coerce')

    print('data types')
    print(df.dtypes)
    return df


#Get current run
run = Run.get_context()

#Get associated AML workspace
ws = run.experiment.workspace

# Read input dataset to pandas dataframe
X_train_dataset = run.input_datasets['Exp_Training_Data'].to_pandas_dataframe()
X_test_dataset = run.input_datasets['Exp_Testing_Data'].to_pandas_dataframe()

X_train_dataset = converttypes(X_train_dataset)
X_test_dataset = converttypes(X_test_dataset)


print(type(X_train_dataset))

# Separate features and labels
X_train, y_train = X_train_dataset[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, X_train_dataset['Diabetic'].values
X_test, y_test   = X_test_dataset[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, X_test_dataset['Diabetic'].values

print('***********')
print(type(X_train))
print(type(X_test))
print(type(y_train[0]))
print(type(y_test[0]))
print('**********')
# Set regularization hyperparameter
reg = 0.01

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

print('y_hat[0] is if type=:' + str(type(y_hat[0])))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

run.parent.log(name='AUC', value=np.float(auc))
run.parent.log(name='Accuracy', value=np.float(acc))

# Save the trained model in the outputs folder
os.makedirs('./outputs', exist_ok=True)
joblib.dump(value=model, filename='./outputs/diabetes_model_remote.pkl')

os.makedirs(model_file_output, exist_ok=True)

shutil.copyfile('./outputs/diabetes_model_remote.pkl', os.path.join(model_file_output, 'diabetes_model_remote.pkl'))
