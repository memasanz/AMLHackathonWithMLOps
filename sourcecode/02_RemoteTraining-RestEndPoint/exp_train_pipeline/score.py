
import json
import joblib
import numpy as np
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
import time
import os
import pandas as pd


#version 2
# Called when the service is loaded
def init():
    global model
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))
    # Get the path to the deployed model file and load it
    path = os.path.join(Model.get_model_path('diabetes_model_remote'))
    
    print(path)
    model = joblib.load(path)

    
    global inputs_dc, prediction_dc
    inputs_dc = ModelDataCollector("best_model", designation="inputs", feature_names=['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age'])
    prediction_dc = ModelDataCollector("best_model", designation="predictions", feature_names=["Diabetic"])



# Called when a request is received
def run(raw_data):
    try:
        # Get the input data as a numpy array
        #data = np.array(json.loads(raw_data)['data'])
        # Get a prediction from the model
        
        json_data = json.loads(raw_data)
        predictions = model.predict(json_data['data'])
        print ("Prediction created" + time.strftime("%H:%M:%S"))
        # Get the corresponding classname for each prediction (0 or 1)
        classnames = ['not-diabetic', 'diabetic']
        predicted_classes = []
        for prediction in predictions:
            val = int(prediction)
            predicted_classes.append(classnames[val])
        # Return the predictions as JSON
        
         # Log the input and output data to appinsights:
        info = {
            "input": raw_data,
            "output": predicted_classes
            }
        print(json.dumps(info))
        
        inputs_dc.collect(json_data['data']) #this call is saving our input data into Azure Blob
        prediction_dc.collect(predicted_classes) #this call is saving our prediction data into Azure Blob

        
        return json.dumps(predicted_classes)
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error
