#!/usr/bin/env python
# coding: utf-8

# ## MLOps with Azure ML Pipelines
# 
# ML Pipeline - Training & Registration.  ML Pipelines can help you to build, optimize and manage your machine learning workflow. 
# 
# ML Pipelines encapsulate a workflow for a machine learning task.  Tasks often include:
# - Data Prep
# - Training 
# - Publishing Models
# - Deployment of Models
# 
# First we will set some key variables to be leveraged inside the notebook

# In[1]:


registered_env_name = "experiment_env"
experiment_folder = 'devOps_train_pipeline'
dataset_prefix_name = 'exp'
cluster_name = "mm-cluster"


# Import required packages

# In[2]:


# Import required packages
from azureml.core import Workspace, Experiment, Datastore, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute, DataFactoryCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineParameter, PipelineData
from azureml.data.output_dataset_config import OutputTabularDatasetConfig, OutputDatasetConfig, OutputFileDatasetConfig
from azureml.data.datapath import DataPath
from azureml.data.data_reference import DataReference
from azureml.data.sql_data_reference import SqlDataReference
from azureml.pipeline.steps import DataTransferStep
import logging


# ### Connect to the workspace and create a cluster for running the AML Pipeline
# 
# Connect to the AML workspace and the default datastore. To run an AML Pipeline, we will want to create compute if a compute cluster is not already available

# In[3]:


# Connect to AML Workspace
ws = Workspace.from_config('./' + experiment_folder + '/config-dev.json')

# Get the default datastore
default_ds = ws.get_default_datastore()

#Select AML Compute Cluster
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


try:
    # Check for existing compute target
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)


# ## Create Run configuration
# 
# The RunConfiguration defines the environment used across all the python steps.  There are a variety of ways of setting up an environment.  An environment holds the required python packages needed for your code to execute on a compute cluster

# In[4]:





# In[5]:


#conda_yml_file = '../configuration/environment.yml'
conda_yml_file = './'+ experiment_folder+ '/environment.yml'


# In[6]:





# In[7]:


# Create a Python environment for the experiment (from a .yml file)
env = Environment.from_conda_specification("experiment_env", conda_yml_file)


run_config = RunConfiguration()
run_config.docker.use_docker = True
run_config.environment = env
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE


# In[8]:


registered_env_name


# In[9]:


from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration

# Create a Python environment for the experiment (from a .yml file)
experiment_env = Environment.from_conda_specification(registered_env_name, conda_yml_file)

# Register the environment 
experiment_env.register(workspace=ws)
registered_env = Environment.get(ws, registered_env_name)

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = pipeline_cluster

# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

print ("Run configuration created.")


# ## Define Output datasets
# 
# 
# The **OutputFileDatasetConfig** object is a special kind of data reference that is used for interim storage locations that can be passed between pipeline steps, so you'll create one and use at as the output for the first step and the input for the second step. Note that you need to pass it as a script argument so your code can access the datastore location referenced by the data reference. 
# 
# Note, in all cases we specify the datastore that should hold the datasets and whether they should be registered following step completion or not. This can optionally be disabled by removing the register_on_complete() call.
# 
# These can be viewed in the Datasets tab directly in the AML Portal

# In[10]:


#get data from storage location and save to exp_raw_data
exp_raw_data       = OutputFileDatasetConfig(name='Exp_Raw_Data', destination=(default_ds, dataset_prefix_name + '_raw_data/{run-id}')).read_delimited_files().register_on_complete(name= dataset_prefix_name + '_Raw_Data')

#data split into testing and training
exp_training_data  = OutputFileDatasetConfig(name='Exp_Training_Data', destination=(default_ds, dataset_prefix_name + '_training_data/{run-id}')).read_delimited_files().register_on_complete(name=dataset_prefix_name + '_Training_Data')
exp_testing_data   = OutputFileDatasetConfig(name='Exp_Testing_Data', destination=(default_ds, dataset_prefix_name + '_testing_data/{run-id}')).read_delimited_files().register_on_complete(name=dataset_prefix_name + '_Testing_Data')


# ## Define Pipeline Data
# 
# Data used in pipeline can be **produced by one step** and **consumed in another step** by providing a PipelineData object as an output of one step and an input of one or more subsequent steps
# 
# This can be leveraged for moving a model from one step into another for model evaluation

# ### Create Python Script Step

# In[11]:


get_data_step = PythonScriptStep(
    name='Get Data',
    script_name='get_data.py',
    arguments =['--exp_raw_data', exp_raw_data],
    outputs=[exp_raw_data],
    compute_target=pipeline_cluster,
    source_directory='./' + experiment_folder,
    allow_reuse=False,
    runconfig=pipeline_run_config
)


# ### Split Data Step

# In[13]:


split_scale_step = PythonScriptStep(
    name='Split  Raw Data',
    script_name='split.py',
    arguments =['--exp_training_data', exp_training_data,
                '--exp_testing_data', exp_testing_data],
    inputs=[exp_raw_data.as_input(name='Exp_Raw_Data')],
    outputs=[exp_training_data, exp_testing_data],
    compute_target=pipeline_cluster,
    source_directory='./' + experiment_folder,
    allow_reuse=False,
    runconfig=pipeline_run_config
)


# In[15]:


### TrainingStep


# In[16]:


#Raw data will be preprocessed and registered as train/test datasets

model_file = PipelineData(name='model_file', datastore=default_ds)

#by specifying as input, it does not need to be included in the arguments
train_model_step = PythonScriptStep(
    name='Train',
    script_name='train.py',
    arguments =['--model_file_output', model_file],
    inputs=[
            exp_training_data.as_input(name='Exp_Training_Data'),
            exp_testing_data.as_input(name='Exp_Testing_Data'),
           ],
    outputs = [model_file],
    compute_target=pipeline_cluster,
    source_directory='./' + experiment_folder,
    allow_reuse=False,
    runconfig=pipeline_run_config
)


# ### Evaluate Model Step

# In[77]:


#Evaluate and register model here
#Compare metrics from current model and register if better than current
#best model


deploy_file = PipelineData(name='deploy_file', datastore=default_ds)

evaluate_and_register_step = PythonScriptStep(
    name='Evaluate and Register Model',
    script_name='evaluate_and_register.py',
    arguments=[
        '--model_file', model_file,
        '--deploy_file_output', deploy_file,       
    ],
    inputs=[model_file.as_input('model_file'),
            exp_training_data.as_input(name='Exp_Training_Data'),
            exp_testing_data.as_input(name='Exp_Testing_Data')
           ],
    outputs=[ deploy_file],
    compute_target=pipeline_cluster,
    source_directory='./' + experiment_folder,
    allow_reuse=False,
    runconfig=pipeline_run_config
)


# ### Deploy ACI

# In[79]:


# exp_deploy_pipeline_data = PipelineData(
#         name='scoring_url_file', 
#         pipeline_output_name='scoring_url_file',
#         datastore=default_ds,
#         output_mode='mount',
#         is_directory=False)

scoring_file = PipelineData(name='scoring_file', datastore=default_ds)

aci_service_name = 'diabetes-service-remote-training'
registered_model_name = 'diabetes_model_remote'

env_name = PipelineParameter(name='environment_name', default_value=registered_env_name)
service_name = PipelineParameter(name='service_name', default_value=aci_service_name)
model_name = PipelineParameter(name='model_name', default_value=registered_model_name)



deploy_test = PythonScriptStep(
    name='Deploy to ACI',
    script_name='deployACI.py',
    arguments=[
        '--scoring_file_output', scoring_file,
        '--deploy_file', deploy_file,
        '--environment_name', env_name,
        '--service_name', service_name,
        '--model_name', model_name
        
    ],
    inputs=[
        deploy_file.as_input('deploy_file'),
            
    ],
    outputs=[scoring_file],
    compute_target=pipeline_cluster,
    source_directory='./' + experiment_folder,
    allow_reuse=False,
    runconfig=pipeline_run_config
)


# ## Create Pipeline steps

# ## Create Pipeline
# Create an Azure ML Pipeline by specifying the steps to be executed. Note: based on the dataset dependencies between steps, exection occurs logically such that no step will execute unless all of the necessary input datasets have been generated.

# In[82]:


pipeline = Pipeline(workspace=ws, steps=[get_data_step, split_scale_step, train_model_step, evaluate_and_register_step, deploy_test])


# In[86]:


experiment = Experiment(ws, 'AML_Automation_DevOpsPipelineTraining')
run = experiment.submit(pipeline)

