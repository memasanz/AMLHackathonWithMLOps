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

# In[11]:


registered_env_name = "experiment_env"
experiment_folder = 'devOps_deploy_pipeline'
dataset_prefix_name = 'exp'
cluster_name = "mm-cluster"


# Import required packages

# In[12]:


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

# In[13]:


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

# In[14]:


#conda_yml_file = '../configuration/environment.yml'
conda_yml_file = './'+ experiment_folder+ '/environment.yml'


# In[ ]:





# In[15]:


# Create a Python environment for the experiment (from a .yml file)
env = Environment.from_conda_specification("experiment_env", conda_yml_file)


run_config = RunConfiguration()
run_config.docker.use_docker = True
run_config.environment = env
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE


# In[16]:


registered_env_name


# In[17]:


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

# ## Define Pipeline Data
# 
# Data used in pipeline can be **produced by one step** and **consumed in another step** by providing a PipelineData object as an output of one step and an input of one or more subsequent steps
# 
# This can be leveraged for moving a model from one step into another for model evaluation

# ### Create Python Script Step

# ### Evaluate Model Step

# ### Deploy ACI

# In[18]:



aci_service_name = 'diabetes-service-remote-training'
registered_model_name = 'diabetes_model_remote'

scoring_file = PipelineData(name='scoring_file', datastore=default_ds)
env_name = PipelineParameter(name='environment_name', default_value=registered_env_name)
service_name = PipelineParameter(name='service_name', default_value=aci_service_name)
model_name = PipelineParameter(name='model_name', default_value=registered_model_name)


deploy_test = PythonScriptStep(
    name='Deploy to ACI',
    script_name='deployACI.py',
    arguments=[
        '--scoring_file_output', scoring_file,
        '--environment_name', env_name,
        '--service_name', service_name,
        '--model_name', model_name
        
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

# In[19]:


pipeline = Pipeline(workspace=ws, steps=[deploy_test])


# In[20]:


experiment = Experiment(ws, 'AML_Automation_DevOpsPipelineDeploy')
run = experiment.submit(pipeline)


# In[ ]:




