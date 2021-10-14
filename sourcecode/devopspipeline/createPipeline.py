# ## Azure ML Pipeline - Parameterized Input Dataset
# This notebook demonstrates creation & execution of an Azure ML pipeline designed to accept a parameterized input reflecting the location of a file in the Azure ML default datastore to be initially registered as a tabular dataset and subsequently processed. This notebook was built as part of a larger solution where files were moved from a blob storage container to the default AML datastore via Azure Data Factory.

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
import os
from env_variables import Env

# ### Connect to Azure ML Workspace, Provision Compute Resources, and get References to Datastores
# Connect to workspace using config associated config file. Get a reference to you pre-existing AML compute cluster or provision a new cluster to facilitate processing. Finally, get references to your default blob datastore.

# Connect to AML Workspace

print('hello')

subscription_id = os.getenv("SUBSCRIPTION_ID", default="")
resource_group = os.getenv("RESOURCE_GROUP", default="")
workspace_name = os.getenv("WORKSPACE_NAME", default="")
workspace_region = os.getenv("WORKSPACE_REGION", default="")

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

conda_yml_file = './environment.yml'

# Create a Python environment for the experiment (from a .yml file)
env = Environment.from_conda_specification("experiment_env", conda_yml_file)


run_config = RunConfiguration()
run_config.docker.use_docker = True
run_config.environment = env
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE


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

# Define Output datasets


exp_raw_data = OutputFileDatasetConfig(name='Exp_Raw_Data', destination=(default_ds, dataset_prefix_name + '_raw_data/{run-id}')).read_delimited_files().register_on_complete(name= dataset_prefix_name + '_Raw_Data')
exp_training_data  = OutputFileDatasetConfig(name='Exp_Training_Data', destination=(default_ds, dataset_prefix_name + '_training_data/{run-id}')).read_delimited_files().register_on_complete(name=dataset_prefix_name + '_Training_Data')
exp_testing_data   = OutputFileDatasetConfig(name='Exp_Testing_Data', destination=(default_ds, dataset_prefix_name + '_testing_data/{run-id}')).read_delimited_files().register_on_complete(name=dataset_prefix_name + '_Testing_Data')

exp_trained_model_pipeline_data = PipelineData(name='exp_trained_model_pipeline_data', datastore=default_ds)


# ### Define Pipeline Steps
# The pipeline below consists of two steps - one step to gather and register the uploaded file in the AML datastore, and a secondary step to consume and process this registered dataset. Also, any PipelineParameters defined above can be passed to and consumed within these steps.

#Get raw data from registered ADLS Gen2 datastore
#Register tabular dataset after retrieval
get_data_step = PythonScriptStep(
    name='Get Data',
    script_name='get_data.py',
    arguments =['--exp_raw_data', exp_raw_data],
    outputs=[exp_raw_data],
    compute_target=pipeline_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

#Normalize the raw data using a MinMaxScaler
#and then split into test and train datasets
split_scale_step = PythonScriptStep(
    name='Split  Raw Data',
    script_name='split.py',
    arguments =['--exp_training_data', exp_training_data,
                '--exp_testing_data', exp_testing_data],
    inputs=[exp_raw_data.as_input(name='Exp_Raw_Data')],
    outputs=[exp_training_data, exp_testing_data],
    compute_target=pipeline_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

#Train autoencoder using raw data as an input
#Raw data will be preprocessed and registered as train/test datasets
#Scaler and train autoencoder will be saved out
train_model_step = PythonScriptStep(
    name='Train',
    script_name='train.py',
    arguments =['--exp_trained_model_pipeline_data', exp_trained_model_pipeline_data],
    inputs=[exp_training_data.as_input(name='Exp_Training_Data'),
            exp_testing_data.as_input(name='Exp_Testing_Data'),
           ],
    outputs=[exp_trained_model_pipeline_data],
    compute_target=pipeline_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

#Evaluate and register model here
#Compare metrics from current model and register if better than current
#best model
evaluate_and_register_step = PythonScriptStep(
    name='Evaluate and Register Model',
    script_name='evaluate_and_register.py',
    arguments=['--exp_trained_model_pipeline_data', exp_trained_model_pipeline_data],
    inputs=[ exp_trained_model_pipeline_data.as_input('exp_trained_model_pipeline_data')],
    compute_target=pipeline_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

pipeline = Pipeline(workspace=ws, steps=[get_data_step, split_scale_step, train_model_step, evaluate_and_register_step])

# ### Publish Pipeline
# Create a published version of your pipeline that can be triggered via an authenticated REST API request.

build_id = os.getenv('BUILD_BUILDID', default='1')
pipeline_name = os.getenv("PIPELINE_NAME", default="mlops-training-registration-pipeline")

published_pipeline = pipeline.publish(name = pipeline_name,
                                        version=build_id,
                                     description = 'Pipeline to load/register  data from datastore, train model, and register the trained model if it performs better than the current best model.',
                                     continue_on_step_failure = False)










