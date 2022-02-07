
import argparse
from azureml.core import Workspace, Environment
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice, AciWebservice
from azureml.exceptions import WebserviceException

parser = argparse.ArgumentParser(description='Deploy arg parser')
parser.add_argument('--scoring_file_output', type=str, help='File storing the scoring url')
parser.add_argument('--deploy_file', type=str, help='File storing if model should be deployed')
parser.add_argument('--environment_name', type=str,help='Environment name')
parser.add_argument('--service_name', type=str,help='service name')
parser.add_argument('--model_name', type=str,help='model name')



args = parser.parse_args()
scoring_url_file = args.scoring_file_output
deploy_file      = args.deploy_file
environment_name = args.environment_name
service_name     = args.service_name
model_name       = args.model_name


run = Run.get_context()

#Get associated AML workspace
ws = run.experiment.workspace

model = Model(ws, model_name)
env = Environment.get(ws, environment_name)
inference_config = InferenceConfig(entry_script='score.py', environment=env)

# Deploy model
aci_config = AciWebservice.deploy_configuration(
            cpu_cores = 1, 
            memory_gb = 2, 
            tags = {'model': 'diabetes remote training'},
            auth_enabled=True,
            enable_app_insights=True,
            collect_model_data=True)

try:
    service = Webservice(ws, name=service_name)
    if service:
        service.delete()
except WebserviceException as e:
         print()

service = Model.deploy(ws, service_name, [model], inference_config, aci_config)
service.wait_for_deployment(True)
    

# Output scoring url
print(service.scoring_uri)
with open(scoring_url_file, 'w+') as f:
    f.write(service.scoring_uri)

