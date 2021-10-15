from azureml.pipeline.core import PublishedPipeline
from azureml.core import Workspace
import argparse
import os


def main():

    print('hello')

    subscription_id = os.getenv("SUBSCRIPTION_ID", default="")
    resource_group = os.getenv("RESOURCE_GROUP", default="")
    workspace_name = os.getenv("WORKSPACE_NAME", default="")
    pipeline_name = os.getenv("PIPELINE_NAME", default="")
    build_id = os.getenv("BUILD_BUILDID", default='1')
    workspace_region = os.getenv("WORKSPACE_REGION", default="")
    cluster_name = os.getenv("CLUSTER_NAME", default="")
    
    
    print('subscription_id = ' + str(subscription_id))
    print('build_id = ' + str(build_id))
    print('resource_group = ' + str(resource_group))
    print('workspace_name = ' + str(workspace_name))
    print('workspace_region = ' + str(workspace_region))
    print('cluster_name = ' + str(cluster_name))
    print('pipeline_name = ' + str(pipeline_name))

    workspace_name = 'mm-aml-dev'
    resource_group = 'mm-machine-learning-dev-rg'
    workspace_region = 'eastus2'

    registered_env_name = "experiment_env"
    experiment_folder = 'exp_pipeline'
    dataset_prefix_name = 'exp'
    cluster_name = "mm-cluster"
    pipeline_name = 'mlops-training-registration-pipeline'


    parser = argparse.ArgumentParser("register")
    parser.add_argument(
        "--output_pipeline_id_file",
        type=str,
        default="pipeline_id.txt",
        help="Name of a file to write pipeline ID to"
    )
    args = parser.parse_args()

    aml_workspace = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )

    # Find the pipeline that was published by the specified build ID
    pipelines = PublishedPipeline.list(aml_workspace)
    print('looking for pipeline:' + pipeline_name + ', this is configured in the yml file')
    matched_pipes = []
    bfound = False
    
    for p in pipelines:
        if p.name == pipeline_name:
            bfound = True
            print('found:' + p.name + ' ' + p.version + ', looking for version:' + p.version)
            if p.version == build_id:
                matched_pipes.append(p)
            else:
                print('Found pipeline name: ' +  p.name + ', but NOT a match on version - pipeline version: ' + p.version + ', build id:' + build_id)
        else:
            print('Not a match: ' + p.name)
            
                
    if bfound == False:
        print('unable to find pipeline')
        
    if(len(matched_pipes) > 1):
        published_pipeline = None
        raise Exception(f"Multiple active pipelines are published for build {build_id}.")  # NOQA: E501
    elif(len(matched_pipes) == 0):
        published_pipeline = None
        raise KeyError(f"Unable to find a published pipeline for this build {build_id}")  # NOQA: E501
    else:
        published_pipeline = matched_pipes[0]
        print("published pipeline id is", published_pipeline.id)

        # Save the Pipeline ID for other AzDO jobs after script is complete
        if args.output_pipeline_id_file is not None:
            with open(args.output_pipeline_id_file, "w") as out_file:
                out_file.write(published_pipeline.id)


if __name__ == "__main__":
    main()