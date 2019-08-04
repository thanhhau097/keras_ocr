#!/usr/bin/python
import os 
import sagemaker
import subprocess


# Define instance configurations 
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name

repo_name = 'lionel-ocr' # ECR repository
image_tag = 'prj_scsk' # ECR image tag
base_job_name = 'scsk-lionelocr' # SageMaker training prefix
dockerfile = os.path.abspath('./new_dockerfile')

print("Account: {0}".format(account))
print("Region: {0}".format(region))
print("Repo name: {0}".format(repo_name))
print("Image tag: {0}".format(image_tag))
print("Base job name: {0}".format(base_job_name))
print("Docker file: {0}".format(dockerfile))

# Build docker and push to ionstance
subprocess.run("docker build -t {0} -f {1} . ".format(image_tag, dockerfile), shell=True)
subprocess.run("docker tag {0} {1}.dkr.ecr.{2}.amazonaws.com/{3}:latest".format(image_tag, account, region, repo_name), shell=True)
subprocess.run("docker push {0}.dkr.ecr.{1}.amazonaws.com/{2}:latest".format(account, region, repo_name), shell=True)

# Define data path in S3 
s3_directory = 's3://scsk-data/ocr_data/data'
train_input_channel = sagemaker.session.s3_input(s3_directory, distribution='FullyReplicated',  s3_data_type='S3Prefix')

# Define image name, output path to save model 
output_path = 's3://scsk-data/ocr_data/output/lionel'
image_name  = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, repo_name)

## Define instance to train 
train_instance_type = 'ml.p3.2xlarge'
# train_instance_type = 'ml.p3.8xlarge'

# Define space of disk to storage input data
storage_space = 200 # Gb

# Maximum seconds for this training job’s life (days * hours * seconds)
train_max_run = 1 * 24  * 3600

# Set sagemaker estimator and process to train
estimator = sagemaker.estimator.Estimator(
                       image_name=image_name,
                       base_job_name=base_job_name,
                       role=role,
                       input_mode='File',
                       train_instance_count=1,
                       train_volume_size=storage_space,
                       train_instance_type=train_instance_type,
                       output_path=output_path,
                       train_max_run=train_max_run,
                       sagemaker_session=sess)

estimator.fit({'train': train_input_channel})