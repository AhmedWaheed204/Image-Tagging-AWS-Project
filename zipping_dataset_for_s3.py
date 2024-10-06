import boto3
import zipfile
import os

def compress_dataset_to_zip(folder_path, zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
                
                
                
from botocore.exceptions import NoCredentialsError
import boto3

# Set up your AWS credentials and S3 client
aws_access_key_id = 'YOUR_ACCESS_KEY'
aws_secret_access_key = 'YOUR_SECRET_KEY'
aws_region = 'us-east-1'  # Update to your region

# Function to upload file
def upload_file(file_name, bucket, object_name=None):
    # Create an S3 client with credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    s3_client = session.client('s3')
    
    if object_name is None:
        object_name = file_name
    
    try:
        # Upload the file
        s3_client.upload_file(file_name, bucket, object_name)
        print(f'Successfully uploaded {file_name} to {bucket}/{object_name}')
    except FileNotFoundError:
        print(f'The file {file_name} was not found.')
    except NoCredentialsError:
        print('Credentials not available.')

# Example usage
bucket_name = 'dkh1-ais1-m1d-data'
file_path = '/kaggle/working/training.zip'
s3_path_file = 'data/training.zip'

upload_file(file_path, bucket_name, s3_path_file)