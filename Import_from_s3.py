import boto3
import zipfile
import os

# Define your S3 bucket and file path
bucket_name = 'your-bucket-name'
file_key = 'your-file-path/file.zip'  # Example: 'data/mydata.zip'

# Create a session using Boto3 to access S3
s3 = boto3.client('s3')

# Download the zip file from S3
zip_file_path = '/tmp/file.zip'
s3.download_file(Bucket=bucket_name, Key=file_key, Filename=zip_file_path)

# Define extraction path
extract_dir = '/tmp/extracted_files'

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Check the extracted files
extracted_files = os.listdir(extract_dir)
print("Extracted files:", extracted_files)