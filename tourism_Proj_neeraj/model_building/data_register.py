from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Use token from environment variable
repo_id = "neerajig/Tourism-Package-Prediction"
repo_type = "dataset"  # must be singular

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Step 2: Upload folder
try:
    api.upload_folder(
        folder_path="tourism_Proj_neeraj/data",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Folder uploaded successfully.")
except HfHubHTTPError as e:
    print(f"Failed to upload folder: {e}")
