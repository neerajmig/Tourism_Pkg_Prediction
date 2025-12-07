from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

space_repo_id = "neerajig/tourism_package_prediction"
repo_type = "space"

# Step 1: Check if the space exists, if not, create it
try:
    api.repo_info(repo_id=space_repo_id, repo_type=repo_type)
    print(f"Hugging Face Space '{space_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Hugging Face Space '{space_repo_id}' not found. Creating new space...")
    # Changed space_sdk from "streamlit" to "docker"
    create_repo(repo_id=space_repo_id, repo_type=repo_type, private=False, space_sdk="docker")
    print(f"Hugging Face Space '{space_repo_id}' created.")

# Step 2: Upload folder to the Space
api.upload_folder(
    folder_path="tourism_Proj_neeraj/deployment",     # the local folder containing your files
    repo_id=space_repo_id,                          # the target repo
    repo_type=repo_type,                          # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
    commit_message="Add Streamlit app files"
)
print(f"Files from 'tourism_project/deployment' uploaded to Hugging Face Space '{space_repo_id}'.")
