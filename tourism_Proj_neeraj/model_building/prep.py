import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi, hf_hub_download

# -----------------------------
# 1. Get and validate HF token
# -----------------------------
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError(
        "HF_TOKEN environment variable is not set. "
        "Please add a valid Hugging Face token as a GitHub Actions secret named HF_TOKEN."
    )

# Login & API client
login(token)
api = HfApi(token=token)

# -----------------------------
# 2. Download dataset from HF
# -----------------------------
# Repo and filename on Hugging Face
HF_DATASET_REPO = "neerajig/Tourism-Package-Prediction"
HF_DATASET_FILE = "tourism.csv"

local_dataset_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=HF_DATASET_FILE,
    repo_type="dataset",
    token=token,
)

df = pd.read_csv(local_dataset_path)
print("Dataset loaded successfully from Hugging Face.")

# -----------------------------
# 3. Basic cleaning
# -----------------------------

# Drop the unique identifier and unnamed column if present
df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

# Clean 'Gender' and 'MaritalStatus'
if "Gender" in df.columns:
    df["Gender"] = (
        df["Gender"]
        .astype(str)
        .str.strip()
        .str.replace("Fe Male", "FeMale", regex=False)
    )

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = (
        df["MaritalStatus"]
        .astype(str)
        .str.strip()
        .str.replace("Unmarried", "Single", regex=False)
    )

# Encode 'TypeofContact'
if "TypeofContact" in df.columns:
    le = LabelEncoder()
    df["TypeofContact"] = le.fit_transform(df["TypeofContact"].astype(str))
else:
    raise KeyError("Column 'TypeofContact' not found in dataset.")

# -----------------------------
# 4. Train–test split
# -----------------------------
target_col = "ProdTaken"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset.")

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Save splits locally
# -----------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train/test splits saved locally.")

# -----------------------------
# 6. Upload splits back to HF
# -----------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,  # same name in repo
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )

print("All split files uploaded to Hugging Face dataset repo.")
