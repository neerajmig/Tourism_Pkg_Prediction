import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow


login(os.getenv("HF_TOKEN"))

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")  
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Tourism-Package-Prediction-Experiment")

api = HfApi()

dataset_repo_id = "neerajig/Tourism-Pkg-Prediction"

# Construct direct URLs for reading the CSV files from Hugging Face Hub
# This bypasses potential issues with hf_hub_download or HfFileSystem
base_url = f"https://huggingface.co/datasets/{dataset_repo_id}/raw/main/"
Xtrain_url = f"{base_url}Xtrain.csv"
Xtest_url = f"{base_url}Xtest.csv"
ytrain_url = f"{base_url}ytrain.csv"
ytest_url = f"{base_url}ytest.csv"

# Read files using pandas directly from URLs
Xtrain = pd.read_csv(Xtrain_url)
Xtest = pd.read_csv(Xtest_url)
ytrain = pd.read_csv(ytrain_url).squeeze() # Use .squeeze() to convert to Series
ytest = pd.read_csv(ytest_url).squeeze() # Use .squeeze() to convert to Series

# Corrected feature lists for the Tourism-Package-Prediction dataset
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups'
]
categorical_features = [
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]


# Set the clas weight to handle class imbalance
# Ensure ytrain is a Series for value_counts
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_pkg_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    model_repo_id = "neerajig/tourism_pkg_prediction_model" # Updated repo_id
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Model repository '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{model_repo_id}' not found. Creating new repository...")
        create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
        print(f"Model repository '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_package_prediction_model_v1.joblib",
        repo_id=model_repo_id,
        repo_type=repo_type,
    )
