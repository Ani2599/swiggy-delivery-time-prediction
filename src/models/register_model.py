import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging

# Create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# Console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)

# Create a formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='aniketnandanwar09', repo_name='swiggy-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/aniketnandanwar09/swiggy-delivery-time-prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


if __name__ == "__main__":
    # Root path
    root_path = Path(__file__).parent.parent.parent
    
    # Run information file path
    run_info_path = root_path / "run_information.json"

    # Load run information
    run_info = load_model_information(run_info_path)
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    # Model registry path
    model_registry_path = f"runs:/{run_id}/{model_name}"

    # Register the model
    model_version = mlflow.register_model(model_uri=model_registry_path, name=model_name)

    # Get registered model details
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"✅ Model {registered_model_name} registered with version {registered_model_version}")

    # Promote model to "Production" (not just "Staging")
    client = MlflowClient()
    
    # Transition model to "Production"
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Production"
    )

    logger.info(f"🚀 Model {registered_model_name} (version {registered_model_version}) promoted to Production")
