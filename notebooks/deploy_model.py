import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from src.config.config import ConfigurationManager
from src.components.model_registry import ModelRegistry
from src.components.model_deployer import ModelDeployer


logger.info("Starting model deployment pipeline")

spark = SparkSession.builder.getOrCreate()

config_manager = ConfigurationManager()

project_config = config_manager.get_project_config()
model_registry_config = config_manager.get_model_registry()
model_deployer_config = config_manager.get_model_deployer()

mlflow.set_registry_uri("databricks")

registry = ModelRegistry(model_registry_config)
deployer = ModelDeployer(model_deployer_config)

logger.info(
    "Fetching latest Production model for '{}'",
    registry.model_name
)

prod_info = registry.get_latest_model_version(
    stages=["Production"]
)

if not prod_info or not prod_info.get("versions"):
    raise RuntimeError("No Production model found in Model Registry")

model_version = prod_info["versions"][0]["version"]

logger.info(
    "Found Production model - version={}",
    model_version
)

endpoint_info = deployer.get_endpoint()

if endpoint_info is None:
    logger.info("Endpoint not found. Creating a new one")

    deployer.create_endpoint(
        model_version=model_version
    )

    logger.success(
        "Endpoint '{}' created successfully",
        deployer.endpoint_name
    )

else:
    logger.info(
        "Endpoint already exists. Updating to model version {}",
        model_version
    )

    deployer.update_endpoint(
        model_version=str(model_version)
    )

    logger.success(
        "Endpoint '{}' updated successfully",
        deployer.endpoint_name
    )

endpoint_info = deployer.get_endpoint()

if endpoint_info:
    logger.info("Endpoint name: {}", endpoint_info["name"])
    logger.info("Endpoint state: {}", endpoint_info["state"])
    logger.info(
        "Served models: {}",
        endpoint_info["config"]["served_models"]
    )
else:
    logger.warning("Unable to fetch endpoint info after deployment")

logger.success("Model deployment pipeline finished successfully")
