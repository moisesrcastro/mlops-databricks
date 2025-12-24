import mlflow
import yaml

from loguru import logger
from pathlib import Path
from typing import Dict, Union

from box import ConfigBox
from box.exceptions import BoxValueError
from mlflow.tracking import MlflowClient


def setup_mlflow(
    experiment_name: str | None = None
) -> None:
    """
    Setup MLflow tracking with Databricks.
    """
    mlflow.set_tracking_uri("databricks")

    exp_name = experiment_name

    if not exp_name:
        logger.warning("No MLflow experiment name configured")
        return

    try:
        mlflow.set_experiment(exp_name)
        logger.info(f"MLflow experiment set to: {exp_name}")
    except Exception as e:
        logger.exception(
            f"Failed to set MLflow experiment: {exp_name}"
        )
        raise e


def get_mlflow_client()->MlflowClient:
    """
    Get MLflow client for model registry operations.
    """
    return MlflowClient()


def log_model_metrics_experiment(
    metrics: Dict[str, Union[int, float]]
) -> None:
    """
    Log metrics to MLflow.

    :param metrics: Dictionary of metric names and numeric values
    """
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            logger.warning(
                f"Metric '{key}' ignored (non-numeric value: {value})"
            )
            continue

        mlflow.log_metric(key, value)
        logger.debug(f"Logged metric: {key} = {value}")


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read YAML file and return a ConfigBox object.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        if content is None:
            raise ValueError("YAML file is empty")

        return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty or malformed")
    except Exception as error:
        logger.exception(
            f"Error reading YAML file: {path_to_yaml}"
        )
        raise error
