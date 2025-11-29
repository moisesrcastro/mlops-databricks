def setup_mlflow(config: ProjectConfig, experiment_name: str | None = None):
    """Setup MLflow tracking with Databricks.
    """
    mlflow.set_tracking_uri("databricks")
    
    exp_name = experiment_name or config.experiment_name_basic
    if exp_name:
        mlflow.set_experiment(exp_name)
        logger.info(f"MLflow experiment set to: {exp_name}")
    else:
        logger.warning("No experiment name configured")


def get_mlflow_client() -> MlflowClient:
    """Get MLflow client for model registry operations."""
    return MlflowClient()


def log_model_metrics_experiment(metrics: Dict[str, Any]):
    """Log metrics to MLflow.
    
    :param metrics: Dictionary of metric names and values
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
        logger.debug(f"Logged metric: {key} = {value}")