import mlflow
from loguru import logger

from src.components.model_registry import ModelRegistry
from src.components.model_validator import ModelValidator
from src.config.config import ConfigurationManager
from pyspark.sql import SparkSession
# Configuration loading
spark = SparkSession.builder.getOrCreate()

logger.info("Loading project configuration")

config_manager = ConfigurationManager()

project_config = config_manager.get_project_config()
model_registry_config = config_manager.get_model_registry()
model_trainer_config = config_manager.get_model_trainer()
model_validator_config = config_manager.get_model_validator()
feature_store_config = config_manager.get_feature_store()

# MLflow experiment & run selection
logger.info(
    "Fetching MLflow experiment: {}", 
    model_trainer_config.experiment_name
)

experiment = mlflow.get_experiment_by_name(
    model_trainer_config.experiment_name
)

if experiment is None:
    raise RuntimeError("MLflow experiment not found")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=2
)

if runs.empty:
    raise RuntimeError("No MLflow runs found for this experiment")

run_id = runs.iloc[1]["run_id"]
logger.info("Using MLflow run_id={}", run_id)

# Model loading
model_uri = f"runs:/{run_id}/model"
logger.info("Loading model from {}", model_uri)

model = mlflow.sklearn.load_model(model_uri)

# Feature loading
table_fqn = (
    f"{project_config.catalog}."
    f"{project_config.schema}."
    f"{feature_store_config.table_name}"
)

logger.info("Loading features from table {}", table_fqn)
df_features = spark.sql(f"SELECT * FROM {table_fqn}")

target_col = "entrada_mes"

feature_cols = [
    c for c in df_features.columns
    if c not in {target_col, "semana", "mes"}
]

logger.info(
    "Dataset loaded with {} rows and {} feature columns",
    df_features.count(),
    len(feature_cols)
)

# Train / test split (deterministic)
logger.info("Splitting dataset into train/test")

df_train, df_test = df_features.randomSplit(
    [0.8, 0.2],
    seed=42
)

logger.info(
    "Train rows: {}, Test rows: {}",
    df_train.count(),
    df_test.count()
)

logger.info("Converting Spark DataFrames to Pandas")

X_train = df_train.select(feature_cols).toPandas()
y_train = df_train.select(target_col).toPandas().values.ravel()

X_test = df_test.select(feature_cols).toPandas()
y_test = df_test.select(target_col).toPandas().values.ravel()

# Model validation
logger.info("Starting model validation")

validator = ModelValidator(
    config=model_validator_config,
    model_type="regression"
)

validation_result = validator.validate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    y_train=y_train
)

logger.info(
    "Validation result: passed={}",
    validation_result["validation_passed"]
)


if validation_result["validation_passed"]:
    logger.success("Model approved by automated validation")

    mlflow.set_registry_uri("databricks")
    registry = ModelRegistry(model_registry_config)

    logger.info("Registering model in MLflow Model Registry")

    model_version = registry.register_model(
        run_id=run_id,
        model_path="model",
        description="Model approved by automated validation"
    )

    logger.info(
        "Transitioning model version {} to Production",
        model_version
    )

    registry.transition_model_stage(
        stage="Production",
        model_version=model_version
    )

    logger.success(
        "Model successfully promoted to Production (version={})",
        model_version
    )

else:
    logger.warning("Model rejected by automated validation")
