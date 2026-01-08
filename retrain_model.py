import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from src.config.config import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.components.model_registry import ModelRegistry
from src.components.model_validator import ModelValidator
from src.components.data_processor import DataProcessor
from src.components.feature_store import FeatureStoreManager

# Setup
logger.info("Starting model retraining pipeline")

spark = SparkSession.builder.getOrCreate()
mlflow.set_registry_uri("databricks")

config_manager = ConfigurationManager()

project_config = config_manager.get_project_config()
model_trainer_config = config_manager.get_model_trainer()
model_registry_config = config_manager.get_model_registry()
model_validator_config = config_manager.get_model_validator()
feature_store_config = config_manager.get_feature_store()
data_processor_config = config_manager.get_data_processor_config()

# Build feature dataset
logger.info("Building feature dataset")

data_processor = DataProcessor(
    spark=spark,
    config=data_processor_config
)

df = data_processor.build().dropna()

logger.info(
    "Feature dataset built with {} rows",
    df.count()
)

# Feature Store persistence
feature_store = FeatureStoreManager(
    spark=spark,
    project_config=project_config,
    config=feature_store_config
)

logger.info("Persisting features to Feature Store")

feature_store.create_feature_table(
    df=df,
    description="Weekly aggregated production features"
)

df = feature_store.read_features().orderBy("semana")

# Feature / target definition
target_col = "entrada_mes"

feature_cols = [
    c for c in df.columns
    if c not in {target_col, "semana", "mes"}
]

logger.info(
    "Final dataset: {} rows | {} feature columns",
    df.count(),
    len(feature_cols)
)

# Train / test split
df_train, df_test = df.randomSplit([0.8, 0.2], seed=42)

X_train = df_train.select(feature_cols).toPandas()
y_train = df_train.select(target_col).toPandas().values.ravel()

X_test = df_test.select(feature_cols).toPandas()
y_test = df_test.select(target_col).toPandas().values.ravel()

# Baseline validation (Production model)
logger.info("Validating current Production model")

validator = ModelValidator(
    config=model_validator_config,
    model_type="regression"
)

registry = ModelRegistry(model_registry_config)

try:
    model_uri = f"models:/{registry.model_name}/Production"
    baseline = mlflow.sklearn.load_model(model_uri)

    baseline_metrics = validator.validate_model(
        model=baseline,
        X_test=X_test,
        y_test=y_test,
        y_train=y_train
    )

    logger.info("Baseline validation result: {}", baseline_metrics)

except Exception as e:
    logger.warning(
        "No Production model found or failed to load baseline: {}",
        str(e)
    )
    baseline_metrics = {"validation_passed": False}

# Decision: retrain or keep Production
if baseline_metrics["validation_passed"]:
    logger.info(
        "Production model passed validation. No retraining required."
    )

else:
    logger.warning(
        "Production model failed validation. Starting retraining."
    )

    # Model retraining
    trainer = ModelTrainer(config=model_trainer_config)

    logger.info("Training candidate model")

    results = trainer.train_multiple_models(
        models_list=[["PassiveAggressiveRegressor", baseline]],
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test
    )

    candidate_model = results[0][1]

    # Feature lineage
    logger.info("Logging feature lineage to MLflow")

    mlflow.log_input(
        mlflow.data.from_spark(
            df_train,
            table_name=feature_store.full_table_name()
        ),
        context="retraining"
    )

    # Candidate validation
    logger.info("Validating candidate model")

    validation = validator.validate_model(
        model=candidate_model,
        X_test=X_test,
        y_test=y_test,
        y_train=y_train
    )

    if not validation["validation_passed"]:
        logger.warning("Candidate model rejected by validation")
        raise RuntimeError("Retrained model did not pass validation")

    logger.success("Candidate model approved")

    # Archive current Production model
    prod_info = registry.get_latest_model_version(
        stages=["Production"]
    )

    prod_version = prod_info["versions"][0]["version"]

    logger.info(
        "Archiving current Production model (version={})",
        prod_version
    )

    registry.transition_model_stage(
        stage="Archived",
        model_version=prod_version
    )

    # Register new model version
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

    run_id = runs.iloc[1]["run_id"]

    logger.info(
        "Registering new model from run_id={}",
        run_id
    )

    model_version = registry.register_model(
        run_id=run_id,
        model_path="model",
        description="Retrained model approved by automated validation"
    )

    # Promote to Production
    logger.info(
        "Promoting model version {} to Production",
        model_version
    )

    registry.transition_model_stage(
        stage="Production",
        model_version=model_version
    )

    logger.success(
        "Production model successfully updated (version={})",
        model_version
    )