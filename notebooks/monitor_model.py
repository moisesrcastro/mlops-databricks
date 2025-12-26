import mlflow
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from src.config.config import ConfigurationManager
from src.components.model_registry import ModelRegistry
from src.components.model_monitor import ModelMonitor
from src.components.feature_store import FeatureStoreManager
from datetime import datetime

logger.info("Starting model monitoring pipeline")

spark = SparkSession.builder.getOrCreate()

config_manager = ConfigurationManager()

project_config = config_manager.get_project_config()
model_registry_config = config_manager.get_model_registry()
feature_store_config = config_manager.get_feature_store()

mlflow.set_registry_uri("databricks")

registry = ModelRegistry(model_registry_config)

prod_model_info = registry.get_latest_model_version(
    stages=["Production"]
)

if not prod_model_info:
    raise RuntimeError("No model found in Production stage")

model_version = prod_model_info["versions"][0]["version"]

logger.info(
    "Monitoring model '{}' version {}",
    registry.model_name,
    model_version
)

model = registry.load_model(stage="Production")

feature_store = FeatureStoreManager(
    spark=spark,
    project_config=project_config,
    config=feature_store_config
)

reference_df = (
    feature_store
    .read_features()
    .orderBy("semana")
)

reference_pd = reference_df.toPandas()

target_col = 'entrada_mes'

feature_cols = [
    c for c in reference_pd.columns
    if c not in {target_col, "semana", "mes"}
]

reference_features = reference_pd[feature_cols]

logger.info(
    "Reference data loaded | rows={} | features={}",
    reference_pd.shape[0],
    len(feature_cols)
)

current_df = (
    feature_store
    .read_features()
    .filter("semana >= date_sub(current_date(), 28)")
)

current_pd = current_df.toPandas()
current_features = current_pd[feature_cols]

logger.info(
    "Current data loaded | rows={}",
    current_pd.shape[0]
)

monitor = ModelMonitor(config=project_config)

drift_results = monitor.detect_data_drift(
    reference_data=reference_features,
    current_data=current_features,
    features=feature_cols,
    threshold=0.05
)

logger.info(
    "Data drift detected: {}",
    drift_results["drift_detected"]
)

reference_predictions = pd.Series(
    model.predict(reference_features)
)

current_predictions = pd.Series(
    model.predict(current_features)
)

prediction_results = monitor.monitor_prediction_distribution(
    predictions=current_predictions,
    reference_predictions=reference_predictions,
    threshold=0.05
)

if target_col in current_pd.columns:

    performance_results = monitor.check_performance_degradation(
        current_metrics=monitor.calculate_performance_metrics(
            y_true=current_pd[target_col],
            y_pred=current_predictions,
            model_type="regression",
        ),
        baseline_metrics=monitor.calculate_performance_metrics(
            y_true=reference_pd[target_col],
            y_pred=reference_predictions,
            model_type="regression",
        ),
        degradation_threshold=0.1,
    )
else:
    logger.info("No ground truth available for performance monitoring")
    performance_results = None

monitoring_report = monitor.generate_monitoring_report(
    drift_results=drift_results,
    performance_results=performance_results,
    prediction_results=prediction_results,
)

logger.info("Monitoring status: {}", monitoring_report["overall_status"])


monitoring_row = {
    "model_name": registry.model_name,
    "model_version": int(model_version),
    "monitoring_timestamp": datetime.utcnow(),  
    "overall_status": monitoring_report["overall_status"],
    "drift_detected": bool(drift_results["drift_detected"]),
    "drifted_features": ",".join(drift_results["drifted_features"]),
    "alerts": str(monitoring_report["alerts"]),
}

monitoring_df = spark.createDataFrame([monitoring_row])

monitoring_table = (
    f"{project_config.catalog}."
    f"{project_config.schema}."
    "model_monitoring_results"
)

monitoring_df.write.mode("append").saveAsTable(monitoring_table)

logger.info(
    "Monitoring results saved to {}",
    monitoring_table
)
