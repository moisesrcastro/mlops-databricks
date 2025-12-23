import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
import mlflow

from src.config.config import ConfigurationManager
from src.components.data_processor import DataProcessor
from src.components.feature_store import FeatureStoreManager
from src.components.model_trainer import ModelTrainer

logger.info("Starting Spark session")
spark = SparkSession.builder.getOrCreate()

logger.info("Loading configuration")
config_manager = ConfigurationManager()

project_config = config_manager.get_project_config()
data_processor_config = config_manager.get_data_processor_config()
feature_store_config = config_manager.get_feature_store()
model_trainer_config = config_manager.get_model_trainer()

logger.info("Building feature dataset")
data_processor = DataProcessor(
    spark=spark,
    config=data_processor_config
)

df = data_processor.build().dropna()

logger.info("Persisting features to Feature Store")
feature_store = FeatureStoreManager(
    spark=spark,
    project_config=project_config,
    config=feature_store_config
)

feature_store.create_feature_table(
    df=df,
    description="Features semanais de produção agregadas"
)

logger.info("Reading features from Feature Store")
df_features = feature_store.read_features().orderBy("semana")

target = "entrada_mes"

feature_cols = [
    c for c in df_features.columns
    if c not in [target, "semana", "mes", "entrada_mes"]
]

logger.info("Performing temporal train/test split")
total_rows = df_features.count()
split_index = int(total_rows * 0.8)

df_train = df_features.limit(split_index)
df_test = df_features.subtract(df_train)

logger.info("Preparing train/test matrices")
X_train = df_train.select(feature_cols).toPandas()
y_train = df_train.select(target).toPandas().values.ravel()

X_test = df_test.select(feature_cols).toPandas()
y_test = df_test.select(target).toPandas().values.ravel()

logger.info("Applying feature normalization")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = PassiveAggressiveRegressor(
    random_state=882,
    max_iter=8000,
    epsilon=0.01,
    C=0.9
)

trainer = ModelTrainer(
    config=model_trainer_config
)

logger.info("Starting model training")
resultados = trainer.train_multiple_models(
    models_list=[["PassiveAggressiveRegressor", model]],
    X_train=X_train_scaled,
    y_train=y_train,
    X_val=X_test_scaled,
    y_val=y_test,
    df_plot=df_features.toPandas(),
    target=target,
    split_idx=split_index
)

logger.info("Logging feature lineage in MLflow")
mlflow.log_input(
    mlflow.data.from_spark(
        df_train,
        table_name=feature_store.full_table_name()
    ),
    context="training"
)

logger.info("Pipeline execution completed successfully")