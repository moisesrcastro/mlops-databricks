from typing import Any, Dict, List, Tuple
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from loguru import logger
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
from src.entity.config_entity import ModelTrainerConfig
from mlflow.models.signature import infer_signature


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        mlflow.set_tracking_uri("databricks")
        
        
        if self.config.model_name is None or str(self.config.model_name).lower() == "none":
            self.model_name = None
        else:
            self.model_name = str(self.config.model_name)

        # Garantir que experiment_name nunca vem "None"
        if self.config.experiment_name is None or str(self.config.experiment_name).lower() == "none":
            raise ValueError("experiment_name não pode ser None ou 'None'")

        self.experiment_name = self.config.experiment_name
        self.params = self.config.params or {}

        mlflow.set_experiment(self.experiment_name)


    def _calculate_metrics(self, y_true, y_pred):
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred)
        }

    def train_single(
        self,
        model,
        model_name: str,
        X_train, y_train,
        X_val=None, y_val=None,
        df_plot=None, target=None, split_idx=None
    ):

        logger.info(f"Treinando modelo: {model_name}")

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("selected_model", model_name)
            mlflow.log_params(self.params)

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_pred)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

            if X_val is not None:
                val_pred = model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

            # Registro seguro (não registra se prefixo não fornecido)
            signature = infer_signature(X_train, model.predict(X_train))
            if self.model_name:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=f"model",
                    registered_model_name=f"{self.model_name}",
                    signature=signature
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=f"model",
                    signature=signature
                )

        return model

    def train_multiple_models(
        self,
        models_list: List[Tuple[str, Any]],
        X_train, y_train,
        X_val=None, y_val=None,
        df_plot=None, target=None, split_idx=None
    ):

        resultados = []

        for model_name, model_obj in models_list:
            print("-" * 50)
            try:
                trained = self.train_single(
                    model=model_obj,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    df_plot=df_plot,
                    target=target,
                    split_idx=split_idx
                )
                resultados.append((model_name, trained))

            except Exception as e:
                logger.error(f"Erro ao treinar {model_name}: {e}")

        return resultados

