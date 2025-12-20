from typing import Any, Dict, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import pandas as pd
from loguru import logger
import numpy as np
from .config import ProjectConfig
from .utils import setup_mlflow, log_model_metrics

class ModelTrainer:
    """
    Classe responsável por treinar modelos e registrar tudo no MLflow.
    """

    def __init__(self, config: ProjectConfig, model_type: str = "regression"):
        #Inicializa o ModelTrainer.
        self.config = config
        self.model_type = model_type
        self.model = None #permite carregar o modelo direto do MLFlow
        
        #Direcionar para o experimento correto:
        setup_mlflow(config, config.experiment_name_basic)



    def train(self,X_train: pd.DataFrame,y_train: pd.Series,X_val: Optional[pd.DataFrame] = None,y_val: Optional[pd.Series] = None):
        """
        Treina o modelo e registra tudo no MLflow.
        """
        logger.info(f'Starting training for {self.model_type} model')

        with mlflow.start_run():
            
            #Registra as informações dentro do experimento
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_params(self.config.parameters)
            mlflow.log_param("num_variables", len(self.config.num_features))
            mlflow.log_param("cat_variables", len(self.config.cat_features))
        
            if self.model_type == "classification":
                self.model = GradientBoostingClassifier(
                                                        learning_rate = self.config.parameters.get('learning_rate'),
                                                        random_state = 42
                                                        )
            else:
                self.model = GradientBoostingRegressor(
                                                        learning_rate = self.config.parameters.get('learning_rate'),
                                                        random_state = 42
                                                        )
            #4. Treinar o modelo
            logger.info('Training model...')
            self.model.fit(X_train, y_train)

            train_pred = self.model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_pred)

            logger.info('Training metrics...')

            for metric_name, metric_value in train_metrics.items():
                logger.info(f'{metric_name}: {metric_value}')
                mlflow.log_metric(metric_name, metric_value)

            #Registra no experimento as metricas de treinamento
            log_model_metrics_experiment({f'train_{metric_name}': metric_value for metric_name, metric_value in train_metrics.items()})

            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred)

                logger.info('Validation metrics...')
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f'{metric_name}: {metric_value}')
                
                #Registra no experimento as metricas de treinamento
                log_model_metrics_experiment({f'val_{metric_name}': metric_value for metric_name, metric_value in val_metrics.items()})
            
            #8. Registrar o modelo no MLflow Model Registry
            mlflow.sklearn.log_model(
                                        self.model,
                                        "model",
                                        registered_model_name=self.config.model.get("name", "mlops-model"),
                                    )

            logger.info("Model training complete")
        
        return self.model
        


    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series):
        """
        Calcula métricas conforme o tipo de modelo.
        """
        metrics = {}
        
        if self.model_type == "classification":
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            
        else:
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2"] = r2_score(y_true, y_pred)
        
        return metrics


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Faz predições simples com o modelo treinado.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return pd.Series(self.model.predict(X))
