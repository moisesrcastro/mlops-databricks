from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_percentage_error,
    r2_score,
)
import pandas as pd
import numpy as np
from loguru import logger

try:
    import mlflow
except ImportError:
    mlflow = None


class ModelValidator:
    def __init__(self, config, model_type: str = "regression"):
        self.config = config
        self.model_type = model_type
        self.thresholds = (
            config.validation.get(model_type) or self._get_default_threshold()
        )
        self.registry_name = getattr(config, "registry_name", None)


    def _get_default_threshold(self):
        if self.model_type == "classification":
            return {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.7}
        else:
            return {"r2": 0.65, "mape": 0.15}


    def calculate_metrics(self, y_true, y_pred):
        if self.model_type == "classification":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
            }
        else:
            return {
                "r2": r2_score(y_true, y_pred),
                "mape": mean_absolute_percentage_error(y_true, y_pred)
            }


    def _check_thresholds(self, metrics: Dict[str, float]):
        checks = {}
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                val = metrics[metric_name]
                if metric_name in ["mape"]:
                    checks[metric_name] = val <= threshold
                else:
                    checks[metric_name] = val >= threshold
            else:
                checks[metric_name] = True
        return checks


    def _load_previous_model(self):
        """Try to load the previous version from MLflow."""
        if self.registry_name and mlflow is not None:
            try:
                model_uri = f"models:/{self.registry_name}/Production"
                logger.info(f"Trying to load previous model: {model_uri}")
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                logger.warning(f"No previous model found in registry: {e}")
                return None
        return None


    def _baseline_naive(self, y_train, y_test):
        if self.model_type == "classification":
            most_freq = y_train.value_counts().idxmax()
            return np.full(len(y_test), most_freq)
        else:
            mean_val = y_train.mean()
            return np.full(len(y_test), mean_val)


    def _compute_delta(self, model_metrics, baseline_metrics):
        delta = {}
        for m in model_metrics:
            if m == "mape":  
                delta[m] = (baseline_metrics[m] - model_metrics[m]) / baseline_metrics[m]
            else:  
                delta[m] = (model_metrics[m] - baseline_metrics[m]) / baseline_metrics[m]
        return delta


    def validate_model(self, model, X_test, y_test, y_train=None):
        y_pred = model.predict(X_test)
        model_metrics = self.calculate_metrics(y_test, y_pred)

        prev = self._load_previous_model()
        if prev is not None:
            y_baseline = prev.predict(X_test)
            baseline_source = "previous_model"
        else:
            if y_train is None:
                raise ValueError("y_train required to compute naive baseline")
            y_baseline = self._baseline_naive(y_train, y_test)
            baseline_source = "naive_baseline"

        baseline_metrics = self.calculate_metrics(y_test, y_baseline)
        delta = self._compute_delta(model_metrics, baseline_metrics)

        checks = self._check_thresholds(model_metrics)
        passed = all(checks.values())

        if passed:
            logger.info("Model validation passed")
        else:
            logger.warning("Model validation failed")

        return {
            "validation_passed": passed,
            "metrics": model_metrics,
            "checks": checks,
            "baseline_metrics": baseline_metrics,
            "delta_vs_baseline": delta,
            "baseline_source": baseline_source
        }