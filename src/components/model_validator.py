from typing import Any, Dict, Optional
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
from entity.config_entity import ModelValidatorConfig

try:
    import mlflow
except ImportError:
    mlflow = None


LOWER_BETTER = {"mae", "mape", "rmse"}


class ModelValidator:
    def __init__(self, config: ModelValidatorConfig, model_type: str = "regression"):
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

    def calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        if self.model_type == "classification":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
        else:
            return {
                "r2": r2_score(y_true, y_pred),
                "mape": mean_absolute_percentage_error(y_true, y_pred),
            }

    def _check_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                val = metrics[metric_name]
                if metric_name in LOWER_BETTER:
                    checks[metric_name] = val <= threshold
                else:
                    checks[metric_name] = val >= threshold
            else:
                checks[metric_name] = True
        return checks

    def _load_previous_model(self):
        """Try to load the previous version from MLflow (Production)."""
        if self.registry_name and mlflow is not None:
            try:
                model_uri = f"models:/{self.registry_name}/Production"
                logger.info(f"Trying to load previous model: {model_uri}")
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                logger.info(f"No previous model found in registry: {e}")
                return None
        return None

    def _baseline_naive(self, y_train, y_test):
        if self.model_type == "classification":
            most_freq = y_train.value_counts().idxmax()
            return np.full(len(y_test), most_freq)
        else:
            mean_val = y_train.mean()
            return np.full(len(y_test), mean_val)

    def _compute_delta(self, model_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Dict[str, Optional[float]]:
        delta: Dict[str, Optional[float]] = {}
        for m in model_metrics:
            base = baseline_metrics.get(m)
            new = model_metrics.get(m)
            if base is None or new is None:
                delta[m] = None
                continue

            if base == 0:
                delta[m] = None
                continue

            if m in LOWER_BETTER:
                delta[m] = (base - new) / base
            else:
                delta[m] = (new - base) / base
        return delta

    def _compare_with_baseline(self, model_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Retorna um dicionário indicando, para cada métrica presente em model_metrics,
        se o modelo novo é >= baseline (ou <= para métricas onde menor é melhor).
        """
        comp: Dict[str, bool] = {}
        for m, new_val in model_metrics.items():
            base_val = baseline_metrics.get(m)
            if base_val is None:
                comp[m] = True 
                continue

            if m in LOWER_BETTER:
                comp[m] = new_val <= base_val
            else:
                comp[m] = new_val >= base_val
        return comp

    def validate_model(self, model: Any, X_test, y_test, y_train=None) -> Dict[str, Any]:
        """
        Valida o modelo:
          - verifica thresholds (threshold_passed)
          - carrega baseline (modelo anterior em Production, se existir) ou usa baseline naive
          - compara desempenho do modelo novo vs baseline (baseline_passed)
          - decisão final: validation_passed = threshold_passed and baseline_passed

        """
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error(f"Failed to predict with candidate model: {e}")
            raise

        model_metrics = self.calculate_metrics(y_test, y_pred)
        logger.info(f"Candidate metrics: {model_metrics}")

        threshold_checks = self._check_thresholds(model_metrics)
        threshold_passed = all(threshold_checks.values())

        failed_thresholds = [k for k, ok in threshold_checks.items() if not ok]
        if failed_thresholds:
            logger.warning(f"Threshold checks failed for metrics: {failed_thresholds}")
        else:
            logger.info("All threshold checks passed.")

        prev = self._load_previous_model()
        if prev is not None:
            try:
                y_baseline = prev.predict(X_test)
                baseline_source = "previous_model"
                logger.info("Using previous Production model as baseline.")
            except Exception as e:
                logger.warning(f"Previous model loaded but prediction failed: {e}. Falling back to naive baseline.")
                prev = None  
        if prev is None:
            if y_train is None:
                raise ValueError("y_train is required to compute naive baseline when no previous model is available.")
            y_baseline = self._baseline_naive(y_train, y_test)
            baseline_source = "naive_baseline"
            logger.info("Using naive baseline (mean / majority class).")

        baseline_metrics = self.calculate_metrics(y_test, y_baseline)
        logger.info(f"Baseline metrics ({baseline_source}): {baseline_metrics}")

        comp_checks = self._compare_with_baseline(model_metrics, baseline_metrics)
        baseline_passed = all(comp_checks.values())
        failed_baseline = [k for k, ok in comp_checks.items() if not ok]
        
        if failed_baseline:
            logger.warning(f"Baseline comparison failed for metrics: {failed_baseline}")
        else:
            logger.info("Candidate is better or equal to baseline on compared metrics.")

        validation_passed = threshold_passed and baseline_passed

        if validation_passed:
            logger.info("Model validation PASSED")
        else:
            logger.warning("Model validation FAILED")

        delta_vs_baseline = self._compute_delta(model_metrics, baseline_metrics)
        logger.debug(f"Delta vs baseline: {delta_vs_baseline}")

        result = {
            "validation_passed": validation_passed,
            "threshold_passed": threshold_passed,
            "baseline_passed": baseline_passed,
            "metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "threshold_checks": threshold_checks,
            "baseline_source": baseline_source,
            "thresholds_used": self.thresholds,
        }

        return result