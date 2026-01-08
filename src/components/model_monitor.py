from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class ModelMonitor:

    def __init__(self, config):
        self.config = config

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:

        drift_results = {}
        drifted_features = []

        if features is None:
            features = reference_data.columns.tolist()

        for feature in features:

            if feature not in reference_data.columns or feature not in current_data.columns:
                logger.warning(f"Feature '{feature}' not found in both reference and current data.")
                continue

            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                logger.warning(f"Feature '{feature}' has no valid values in either reference or current data.")
                continue

            is_numeric = pd.api.types.is_numeric_dtype(ref_values)

            if is_numeric:
                statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_detected = p_value < threshold
            else:
                all_categories = ref_values.unique().tolist() + [c for c in curr_values.unique() if c not in ref_values.unique()]
                ref_counts = ref_values.value_counts().reindex(all_categories, fill_value=0)
                curr_counts = curr_values.value_counts().reindex(all_categories, fill_value=0)
                statistic, p_value = stats.chisquare(f_obs=curr_counts, f_exp=ref_counts)
                drift_detected = p_value < threshold

            drift_results[feature] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": drift_detected
            }

            if drift_detected:
                drifted_features.append(feature)

        overall_drift_detected = len(drifted_features) > 0

        results = {
            "drift_detected": overall_drift_detected,
            "drifted_features": drifted_features,
            "drift_results": drift_results
        }

        if overall_drift_detected:
            logger.warning(f"Data drift detected in features: {', '.join(drifted_features)} (threshold={threshold})")
        else:
            logger.info("No data drift detected.")

        return results

    def monitor_prediction_distribution(
        self,
        predictions: pd.Series,
        reference_predictions: Optional[pd.Series] = None,
        threshold: float = 0.05):

        logger.info("Monitoring prediction distribution...")

        results = {
            "statistics": {
                "mean": float(predictions.mean()),
                "std": float(predictions.std()),
                "min": float(predictions.min()),
                "max": float(predictions.max()),
                "median": float(predictions.median()),
            },
        }

        if reference_predictions is not None:
            statistic, p_value = stats.ks_2samp(reference_predictions, predictions)
            anomaly_detected = p_value < threshold

            results["anomaly_detected"] = anomaly_detected
            results["p_value"] = float(p_value)
            results["statistic"] = float(statistic)

            if anomaly_detected:
                logger.warning("Anomaly detected in prediction distribution")
        else:
            q1 = predictions.quantile(0.25)
            q3 = predictions.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = predictions[(predictions < lower_bound) | (predictions > upper_bound)]
            outlier_rate = len(outliers) / len(predictions)

            results["outlier_rate"] = float(outlier_rate)
            results["outlier_count"] = len(outliers)
            results["anomaly_detected"] = outlier_rate > 0.05

        return results

    def calculate_performance_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        model_type: str = "classification",
    ) -> Dict[str, float]:

        metrics = {}

        if model_type == "classification":
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        elif model_type == "regression":
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)

        return metrics

    def check_performance_degradation(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        degradation_threshold: float = 0.1,
    ) -> Dict[str, Any]:

        logger.info("Checking performance degradation...")

        degradations = {}
        degraded_metrics = []

        for metric in current_metrics:
            if metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]

                if metric in ["rmse", "mae"]:
                    degradation = (current_val - baseline_val) / baseline_val
                    is_degraded = degradation > degradation_threshold
                else:
                    degradation = (baseline_val - current_val) / baseline_val
                    is_degraded = degradation > degradation_threshold

                degradations[metric] = {
                    "current": current_val,
                    "baseline": baseline_val,
                    "degradation": degradation,
                    "is_degraded": is_degraded,
                }

                if is_degraded:
                    degraded_metrics.append(metric)

        overall_degradation = len(degraded_metrics) > 0

        results = {
            "degradation_detected": overall_degradation,
            "degraded_metrics": degraded_metrics,
            "metric_degradations": degradations,
            "threshold": degradation_threshold,
        }

        if overall_degradation:
            logger.warning(f"Performance degradation detected in: {degraded_metrics}")
        else:
            logger.info("No performance degradation detected")

        return results

    def generate_monitoring_report(
        self,
        drift_results: Dict[str, Any],
        performance_results: Optional[Dict[str, Any]] = None,
        prediction_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        report = {
            "data_drift": drift_results,
            "overall_status": "healthy",
            "alerts": [],
        }

        if drift_results.get("drift_detected", False):
            report["alerts"].append({
                "type": "data_drift",
                "severity": "warning",
                "message": f"Data drift detected in {len(drift_results.get('drifted_features', []))} features",
            })
            report["overall_status"] = "warning"

        if performance_results:
            report["performance"] = performance_results
            if performance_results.get("degradation_detected", False):
                report["alerts"].append({
                    "type": "performance_degradation",
                    "severity": "critical",
                    "message": f"Performance degradation in {len(performance_results.get('degraded_metrics', []))} metrics",
                })
                report["overall_status"] = "critical"

        if prediction_results:
            report["predictions"] = prediction_results
            if prediction_results.get("anomaly_detected", False):
                report["alerts"].append({
                    "type": "prediction_anomaly",
                    "severity": "warning",
                    "message": "Anomaly detected in prediction distribution",
                })
                if report["overall_status"] == "healthy":
                    report["overall_status"] = "warning"

        return report
