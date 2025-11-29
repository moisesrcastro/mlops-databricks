import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class ProjectConfig:
    experiment_name_basic: str
    model: Dict[str, Any]
    data: Dict[str, Any]
    features: Dict[str, List[str]]

    @staticmethod
    def from_yaml(path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        return ProjectConfig(
            experiment_name_basic=cfg.get("experiment_name_basic"),
            model=cfg.get("model", {}),
            data=cfg.get("data", {}),
            features=cfg.get("features", {}),
        )

    @property
    def parameters(self):
        return self.model.get("parameters", {})

    @property
    def num_features(self):
        return self.features.get("num_features", [])

    @property
    def cat_features(self):
        return self.features.get("cat_features", [])
