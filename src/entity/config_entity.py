from dataclasses import dataclass

@dataclass
class ProjectConfig:
    catalog:str;
    schema:str;
@dataclass
class DataProcessorConfig:
    sql_query: str;
    value_columns:list

@dataclass
class FeatureStoreConfig:
    table_name: str
    primary_keys: list[str]

@dataclass
class ModelTrainerConfig:
    experiment_name: str
    registered_model_prefix: str
    params: dict

@dataclass
class ModelValidatorConfig:
    model_name: str
    registry_name: str
    regression: dict  

@dataclass
class ModelRegistryConfig:
    model_name: str

@dataclass
class ModelDeployerConfig:
    model_name: str
    endpoint_name:str