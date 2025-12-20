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