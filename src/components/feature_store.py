from entity.config_entity import FeatureStoreConfig, ProjectConfig
from databricks import feature_store
from loguru import logger


class FeatureStoreManager:

    def __init__(self, spark, project_config:ProjectConfig, config: FeatureStoreConfig):
        self.spark = spark
        self.project_config = project_config
        self.config = config
        self.feature_store = feature_store.FeatureStoreClient()

    def full_table_name(self) -> str:
        catalog = self.project_config.catalog
        schema = self.project_config.schema
        table_name = self.config.table_name
        return f"{catalog}.{schema}.{table_name}"

    def create_feature_table(
        self,
        df,
        primary_keys: list[str],
        description: str = ""
    ):
        full_name = self.full_table_name()

        logger.info(f"Creating feature table: {full_name}")

        self.feature_store.create_table(
            name=full_name,
            primary_keys=primary_keys,
            df=df,
            description=description
        )
    
    def upsert_features(
        self,
        df,
        mode: str = "overwrite"
    ):
        full_name = self.full_table_name()

        logger.info(f"Writing features to {full_name} | mode={mode}")

        self.feature_store.write_table(
            name=full_name,
            df=df,
            mode=mode
        )
    
    def read_features(self):
        full_name = self.full_table_name()
        return self.feature_store.read_table(full_name)