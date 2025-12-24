from loguru import logger
from pyspark.sql import DataFrame
from entity.config_entity import FeatureStoreConfig, ProjectConfig


class FeatureStoreManager:

    def __init__(
        self,
        spark,
        project_config: ProjectConfig,
        config: FeatureStoreConfig,
    ):
        self.spark = spark
        self.project_config = project_config
        self.config = config

    def full_table_name(self) -> str:
        print(self.config.table_name)
        return (
            f"{self.project_config.catalog}."
            f"{self.project_config.schema}."
            f"{self.config.table_name}"
        )

    def create_feature_table(
        self,
        df: DataFrame,
        description: str = "",
        mode: str = "overwrite",
    ):
        full_name = self.full_table_name()

        logger.info(f"Creating feature table: {full_name}")

        (
            df.write
            .format("delta")
            .mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(full_name)
        )

        if description:
            self.spark.sql(
                f"COMMENT ON TABLE {full_name} IS '{description}'"
            )

    def upsert_features(
        self,
        df: DataFrame,
        mode: str = "overwrite",
    ):
        full_name = self.full_table_name()

        logger.info(f"Writing features to {full_name} | mode={mode}")

        (
            df.write
            .format("delta")
            .mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(full_name)
        )

    def read_features(self) -> DataFrame:
        full_name = self.full_table_name()
        return self.spark.table(full_name)
