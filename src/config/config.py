from components import model_trainer
from utils import read_yaml
from entity.config_entity import (
                                DataProcessorConfig, 
                                FeatureStoreConfig,
                                ModelTrainerConfig,
                                ModelValidatorConfig,
                                ModelRegistryConfig,
                                ProjectConfig)

class ConfigurationManager:

    def __init__(
        self, 
        config_file_path='../config/project_config.yaml',
        params_file_path='../params.yaml',
        schema_file_path='../schema.yaml'
        ):

        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

    def get_project_config(self) -> ProjectConfig:
        config = self.config.project_config

        project_config = ProjectConfig(
            catalog=config.catalog,
            schema=config.schema
        )
        return project_config
        
    def get_data_processor_config(self) -> DataProcessorConfig:
        config = self.config.data_processor

        data_processor_config = DataProcessorConfig(
            sql_query = config.sql_query,
            value_columns = config.value_columns
        )
        return data_processor_config

    def get_feature_store(self) -> FeatureStoreConfig:
        config = self.config.feature_store

        feature_store_config = FeatureStoreConfig(
            table_name = config.table_name,
            primary_keys = config.primary_keys
        )
        return feature_store_config

    def get_model_trainer(self) -> ModelTrainerConfig:
        config = self.config.model_training

        model_trainer_config = ModelTrainerConfig(
                                    experiment_name = config.experiment_name,
                                    registered_model_prefix = config.registered_model_prefix,
                                    params = config.params)
        return model_trainer_config

    def get_model_validator(self) -> ModelValidatorConfig:
        config = self.config.model_validation

        model_validation_config = ModelValidatorConfig(
                                    model_name = config.model_name,
                                    registry_name = config.registry_name,
                                    regression = config.repression)
 
        return model_validation_config

    def get_model_registry(self) -> ModelRegistryConfig:
        config = self.config.model_registry

        model_registry_config = ModelRegistryConfig(
                                    model_name = config.model_name
                                    )
 
        return model_registry_config