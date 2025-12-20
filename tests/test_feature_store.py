import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType

import sys
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components.feature_store import FeatureStoreManager
from entity.config_entity import FeatureStoreConfig


@pytest.fixture(scope="session")
def spark():
    """Cria uma sessão Spark para os testes"""
    spark = SparkSession.builder \
        .appName("test_feature_store") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def mock_project_config():
    """Configuração mock do projeto"""
    config = Mock()
    config.catalog = "test_catalog"
    config.schema = "test_schema"
    return config


@pytest.fixture
def feature_store_config():
    """Configuração de teste para FeatureStore"""
    return FeatureStoreConfig(
        table_name="test_feature_table",
        primary_keys=["id", "semana"]
    )


@pytest.fixture
def sample_dataframe(spark):
    """Cria um DataFrame de exemplo para os testes"""
    data = [
        ("2024-01-01", 1, 100, 1000),
        ("2024-01-08", 2, 200, 2000),
        ("2024-01-15", 3, 300, 3000)
    ]
    
    schema = StructType([
        StructField("semana", StringType(), True),
        StructField("id", IntegerType(), True),
        StructField("vendas", IntegerType(), True),
        StructField("receita", IntegerType(), True)
    ])
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def mock_feature_store_client():
    """Mock do FeatureStoreClient do Databricks"""
    with patch('components.feature_store.feature_store.FeatureStoreClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


class TestFeatureStoreManager:
    """Testes para a classe FeatureStoreManager"""

    def test_init(self, spark, mock_project_config, feature_store_config, mock_feature_store_client):
        """Testa a inicialização do FeatureStoreManager"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        assert manager.spark == spark
        assert manager.project_config == mock_project_config
        assert manager.config == feature_store_config
        assert manager.feature_store is not None

    def test_full_table_name(self, spark, mock_project_config, feature_store_config, mock_feature_store_client):
        """Testa a construção do nome completo da tabela"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        full_name = manager.full_table_name()
        
        expected_name = f"{mock_project_config.catalog}.{mock_project_config.schema}.{feature_store_config.table_name}"
        assert full_name == expected_name
        assert full_name == "test_catalog.test_schema.test_feature_table"

    def test_full_table_name_different_configs(self, spark, mock_project_config, mock_feature_store_client):
        """Testa a construção do nome completo com diferentes configurações"""
        # Teste com diferentes valores
        config1 = FeatureStoreConfig(
            table_name="table1",
            primary_keys=["id"]
        )
        manager1 = FeatureStoreManager(spark, mock_project_config, config1)
        assert manager1.full_table_name() == "test_catalog.test_schema.table1"
        
        # Mudar catalog e schema
        mock_project_config.catalog = "prod_catalog"
        mock_project_config.schema = "prod_schema"
        config2 = FeatureStoreConfig(
            table_name="table2",
            primary_keys=["key"]
        )
        manager2 = FeatureStoreManager(spark, mock_project_config, config2)
        assert manager2.full_table_name() == "prod_catalog.prod_schema.table2"

    def test_create_feature_table(self, spark, mock_project_config, feature_store_config, 
                                   mock_feature_store_client, sample_dataframe):
        """Testa a criação de uma feature table"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        primary_keys = ["id", "semana"]
        description = "Test feature table"
        
        manager.create_feature_table(
            df=sample_dataframe,
            primary_keys=primary_keys,
            description=description
        )
        
        # Verificar que create_table foi chamado com os parâmetros corretos
        mock_feature_store_client.create_table.assert_called_once()
        call_args = mock_feature_store_client.create_table.call_args
        
        assert call_args.kwargs['name'] == "test_catalog.test_schema.test_feature_table"
        assert call_args.kwargs['primary_keys'] == primary_keys
        assert call_args.kwargs['df'] == sample_dataframe
        assert call_args.kwargs['description'] == description

    def test_create_feature_table_default_description(self, spark, mock_project_config, 
                                                      feature_store_config, mock_feature_store_client,
                                                      sample_dataframe):
        """Testa a criação de feature table com descrição padrão"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        primary_keys = ["id"]
        manager.create_feature_table(
            df=sample_dataframe,
            primary_keys=primary_keys
        )
        
        call_args = mock_feature_store_client.create_table.call_args
        assert call_args.kwargs['description'] == ""

    def test_create_feature_table_single_primary_key(self, spark, mock_project_config,
                                                      feature_store_config, mock_feature_store_client,
                                                      sample_dataframe):
        """Testa a criação com uma única primary key"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        primary_keys = ["id"]
        manager.create_feature_table(
            df=sample_dataframe,
            primary_keys=primary_keys
        )
        
        call_args = mock_feature_store_client.create_table.call_args
        assert call_args.kwargs['primary_keys'] == ["id"]

    def test_upsert_features_overwrite_mode(self, spark, mock_project_config, feature_store_config,
                                            mock_feature_store_client, sample_dataframe):
        """Testa o upsert de features com modo overwrite"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        manager.upsert_features(df=sample_dataframe, mode="overwrite")
        
        # Verificar que write_table foi chamado
        mock_feature_store_client.write_table.assert_called_once()
        call_args = mock_feature_store_client.write_table.call_args
        
        assert call_args.kwargs['name'] == "test_catalog.test_schema.test_feature_table"
        assert call_args.kwargs['df'] == sample_dataframe
        assert call_args.kwargs['mode'] == "overwrite"

    def test_upsert_features_merge_mode(self, spark, mock_project_config, feature_store_config,
                                       mock_feature_store_client, sample_dataframe):
        """Testa o upsert de features com modo merge"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        manager.upsert_features(df=sample_dataframe, mode="merge")
        
        call_args = mock_feature_store_client.write_table.call_args
        assert call_args.kwargs['mode'] == "merge"

    def test_upsert_features_default_mode(self, spark, mock_project_config, feature_store_config,
                                         mock_feature_store_client, sample_dataframe):
        """Testa o upsert de features com modo padrão"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        manager.upsert_features(df=sample_dataframe)
        
        call_args = mock_feature_store_client.write_table.call_args
        assert call_args.kwargs['mode'] == "overwrite"

    def test_read_features(self, spark, mock_project_config, feature_store_config,
                          mock_feature_store_client, sample_dataframe):
        """Testa a leitura de features"""
        # Configurar o mock para retornar o DataFrame de exemplo
        mock_feature_store_client.read_table.return_value = sample_dataframe
        
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        result = manager.read_features()
        
        # Verificar que read_table foi chamado com o nome correto
        mock_feature_store_client.read_table.assert_called_once_with(
            "test_catalog.test_schema.test_feature_table"
        )
        
        # Verificar que o resultado é o DataFrame esperado
        assert result == sample_dataframe

    def test_read_features_empty_table(self, spark, mock_project_config, feature_store_config,
                                      mock_feature_store_client):
        """Testa a leitura de uma tabela vazia"""
        empty_df = spark.createDataFrame([], StructType([
            StructField("id", IntegerType(), True),
            StructField("semana", StringType(), True)
        ]))
        
        mock_feature_store_client.read_table.return_value = empty_df
        
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        result = manager.read_features()
        
        assert result.count() == 0
        mock_feature_store_client.read_table.assert_called_once()

    def test_integration_workflow(self, spark, mock_project_config, feature_store_config,
                                  mock_feature_store_client, sample_dataframe):
        """Testa um workflow completo: criar tabela, escrever e ler"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        # Criar a tabela
        primary_keys = ["id", "semana"]
        manager.create_feature_table(
            df=sample_dataframe,
            primary_keys=primary_keys,
            description="Integration test table"
        )
        
        # Upsert features
        manager.upsert_features(df=sample_dataframe, mode="overwrite")
        
        # Configurar mock para leitura
        mock_feature_store_client.read_table.return_value = sample_dataframe
        
        # Ler features
        result = manager.read_features()
        
        # Verificar que todos os métodos foram chamados
        assert mock_feature_store_client.create_table.called
        assert mock_feature_store_client.write_table.called
        assert mock_feature_store_client.read_table.called
        
        # Verificar que o resultado da leitura é o DataFrame esperado
        assert result == sample_dataframe

    def test_different_table_names(self, spark, mock_project_config, mock_feature_store_client):
        """Testa com diferentes nomes de tabela"""
        config1 = FeatureStoreConfig(
            table_name="table_one",
            primary_keys=["id"]
        )
        manager1 = FeatureStoreManager(spark, mock_project_config, config1)
        assert manager1.full_table_name() == "test_catalog.test_schema.table_one"
        
        config2 = FeatureStoreConfig(
            table_name="table_two",
            primary_keys=["key"]
        )
        manager2 = FeatureStoreManager(spark, mock_project_config, config2)
        assert manager2.full_table_name() == "test_catalog.test_schema.table_two"

    def test_multiple_primary_keys(self, spark, mock_project_config, feature_store_config,
                                   mock_feature_store_client, sample_dataframe):
        """Testa com múltiplas primary keys"""
        manager = FeatureStoreManager(spark, mock_project_config, feature_store_config)
        
        primary_keys = ["id", "semana", "vendas"]
        manager.create_feature_table(
            df=sample_dataframe,
            primary_keys=primary_keys
        )
        
        call_args = mock_feature_store_client.create_table.call_args
        assert len(call_args.kwargs['primary_keys']) == 3
        assert call_args.kwargs['primary_keys'] == primary_keys

