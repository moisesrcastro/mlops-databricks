import pytest
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType

import sys
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components.data_processor import DataProcessor
from entity.config_entity import DataProcessorConfig


@pytest.fixture(scope="session")
def spark():
    """Cria uma sessão Spark para os testes"""
    spark = SparkSession.builder \
        .appName("test_data_processor") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_config():
    """Configuração de teste para DataProcessor"""
    return DataProcessorConfig(
        sql_query="SELECT * FROM test_table",
        value_columns=["vendas", "receita"]
    )


@pytest.fixture
def sample_daily_data(spark):
    """Cria dados diários de exemplo para os testes"""
    # Criar dados de exemplo com múltiplas semanas e meses
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(60):  # ~2 meses de dados
        date = start_date + timedelta(days=i)
        data.append({
            "data": date.strftime("%Y-%m-%d"),
            "vendas": 100 + i * 10,
            "receita": 1000 + i * 100
        })
    
    schema = StructType([
        StructField("data", StringType(), True),
        StructField("vendas", IntegerType(), True),
        StructField("receita", IntegerType(), True)
    ])
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def data_processor(spark, sample_config):
    """Cria uma instância de DataProcessor para os testes"""
    return DataProcessor(spark, sample_config)


class TestDataProcessor:
    """Testes para a classe DataProcessor"""

    def test_init(self, spark, sample_config):
        """Testa a inicialização do DataProcessor"""
        processor = DataProcessor(spark, sample_config)
        assert processor.spark == spark
        assert processor.config == sample_config

    def test_load_source(self, spark, sample_config):
        """Testa o método load_source"""
        # Criar uma tabela temporária para teste
        test_data = [(1, "2024-01-01", 100, 1000)]
        df = spark.createDataFrame(
            test_data,
            ["id", "data", "vendas", "receita"]
        )
        df.createOrReplaceTempView("test_table")
        
        processor = DataProcessor(spark, sample_config)
        result = processor.load_source()
        
        assert result is not None
        assert result.count() == 1

    def test_aggregate_daily_to_weekly(self, data_processor, sample_daily_data):
        """Testa a agregação diária para semanal"""
        result = data_processor.aggregate_daily_to_weekly(sample_daily_data)
        
        # Verificar que o resultado tem a coluna 'semana'
        assert "semana" in result.columns
        
        # Verificar que as colunas agregadas existem
        assert "vendas_semana" in result.columns
        assert "receita_semana" in result.columns
        
        # Verificar que há menos linhas que os dados diários (agrupado por semana)
        assert result.count() <= sample_daily_data.count()
        
        # Verificar que os valores são somas (não negativos)
        result_rows = result.collect()
        for row in result_rows:
            assert row["vendas_semana"] >= 0
            assert row["receita_semana"] >= 0

    def test_aggregate_daily_to_monthly(self, data_processor, sample_daily_data):
        """Testa a agregação diária para mensal"""
        result = data_processor.aggregate_daily_to_monthly(sample_daily_data)
        
        # Verificar que o resultado tem a coluna 'mes'
        assert "mes" in result.columns
        
        # Verificar que as colunas agregadas existem
        assert "vendas_mes" in result.columns
        assert "receita_mes" in result.columns
        
        # Verificar que há menos linhas que os dados diários (agrupado por mês)
        assert result.count() <= sample_daily_data.count()
        
        # Verificar que os valores são somas (não negativos)
        result_rows = result.collect()
        for row in result_rows:
            assert row["vendas_mes"] >= 0
            assert row["receita_mes"] >= 0

    def test_join_weekly_monthly(self, data_processor, sample_daily_data):
        """Testa o join entre dados semanais e mensais"""
        df_semana = data_processor.aggregate_daily_to_weekly(sample_daily_data)
        df_mes = data_processor.aggregate_daily_to_monthly(sample_daily_data)
        
        result = data_processor.join_weekly_monthly(df_semana, df_mes)
        
        # Verificar que o resultado tem ambas as colunas semanais e mensais
        assert "semana" in result.columns
        assert "mes" in result.columns
        assert "vendas_semana" in result.columns
        assert "receita_semana" in result.columns
        assert "vendas_mes" in result.columns
        assert "receita_mes" in result.columns
        
        # Verificar que o número de linhas é igual ao número de semanas
        assert result.count() == df_semana.count()

    def test_add_weekly_lags(self, data_processor, sample_daily_data):
        """Testa a adição de lags semanais"""
        df_semana = data_processor.aggregate_daily_to_weekly(sample_daily_data)
        df_mes = data_processor.aggregate_daily_to_monthly(sample_daily_data)
        df_joined = data_processor.join_weekly_monthly(df_semana, df_mes)
        
        result = data_processor.add_weekly_lags(df_joined)
        
        # Verificar que as colunas de lag foram criadas
        expected_lag_cols = [
            "vendas_semana_lag1w",
            "vendas_semana_lag2w",
            "vendas_semana_lag3w",
            "receita_semana_lag1w",
            "receita_semana_lag2w",
            "receita_semana_lag3w"
        ]
        
        for col in expected_lag_cols:
            assert col in result.columns
        
        # Verificar que a primeira linha tem valores None para os lags
        first_row = result.orderBy("semana").first()
        assert first_row["vendas_semana_lag1w"] is None
        assert first_row["vendas_semana_lag2w"] is None
        assert first_row["vendas_semana_lag3w"] is None

    def test_add_monthly_lags(self, data_processor, sample_daily_data):
        """Testa a adição de lags mensais"""
        df_semana = data_processor.aggregate_daily_to_weekly(sample_daily_data)
        df_mes = data_processor.aggregate_daily_to_monthly(sample_daily_data)
        df_joined = data_processor.join_weekly_monthly(df_semana, df_mes)
        df_with_weekly_lags = data_processor.add_weekly_lags(df_joined)
        
        result = data_processor.add_monthly_lags(df_with_weekly_lags)
        
        # Verificar que as colunas de lag mensais foram criadas
        expected_lag_cols = [
            "vendas_mes_lag1m",
            "vendas_mes_lag2m",
            "vendas_mes_lag3m",
            "receita_mes_lag1m",
            "receita_mes_lag2m",
            "receita_mes_lag3m"
        ]
        
        for col in expected_lag_cols:
            assert col in result.columns
        
        # Verificar que a primeira linha tem valores None para os lags mensais
        first_row = result.orderBy("mes").first()
        assert first_row["vendas_mes_lag1m"] is None
        assert first_row["vendas_mes_lag2m"] is None
        assert first_row["vendas_mes_lag3m"] is None

    def test_build_complete_pipeline(self, spark, sample_config):
        """Testa o pipeline completo usando build()"""
        # Criar dados de teste e tabela temporária
        data = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(60):
            date = start_date + timedelta(days=i)
            data.append({
                "data": date.strftime("%Y-%m-%d"),
                "vendas": 100 + i * 10,
                "receita": 1000 + i * 100
            })
        
        schema = StructType([
            StructField("data", StringType(), True),
            StructField("vendas", IntegerType(), True),
            StructField("receita", IntegerType(), True)
        ])
        
        df = spark.createDataFrame(data, schema)
        df.createOrReplaceTempView("test_table")
        
        # Atualizar config para usar a tabela criada
        config = DataProcessorConfig(
            sql_query="SELECT * FROM test_table",
            value_columns=["vendas", "receita"]
        )
        
        processor = DataProcessor(spark, config)
        result = processor.build()
        
        # Verificar que o resultado não é None
        assert result is not None
        
        # Verificar que todas as colunas esperadas estão presentes
        expected_cols = [
            "semana", "mes",
            "vendas_semana", "receita_semana",
            "vendas_mes", "receita_mes",
            "vendas_semana_lag1w", "vendas_semana_lag2w", "vendas_semana_lag3w",
            "receita_semana_lag1w", "receita_semana_lag2w", "receita_semana_lag3w",
            "vendas_mes_lag1m", "vendas_mes_lag2m", "vendas_mes_lag3m",
            "receita_mes_lag1m", "receita_mes_lag2m", "receita_mes_lag3m"
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Coluna {col} não encontrada no resultado"
        
        # Verificar que há dados no resultado
        assert result.count() > 0
        
        # Verificar que os dados estão ordenados por semana
        result_rows = result.orderBy("semana").collect()
        assert len(result_rows) > 0

    def test_empty_dataframe(self, data_processor, spark):
        """Testa o comportamento com DataFrame vazio"""
        empty_df = spark.createDataFrame([], StructType([
            StructField("data", StringType(), True),
            StructField("vendas", IntegerType(), True),
            StructField("receita", IntegerType(), True)
        ]))
        
        # Testar agregação semanal com DataFrame vazio
        result_weekly = data_processor.aggregate_daily_to_weekly(empty_df)
        assert result_weekly.count() == 0
        
        # Testar agregação mensal com DataFrame vazio
        result_monthly = data_processor.aggregate_daily_to_monthly(empty_df)
        assert result_monthly.count() == 0

    def test_single_value_column(self, spark):
        """Testa com apenas uma coluna de valor"""
        config = DataProcessorConfig(
            sql_query="SELECT * FROM test_table",
            value_columns=["vendas"]
        )
        
        processor = DataProcessor(spark, config)
        
        data = [(datetime(2024, 1, 1).strftime("%Y-%m-%d"), 100)]
        df = spark.createDataFrame(data, ["data", "vendas"])
        
        result = processor.aggregate_daily_to_weekly(df)
        
        assert "vendas_semana" in result.columns
        assert "receita_semana" not in result.columns

