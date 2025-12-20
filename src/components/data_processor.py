from entity.config_entity import DataProcessorConfig
from pyspark.sql import functions as F
from pyspark.sql.window import Window

class DataProcessor:

    def __init__(self, spark, config:DataProcessorConfig):
        self.spark = spark
        self.config = config

    def load_source(self):
        query = self.config.sql_query
        return self.spark.sql(query)

    def aggregate_daily_to_weekly(self, df):
        return (
            df
            .withColumn("data", F.to_date("data"))
            .withColumn("semana", F.date_trunc("week", "data"))
            .groupBy("semana")
            .agg(
                *[F.sum(c).alias(f"{c}_semana") for c in self.config.value_columns]
            )
        )

    def aggregate_daily_to_monthly(self, df):
        return (
            df
            .withColumn("data", F.to_date("data"))
            .withColumn("mes", F.date_trunc("month", "data"))
            .groupBy("mes")
            .agg(
                *[F.sum(c).alias(f"{c}_mes") for c in self.config.value_columns]
            )
        )

    def join_weekly_monthly(self, df_semana, df_mes):
        df_semana = df_semana.withColumn(
            "mes",
            F.date_trunc("month", F.col("semana"))
        )
        return df_semana.join(df_mes, on="mes", how="left")

    def add_weekly_lags(self, df):
        w = Window.orderBy("semana")

        for col in [f"{c}_semana" for c in self.config.value_columns]:
            for lag in (1, 2, 3):
                df = df.withColumn(
                    f"{col}_lag{lag}w",
                    F.lag(col, lag).over(w)
                )
        return df

    def add_monthly_lags(self, df):
        w = Window.orderBy("mes")

        for col in [f"{c}_mes" for c in self.config.value_columns]:
            for lag in (1, 2, 3):
                df = df.withColumn(
                    f"{col}_lag{lag}m",
                    F.lag(col, lag).over(w)
                )
        return df

    def build(self):
        df_daily = self.load_source()

        df_semana = self.aggregate_daily_to_weekly(df_daily)
        df_mes = self.aggregate_daily_to_monthly(df_daily)

        df = self.join_weekly_monthly(df_semana, df_mes)
        df = self.add_weekly_lags(df)
        df = self.add_monthly_lags(df)

        return df
    