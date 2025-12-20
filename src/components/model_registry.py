from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from loguru import logger
from entity.config_entity import ModelRegistryConfig

class ModelRegistry:

    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.client = MlflowClient()
        self.model_name = self.config.model_name

        logger.info(f"[ModelRegistry] Inicializado com model_name='{self.model_name}'")

    def register_model(
        self, 
        run_id: str, 
        model_uri: Optional[str] = None,
        model_path: str = "model", 
        description: Optional[str] = None
    ):
        """
        Registra um modelo no MLflow Model Registry e adiciona uma descrição opcional.
        """

        # Construindo URI caso não tenha sido fornecida
        if model_uri is None:
            model_uri = f"runs:/{run_id}/{model_path}"
            logger.debug(f"[register_model] model_uri não informado. Construído automaticamente: {model_uri}")

        logger.info(f"[register_model] Registrando modelo '{self.model_name}' a partir de: {model_uri}")

        try:
            # Registro do modelo
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name
            )

            logger.success(f"[register_model] Modelo registrado! Nome='{self.model_name}', Versão={model_version.version}")

            # Atualizar descrição (se houver)
            if description is not None:
                logger.info(f"[register_model] Adicionando descrição à versão {model_version.version}")

                self.client.update_model_version(
                    name=self.model_name,
                    version=model_version.version,
                    description=description
                )

                logger.success(f"[register_model] Descrição adicionada com sucesso à versão {model_version.version}")

            return model_version.version

        except MlflowException as e:
            logger.error(f"[register_model] Falha ao registrar modelo '{self.model_name}': {e}")
            raise e
            


    def transition_model_stage(
        self, 
        model_version: int, 
        stage: str, 
        archive_existing_versions: bool = False
    ):
        """
        Transiciona o stage de uma versão de modelo no MLflow Model Registry.
        """

        valid_stages = ['Production', 'Staging', 'Archived', 'None']

        if stage not in valid_stages:
            raise ValueError(f"Stage inválido. Deve ser um dos seguintes: {', '.join(valid_stages)}")

        logger.info(
            f"[transition_model_stage] "
            f"Tentando mover '{self.model_name}' versão {model_version} para stage '{stage}' "
            f"(archive_existing_versions={archive_existing_versions})"
        )

        try:
            result = self.client.transition_model_version_stage(
                name=self.model_name,   
                version=model_version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )

            logger.success(
                f"[transition_model_stage] "
                f"Versão {model_version} do modelo '{self.model_name}' movida para '{stage}' com sucesso!"
            )

            return result

        except MlflowException as e:
            logger.error(
                f"[transition_model_stage] Falha ao atualizar stage do modelo '{self.model_name}': {e}"
            )
            raise e
    
    def get_model_versions(self, max_results: int = 5):
        """
        Retorna as versões do modelo registrado no MLflow.
        """

        logger.info(
            f"[get_model_versions] Buscando até {max_results} versões do modelo '{self.model_name}'."
        )

        try:
            versions = self.client.search_model_versions(
                filter_string=f"name='{self.model_name}'",
                max_results=max_results
            )

            logger.success(
                f"[get_model_versions] Encontradas {len(versions)} versões para o modelo '{self.model_name}'."
            )

            return versions

        except MlflowException as e:
            logger.error(
                f"[get_model_versions] Falha ao buscar versões do modelo '{self.model_name}': {e}"
            )
            raise e

    def get_latest_model_version(self, stages:Optional[str]=None):

        try:
            if stages is None:
                stages = ["None"]

            logger.info(f"[get_latest_model_version] Buscando últimas versões do modelo '{self.model_name}' para os estágios: {stages}")

            latest_versions = self.client.get_latest_versions(self.model_name, stages=stages)

            if not latest_versions:
                logger.warning(f"[get_latest_model_version] Nenhuma versão encontrada para o modelo '{self.model_name}' nos estágios: {stages}")
                return {
                    "model_name": self.model_name,
                    "versions": []
                }

            versions_info = []
            for mv in latest_versions:
                versions_info.append(
                    {
                        "version": mv.version,
                        "current_stage": mv.current_stage,
                        "run_id": mv.run_id,
                        "description": getattr(mv, "description", None),
                        "status": getattr(mv, "status", None),
                    }
                )

            return {
                "model_name": self.model_name,
                "versions": versions_info,
            }

        except MlflowException as e:
            logger.error(
                f"[get_latest_model_version] Falha ao buscar versões do modelo '{self.model_name}': {e}"
            )
            raise e

    def load_model(self, stage: Optional[str] = None, version: Optional[str] = None):

        logger.info(f"[load_model] Iniciando carregamento do modelo '{self.model_name}'.")

        if stage:
            model_uri = f"models:/{self.model_name}/{stage}"
            logger.info(f"[load_model] Carregando modelo pelo stage: {stage}")

        elif version:
            model_uri = f"models:/{self.model_name}/{version}"
            logger.info(f"[load_model] Carregando modelo pela versão: {version}")

        else:
            logger.info("[load_model] Nenhum stage/versão informado. Procurando última versão em 'Production'.")
            latest = self.get_latest_model_version(stages=["Production"])
            versions = latest.get("versions", [])

            if not versions:
                raise ValueError("Nenhum modelo no stage 'Production' para carregar.")

            latest_version = versions[0]["version"]

            model_uri = f"models:/{self.model_name}/{latest_version}"
            logger.info(f"[load_model] Carregando a última versão em Production: {latest_version}")

        try:
            logger.info(f"[load_model] Model URI: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("[load_model] Modelo carregado com sucesso.")

            return model

        except MlflowException as e:
            logger.error(
                f"[load_model] Falha ao carregar modelo '{self.model_name}' "
                f"no caminho '{model_uri}': {e}"
            )
            raise e
        

    
    def delete_model_version(self, version):

        logger.info(f"[delete_model_version] Iniciando remoção da versão {version} do modelo '{self.model_name}'.")

        try:
            self.client.delete_model_version(
                name=self.model_name,
                version=version
            )

            logger.success(f"[delete_model_version] Versão {version} do modelo '{self.model_name}' removida com sucesso.")

            return {
                "model_name": self.model_name,
                "version": version,
                "status": "deleted"
            }

        except MlflowException as e:
            logger.error(
                f"[delete_model_version] Falha ao deletar a versão {version} "
                f"do modelo '{self.model_name}': {e}"
            )
            raise e


    def add_model_tag(self, key, value, version):

        if not key or not isinstance(key, str):
            raise ValueError("[add_model_tag] 'key' deve ser uma string não vazia.")

        logger.info(
            f"[add_model_tag] Adicionando tag '{key}={value}' ao modelo '{self.model_name}' "
            f"(versão {version})."
        )

        try:
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key=key,
                value=value
            )

            logger.success(
                f"[add_model_tag] Tag '{key}={value}' adicionada com sucesso "
                f"ao modelo '{self.model_name}' versão {version}."
            )

            return {
                "model_name": self.model_name,
                "version": version,
                "tag_key": key,
                "tag_value": value,
                "status": "tag_added"
            }

        except MlflowException as e:
            logger.error(
                f"[add_model_tag] Falha ao adicionar tag ao modelo '{self.model_name}' "
                f"versão {version}: {e}"
            )
            raise e