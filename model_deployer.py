from databricks.sdk import WorkspaceClient
from src.entity.config_entity import ModelDeployerConfig
from typing import Optional, Dict, Any
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
)
from loguru import logger
import requests
from databricks.sdk.service.serving import ServedModelInputWorkloadSize


class ModelDeployer:

    def __init__(self, config: ModelDeployerConfig):
        self.config = config
        self.model_name = self.config.model_name
        self.endpoint_name = self.config.endpoint_name
        self.workspace_client = WorkspaceClient()

    def create_endpoint(
        self,
        model_version: str,
    ) -> Dict[str, Any]:

        served_model = ServedModelInput(
            model_name=self.model_name,
            model_version=str(model_version),
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
        )

        endpoint_config = EndpointCoreConfigInput(
            served_models=[served_model],
        )

        endpoint = self.workspace_client.serving_endpoints.create(
            name=self.endpoint_name,
            config=endpoint_config,
        )

        logger.info("Endpoint creation requested: {}", endpoint.response.name)

        return {
            "name": endpoint.response.name,
            "creation_timestamp": endpoint.response.creation_timestamp,
        }

    def update_endpoint(
        self,
        model_version: str,
    ) -> Dict[str, Any]:

        served_model = ServedModelInput(
            model_name=self.model_name,
            model_version=str(model_version),
            workload_size="Small",
        )

        endpoint_config = EndpointCoreConfigInput(
            served_models=[served_model],
        )

        endpoint = self.workspace_client.serving_endpoints.update_config(
            name=self.endpoint_name,
            config=endpoint_config,
        )

        logger.info("Endpoint update requested: {}", endpoint.response.name)

        return {
            "name": endpoint.response.name,
            "state": endpoint.response.state,
        }

    def get_endpoint(self) -> Optional[Dict[str, Any]]:
        try:
            endpoint = self.workspace_client.serving_endpoints.get(
                name=self.endpoint_name
            )

            return {
                "name": endpoint.response.name,
                "creation_timestamp": endpoint.response.creation_timestamp,
                "config": {
                    "served_models": [
                        {
                            "model_name": sm.model_name,
                            "model_version": sm.model_version,
                            "workload_size": sm.workload_size,
                        }
                        for sm in endpoint.response.pending_config
                    ]
                },
            }

        except Exception:
            return None

    def delete_endpoint(self) -> None:
        self.workspace_client.serving_endpoints.delete(
            name=self.endpoint_name
        )
        logger.info("Endpoint deleted: {}", self.endpoint_name)

    def test_endpoint(
        self,
        test_data: Dict[str, Any],
        endpoint_url: Optional[str] = None,
    ) -> Dict[str, Any]:

        if endpoint_url is None:
            endpoint_url = (
                f"{self.workspace_client.config.host}"
                f"/serving-endpoints/{self.endpoint_name}/invocations"
            )

        token = self.workspace_client.config.token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            endpoint_url,
            json={"dataframe_records": [test_data]},
            headers=headers,
        )

        response.raise_for_status()
        return response.json()
