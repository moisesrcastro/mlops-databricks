from databricks.sdk import WorkspaceClient
from utils.config_entity import ModelDeployerConfig
from typing import Optional, List, Dict, Any
from databricks.sdk.service.serving import ( 
    EndpointCoreConfigInput, 
    ServedModelInput) 
from loguru import logger
import requests

class ModelDeployer:

    def __init__(self, config: ModelDeployerConfig):
        
        self.config = config
        self.model_name = self.config.model_name
        self.endpoint_name = self.config.endpoint_name
        self.workspace_client = WorkspaceClient()

    def create_endpoint(self, 
                        model_version: str="Production", 
                        workload_size:str="Small", 
                        scale_to_zero_enabled:bool=True,
                        min_provisioned_throughput:int=1,
                        max_provisioned_throughput:int=5) -> Dict[str, Any]:
        "Create a new model serving endpoint"


        served_model = ServedModelInput(
                                        model_name=self.model_name, 
                                        model_version=model_version if model_version else None, 
                                        workload_size=workload_size, 
                                        scale_to_zero_enabled=scale_to_zero_enabled, 
                                        min_provisioned_throughput=min_provisioned_throughput, 
                                        max_provisioned_throughput=max_provisioned_throughput
            ) 

        endpoint_config = EndpointCoreConfigInput(name=self.endpoint_name, served_models=[served_model], ) 

        try: 
            endpoint = self.workspace_client.serving_endpoints.create(endpoint_config) 
            logger.info(f"Endpoint created successfully: {endpoint.name}") 
            logger.info(f"Endpoint state: {endpoint.state}")

            return { 
                    "name": endpoint.name, 
                    "state": endpoint.state, 
                    "creation_timestamp": endpoint.creation_timestamp, 
                    }

        except Exception as error:
            logger.error(f"Error creating endpoint: {error}") 
            raise

    def update_endpoint(self,
                        model_version: str="Production") -> Dict[str, Any]:
        "Update an existing endpoint"

        served_model = ServedModelInput(
                                        model_name=self.model_name, 
                                        model_version=model_version if model_version else None, 
            ) 
        
        endpoint_config = EndpointCoreConfigInput(name=self.endpoint_name, served_models=[served_model], )

        try:
            endpoint = self.workspace_client.serving_endpoints.update_config(endpoint_config)
            logger.info(f"Endpoint updated successfully: {endpoint.name}") 
            logger.info(f"Endpoint state: {endpoint.state}")
            return { 
                    "name": endpoint.name, 
                    "state": endpoint.state
                    }

        except Exception as error:
            logger.error(f"Error updating endpoint: {error}")
            return None

    def get_endpoint(self) -> Dict[str, Any]:
        "Get information about an edpoint"
        try:
            endpoint = self.workspace_client.serving_endpoints.get(name=self.endpoint_name)

            return { 
                "name": endpoint.name, 
                "state": endpoint.state, 
                "creation_timestamp": endpoint.creation_timestamp, 
                "config": 
                    { "served_models": 
                        [ 
                            { 
                            "model_name": sm.model_name, 
                            "model_version": sm.model_version, 
                            "workload_size": sm.workload_size, 
                            } 
                        for sm in endpoint.config.served_models ], }, }
        
        except Exception as error:
            logger.warning(f"Endpoint not found or error: {error}") 
            return None
    
    def delete_endpoint(self):

        logger.warning(f"Deleting endpoint: {self.endpoint_name}") 
        try: 
            
            logger.info(f"Endpoint deleted: {self.endpoint_name}")
            self.workspace_client.serving_endpoints.delete(name=self.endpoint_name)
        
        except Exception as error: 
            logger.error(f"Error deleting endpoint: {error}") 
            raise

    def test_endpoint(self, test_data:Dict[str, Any], endpoint_url:Optional) -> Dict[str, Any]:
        "Testing an endpoint with sample data"

        if endpoint_url is None: 
            endpoint_url = f"{self.workspace_client.config.host}/serving-endpoints/{self.endpoint_name}/invocations" 
        if not endpoint_url: 
            raise ValueError("Endpoint URL not available. Endpoint may not exist.") 
        logger.info(f"Testing endpoint: {endpoint_url}") 

        token = self.workspace_client.config.token 
        headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json", } 
        try: 
            response = requests.post( endpoint_url, json={"dataframe_records": [test_data]}, headers=headers, ) 
            response.raise_for_status() 
            result = response.json() 
            logger.info("Endpoint test successful") 
            return result 
        except Exception as e: 
            logger.error(f"Error testing endpoint: {e}") 
            raise


