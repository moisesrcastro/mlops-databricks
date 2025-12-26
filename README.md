# MLOps Databricks Project

## Overview
This project demonstrates a **Minimum Viable Product (MVP) MLOps pipeline** implemented in Databricks. The pipeline covers **data ingestion, model training, evaluation, registration in MLflow Model Registry, deployment, and monitoring**, providing a complete flow to ensure models deliver value consistently.  

The project was developed as a learning experience while building end-to-end MLOps capabilities, including retraining and model serving automation.

---

## Features

- **Data Loading & Inspection**: Load datasets from DBFS, Delta tables, or other sources and perform basic inspections (`.shape`, `.dtypes`, null checks).  
- **Preprocessing**: Minimal preprocessing with missing value handling and feature selection.  
- **Train/Test Split**: Ensure reproducibility using a fixed random seed.  
- **Model Training**: Train simple models (e.g., `PassiveAggressiveRegressor`) to demonstrate the workflow.  
- **Metrics & Evaluation**: Compute relevant metrics like MAE, RMSE, and R² for regression tasks.  
- **MLflow Tracking & Registry**: Log parameters, metrics, and artifacts; register models in the MLflow Model Registry with automatic versioning.  
- **Deployment**: Deploy models to Databricks Serving Endpoints for real-time inference.  
- **Validation & Monitoring**: Test endpoints with sample data and monitor status for smooth operations.  
- **Retraining Pipeline**: Automated retraining and versioning of models with consistent logging and metrics tracking.

---

## Project Structure
```
mlops-databricks/
├── notebooks/                 
│   ├── deploy_model.py
│   ├── model_monitor.py
│   ├── train_model.py
│   └── validate_model.py
├── src/                       
│   ├── config/
│   │   └── config.py          
│   ├── components/
│   │   ├── model_registry.py 
│   │   ├── model_trainer.py   
│   │   ├── model_deployer.py 
│   │   ├── model_validator.py
│   │   ├── model_monitor.py
│   │   ├── data_processor.py
│   │   └── feature_store.py
│   └── utils/
│       └── commons.py         
├── tests/                    
├── requirements.txt           
└── README.md                 
```
