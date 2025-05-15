# mlops-fakenews-text-classification

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml


## MLflow

##### cmd
- mlflow ui

### dagshub

MLFLOW_TRACKING_URI=https://dagshub.com/achrafHR/mlops-fakenews-text-classification.mlflow
MLFLOW_TRACKING_USERNAME=achrafHR
MLFLOW_TRACKING_PASSWORD=ad1fbb1078a58c4bfc663e0780d26ec6d22dabb7
python script.py

Run this to export as env variables:

```bash
set MLFLOW_TRACKING_URI=https://dagshub.com/achrafHR/mlops-fakenews-text-classification.mlflow

set MLFLOW_TRACKING_USERNAME=achrafHR

set MLFLOW_TRACKING_PASSWORD=ad1fbb1078a58c4bfc663e0780d26ec6d22dabb7
```
