from pathlib import Path
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, log_loss
from fakeNewsClassifier.utils.common import load_bin, save_json
from fakeNewsClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def get_test_data(self):
        self.X_test = load_bin(self.config.processed_x_test)
        self.y_test = load_bin(self.config.processed_y_test)
    
    def evaluation(self):
        self.model = load_bin(self.config.path_of_model)
        # Pour régression logistique
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calcul des métriques pour régression logistique
        
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.loss = log_loss(self.y_test, self.y_pred_proba)
        self.score = [self.loss, self.accuracy]  
        
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="LogisticRegression")
            else:
                mlflow.sklearn.log_model(self.model, "model")