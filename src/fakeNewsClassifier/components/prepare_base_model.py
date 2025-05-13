import os
import urllib.request as request
from zipfile import ZipFile
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from fakeNewsClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = LogisticRegression(
            C=self.config.params_C,
            penalty=self.config.params_penalty,
            solver=self.config.params_solver,
            class_weight=self.config.params_class_weight,
            max_iter=self.config.params_max_iter
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    @staticmethod
    def save_model(path: Path, model: LogisticRegression):
        joblib.dump(model, path)