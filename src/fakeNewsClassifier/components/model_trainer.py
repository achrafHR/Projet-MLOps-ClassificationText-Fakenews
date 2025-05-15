from fakeNewsClassifier.utils.common import load_bin
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib
from fakeNewsClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = load_bin(
            self.config.base_model_path
        )
    
    def get_train_data(self):
        self.X = load_bin(self.config.processed_x_train)
        self.y = load_bin(self.config.processed_y_train)


    @staticmethod
    def save_model(path: Path, model: LogisticRegression):
        joblib.dump(model, path)
    
    def train(self):

        self.model.fit(
            self.X,
            self.y
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model)
        
        self.save_model(path=Path("model/trained_model.pkl"),
                        model=self.model)