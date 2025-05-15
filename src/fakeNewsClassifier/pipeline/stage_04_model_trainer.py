from fakeNewsClassifier.config.configuration import ConfigurationManager
from fakeNewsClassifier.components.model_trainer import Training
from fakeNewsClassifier import logger
from fakeNewsClassifier.utils.common import timer_decorator
STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    @timer_decorator(STAGE_NAME)
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.get_train_data()
        training.train()


if __name__ == '__main__':
    try:
        obj = ModelTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e
