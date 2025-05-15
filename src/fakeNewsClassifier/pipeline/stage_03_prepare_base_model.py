from fakeNewsClassifier.config.configuration import ConfigurationManager
from fakeNewsClassifier.components.prepare_base_model import PrepareBaseModel
from fakeNewsClassifier import logger
from fakeNewsClassifier.utils.common import timer_decorator
STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    @timer_decorator(STAGE_NAME)    
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()


if __name__ == '__main__':
    try:
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e
