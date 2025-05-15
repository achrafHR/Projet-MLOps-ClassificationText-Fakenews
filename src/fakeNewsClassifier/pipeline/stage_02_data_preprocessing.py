from fakeNewsClassifier.config.configuration import ConfigurationManager
from fakeNewsClassifier.components.data_preprocessing import DataPreprocessing
from fakeNewsClassifier import logger
from fakeNewsClassifier.utils.common import timer_decorator

STAGE_NAME = "Data Preprocessing"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    @timer_decorator(STAGE_NAME)
    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.preprocess_data()


if __name__ == '__main__':
    try:
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e


