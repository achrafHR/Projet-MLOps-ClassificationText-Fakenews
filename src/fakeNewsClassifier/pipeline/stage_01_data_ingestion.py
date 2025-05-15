from fakeNewsClassifier.config.configuration import ConfigurationManager
from fakeNewsClassifier.components.data_ingestion import DataIngestion
from fakeNewsClassifier import logger
from fakeNewsClassifier.utils.common import timer_decorator

STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    @timer_decorator(STAGE_NAME)
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        obj = DataIngestionTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e