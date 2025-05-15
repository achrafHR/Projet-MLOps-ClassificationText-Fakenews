from fakeNewsClassifier import logger
from fakeNewsClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fakeNewsClassifier.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from fakeNewsClassifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from fakeNewsClassifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from fakeNewsClassifier.pipeline.stage_05_model_evaluation import EvaluationPipeline



STAGE_NAME = "Data Ingestion"

try:
    obj = DataIngestionTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Preprocessing"

try:
    obj = DataPreprocessingTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"

try:
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"

if __name__ == '__main__':
    try:
        obj = ModelTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME = "Evaluation"
try:
   logger.info(f"*******************")
   model_evalution = EvaluationPipeline()
   model_evalution.main()

except Exception as e:
        logger.exception(e)
        raise e