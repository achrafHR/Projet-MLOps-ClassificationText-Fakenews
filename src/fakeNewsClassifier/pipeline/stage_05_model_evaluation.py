from fakeNewsClassifier.config.configuration import ConfigurationManager
from fakeNewsClassifier.components.model_evaluation_mlflow import Evaluation
from fakeNewsClassifier import logger
from fakeNewsClassifier.utils.common import timer_decorator 

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    @timer_decorator(STAGE_NAME)
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.get_test_data()
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        obj = EvaluationPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e
