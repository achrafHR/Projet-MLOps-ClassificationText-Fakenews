from fakeNewsClassifier.constants import *
from fakeNewsClassifier.utils.common import read_yaml, create_directories, save_json
from fakeNewsClassifier.entity.config_entity import (DataIngestionConfig,
                                                      DataPreprocessingConfig,
                                                      PrepareBaseModelConfig,
                                                      TrainingConfig, 
                                                      EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Get configuration for data preprocessing
        
        Returns:
            DataPreprocessingConfig: Configuration for data preprocessing
        """
        config = self.config.data_preprocessing
        
        create_directories([config.root_dir])
        
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            fake_news_path=config.fake_news_path,
            true_news_path=config.true_news_path,
            processed_x_train=config.processed_x_train,
            processed_y_train=config.processed_y_train,
            processed_x_test=config.processed_x_test,
            processed_y_test=config.processed_y_test,
            tfidf_vectoriser=config.tfidf_vectoriser,
            test_size=self.params.test_size,
            random_state=self.params.random_state,
            stratify=self.params.stratify,
        )
        
        return data_preprocessing_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_C=self.params.C,
            params_penalty = self.params.penalty,
            params_solver = self.params.solver,
            params_class_weight = self.params.class_weight,
            params_max_iter = self.params.max_iter,
        )

        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        data_preprocessing = self.config.data_preprocessing
        
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            base_model_path= Path(prepare_base_model.base_model_path),
            trained_model_path=Path(training.trained_model_path),
            processed_x_train= Path(data_preprocessing.processed_x_train),
            processed_y_train= Path(data_preprocessing.processed_y_train),
            processed_x_test= Path(data_preprocessing.processed_x_test),
            processed_y_test= Path(data_preprocessing.processed_y_test),
            tfidf_vectoriser= Path(data_preprocessing.tfidf_vectoriser)

        )

        return training_config
      

    def get_evaluation_config(self) -> EvaluationConfig:

        model_params_keys = ["C", "penalty", "solver", "max_iter", "class_weight"]
        model_params = {k: self.params[k] for k in model_params_keys if k in self.params}

        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/trained_model.pkl"),
            processed_x_test=Path("artifacts/data_preprocessing/X_test_tfidf.pkl"),
            processed_y_test=Path("artifacts/data_preprocessing/y_test.pkl"),
            mlflow_uri="https://dagshub.com/achrafHR/mlops-fakenews-text-classification.mlflow",
            params=model_params,
        )
        return eval_config  