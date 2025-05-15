from fakeNewsClassifier.utils.common import *
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Generator, Iterator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from fakeNewsClassifier.entity.config_entity import DataPreprocessingConfig
import warnings
warnings.filterwarnings('ignore')

class ProgressiveTransformer:
    def __init__(self):
        self._current_step = 0
        self._setup_transformations()

    def _setup_transformations(self):
        """Définit la séquence des transformations"""
        self.transformations = [
            {'name': 'remove_reuters', 'function': self._remove_reuters},
            {'name': 'combine_title_text', 'function': self._combine_title_text},
            {'name': 'clean_text_content', 'function': self._clean_text_content}
        ]
        
    def __iter__(self) -> Iterator:
        """Rend la classe itérable"""
        self._current_step = 0
        return self
        
    def __next__(self) -> Dict:
        """Retourne la prochaine transformation à appliquer"""
        if self._current_step < len(self.transformations):
            transform = self.transformations[self._current_step]
            self._current_step += 1
            return transform
        else:
            raise StopIteration
            
    def apply_transform(self, df: pd.DataFrame, transform: Dict) -> pd.DataFrame:
        """Applique une transformation spécifique au dataframe"""
        logger.info(f"Applying transformation: {transform['name']}")
        return transform['function'](df)

    def apply_next(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique la prochaine transformation"""
        if self._current_step < len(self.transformations):
            transform = self.transformations[self._current_step]
            logger.info(f"Applying transformation: {transform['name']}")
            df = transform['function'](df)
            self._current_step += 1
        return df

    def reset(self):
        """Réinitialise le générateur"""
        self._current_step = 0

    @staticmethod
    def _remove_reuters(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df["class"] == 0, "text"] = df.loc[df["class"] == 0, "text"].str.replace(
            r"\(Reuters\)", "", regex=True)
        return df

    @staticmethod
    def _combine_title_text(df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['text'].fillna('')
        df['title'] = df['title'].fillna('')
        df['content'] = df['title'] + ' ' + df['text']
        return df

    @staticmethod
    def _clean_text_content(df: pd.DataFrame) -> pd.DataFrame:
        df['cleaned_text'] = df['content'].apply(clean_text)
        return df

class LazyDataLoader:
    def __init__(self, config: DataPreprocessingConfig, batch_size: int = 5000):
        self.config = config
        self.batch_size = batch_size
        self.transformer = ProgressiveTransformer()
        
    def load_data_lazily(self) -> Generator[pd.DataFrame, None, None]:
        """
        Lazy data loading generator with progressive transformations
        """
        try:
            # Load fake news in chunks
            logger.info(f"Loading fake news data from {self.config.fake_news_path}")
            for fake_chunk in pd.read_csv(self.config.fake_news_path, chunksize=self.batch_size):
                fake_chunk["class"] = 1
                yield from self._process_chunk(fake_chunk)
                
            # Load true news in chunks
            logger.info(f"Loading true news data from {self.config.true_news_path}")
            for true_chunk in pd.read_csv(self.config.true_news_path, chunksize=self.batch_size):
                true_chunk["class"] = 0
                yield from self._process_chunk(true_chunk)
                
        except Exception as e:
            logger.error(f"Error in lazy data loading: {str(e)}")
            raise e
            
    def _process_chunk(self, chunk: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """
        Process each chunk through the transformer pipeline
        """
        chunk = chunk.dropna(subset=['text'])
        
        # Utilisation du protocole d'itérateur de ProgressiveTransformer
        transformer = ProgressiveTransformer()
        for transform in transformer:
            chunk = transformer.apply_transform(chunk, transform)
            
        # Shuffle the chunk to introduce randomness
        chunk = chunk.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
        
        # Yield the processed chunk
        yield chunk

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.transformer = ProgressiveTransformer()
        self.lazy_loader = LazyDataLoader(config)
        # Flag pour contrôler si on utilise le chargement paresseux
        self.use_lazy_loading = True

    def preprocess_data(self) -> None:
        """
        Main method to preprocess data (interface inchangée)
        """
        try:
            if self.use_lazy_loading:
                logger.info("Using lazy loading approach for data processing")
                # Collect all processed chunks
                all_chunks = []
                for chunk in self.lazy_loader.load_data_lazily():
                    all_chunks.append(chunk)
                    logger.info(f"Processed chunk of size {len(chunk)}")
                
                # Combine all chunks
                df = pd.concat(all_chunks, ignore_index=True)
                logger.info(f"Combined dataset with {len(df)} articles after lazy loading")
            else:
                # Approche originale en mémoire
                logger.info("Using in-memory approach for data processing")
                df = self._load_data()
                
                # Utilisation du protocole d'itérateur de ProgressiveTransformer
                for transform in self.transformer:
                    df = self.transformer.apply_transform(df, transform)
            
            # Vectorisation et split (commun aux deux approches)
            X, y, tfidf = self._vectorize_data(df)
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            
            # Sauvegarde (identique à l'original)
            save_bin(X_train, Path(self.config.processed_x_train))
            save_bin(X_test, Path(self.config.processed_x_test))
            save_bin(y_train, Path(self.config.processed_y_train))
            save_bin(y_test, Path(self.config.processed_y_test))
            save_bin(tfidf, Path(self.config.tfidf_vectoriser))
            save_bin(tfidf, Path("vectorizer/tfidf_vectoriser.pkl"))
            
            logger.info("Data preprocessing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise e

    def _load_data(self) -> pd.DataFrame:
        """Même code que votre implémentation originale"""
        try:
            logger.info(f"Loading fake news data from {self.config.fake_news_path}")
            fake_df = pd.read_csv(self.config.fake_news_path)
            logger.info(f"Loaded {len(fake_df)} fake news articles")
            fake_df["class"] = 1
            
            logger.info(f"Loading true news data from {self.config.true_news_path}")
            true_df = pd.read_csv(self.config.true_news_path)
            logger.info(f"Loaded {len(true_df)} true news articles")
            true_df["class"] = 0
            
            combined_df = pd.concat([fake_df, true_df], ignore_index=True)
            combined_df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
            logger.info(f"Combined dataset with {len(combined_df)} articles")

            combined_df = combined_df.dropna(subset=['text'])
            logger.info(f"Delete NaN rows")
            
            shuffled_df = combined_df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
            logger.info("Dataset shuffled")
            
            return shuffled_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise 

    def _vectorize_data(self, df: pd.DataFrame) -> Tuple:
        """Vectorize the cleaned text"""
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(df['cleaned_text'])
        y = df['class']
        return X, y, tfidf

    def _split_data(self, X, y: pd.Series) -> Tuple:
        """Même code que votre implémentation originale"""
        try:
            if self.config.stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    stratify=y,
                    random_state=self.config.random_state
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state
                )
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in splitting data: {str(e)}")
            raise e