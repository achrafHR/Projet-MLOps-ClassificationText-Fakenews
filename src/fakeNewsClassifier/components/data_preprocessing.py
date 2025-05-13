#import os
#import urllib.request as request
#from zipfile import ZipFile
#import joblib
from fakeNewsClassifier.utils.common import *


import pandas as pd
import numpy as np
#import re
#import string

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from fakeNewsClassifier.entity.config_entity import DataPreprocessingConfig
from fakeNewsClassifier.utils.common import *
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def preprocess_data(self) -> None:
        """
        Main method to preprocess data
        """
        try:
            # Read raw data
            logger.info("Reading raw data")
            df = self._load_data()
            
            # Apply preprocessing steps
            logger.info("Applying preprocessing steps")
            X, y, tfidf = self._apply_preprocessing(df)
            
            # Split data
            logger.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            
            # Save processed data

            logger.info("Saving processed data")
            save_bin(X_train, Path(self.config.processed_x_train))
            save_bin(X_test, Path(self.config.processed_x_test))
            save_bin(y_train, Path(self.config.processed_y_train))
            save_bin(y_test, Path(self.config.processed_y_test))
            save_bin(tfidf, Path(self.config.tfidf_vectoriser))
            
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise e
        

    def _load_data(self) -> pd.DataFrame:
        """
        Load and combine fake and true news data
        
        Returns:
            pd.DataFrame: Combined raw data with labels
        """
        try:
            logger.info(f"Loading fake news data from {self.config.fake_news_path}")
            fake_df = pd.read_csv(self.config.fake_news_path)
            logger.info(f"Loaded {len(fake_df)} fake news articles")
            
            # Add label for fake news (1)
            fake_df["class"] = 1
            
            logger.info(f"Loading true news data from {self.config.true_news_path}")
            true_df = pd.read_csv(self.config.true_news_path)
            logger.info(f"Loaded {len(true_df)} true news articles")
            
            # Add label for true news (0)
            true_df["class"] = 0
            
            # Combine datasets
            combined_df = pd.concat([fake_df, true_df], ignore_index=True)
            combined_df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
            logger.info(f"Combined dataset with {len(combined_df)} articles")

            # Drop rows with missing text
            combined_df = combined_df.dropna(subset=['text'])
            logger.info(f"Delete NaN rows")
            
            # Shuffle data
            shuffled_df = combined_df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
            logger.info("Dataset shuffled")
            
            return shuffled_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise 


        
    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to raw data
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: X_tfidf , y and vectoriser 
        """
        try:
            df.loc[df["class"] == 0, "text"] = df.loc[df["class"] == 0, "text"].str.replace(r"\(Reuters\)", "", regex=True)
            
            # Fill nulls and combine title and text
            df['text'] = df['text'].fillna('')
            df['title'] = df['title'].fillna('')
            df['content'] = df['title'] + ' ' + df['text']
            df['cleaned_text'] = df['content'].apply(clean_text)

            tfidf = TfidfVectorizer(max_features=5000)
            X = tfidf.fit_transform(df['cleaned_text'])
            y = df['class']

            # Return processed data
            return X, y, tfidf
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise e
        
    
    def _split_data(self, X , y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df (): Processed data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test data
        """
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