import os
from box.exceptions import BoxValueError
import yaml
from fakeNewsClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')





@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)



def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"File saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> object:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



def clean_text(text):

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Initialize resources
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    """
    Comprehensive text cleaning function combining multiple preprocessing techniques.
    """
    # Convert to string and lowercase
    text = str(text).lower()
        
    # Remove special text patterns
    text = re.sub(r'\[.*?\]', '', text)        # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)         # Remove HTML tags
        
    # Remove numbers and words containing numbers
    text = re.sub(r'\w*\d\w*', '', text)       # Remove words containing digits
    text = re.sub(r'\d+', '', text)            # Remove standalone digits
        
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
        
    # Normalize whitespace
    text = re.sub(r'\n', ' ', text)            # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text).strip()   # Normalize spacing
        
    # Lemmatize and remove stopwords
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        
    return text

"""
import pandas as pd

@ensure_annotations
def read_csv(path_to_csv: Path) -> pd.DataFrame:
    # Reads a CSV file and returns a DataFrame
    try:
        df = pd.read_csv(path_to_csv)
        logger.info(f"CSV file loaded from: {path_to_csv}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise e

"""




