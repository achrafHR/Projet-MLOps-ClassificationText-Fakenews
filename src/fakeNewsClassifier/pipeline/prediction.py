import numpy as np
from fakeNewsClassifier.utils.common import load_bin, clean_text
import os
from pathlib import Path
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        # Initialisation sans besoin de texte
        # Les modèles seront chargés lors de la prédiction
        pass
        
    def predict(self, text):
     
        model = load_bin(Path("model/trained_model.pkl"))
        tfidf = load_bin(Path("vectorizer/tfidf_vectoriser.pkl"))
        
        testing_news = {"text":[text]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(clean_text)
        
        new_x_test = new_def_test["text"]
        new_xv_test = tfidf.transform(new_x_test)
        pred_LR = model.predict(new_xv_test)
        
        print(pred_LR)
        
        if pred_LR[0] == 1:
            prediction = 'Fake News'
            return [{ "text" : prediction}]
        else:
            prediction = 'Not A Fake News'
            return [{ "text" : prediction}]