from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import subprocess
from fakeNewsClassifier.pipeline.prediction import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Définition du modèle de données pour la requête
class TextInput(BaseModel):
    text: str

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Fake News Classifier",
    description="API pour la détection de fausses nouvelles",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des templates
templates = Jinja2Templates(directory="templates")

# Configuration des fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialisation du pipeline de prédiction
prediction_pipeline = PredictionPipeline()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Afficher la page d'accueil"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train")
async def train_model():
    """Lancer l'entraînement du modèle"""
    try:
        subprocess.run(["python", "main.py"], check=True)
        return {"message": "Training done successfully!"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=list)
async def predict(input_data: TextInput):
    """
    Prédire si un texte contient des fake news
    """
    try:
        # Prédiction basée sur le texte
        result = prediction_pipeline.predict(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)