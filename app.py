
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from fakeNewsClassifier.pipeline.prediction import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Récupération du texte depuis la requête JSON
    text = request.json['text']
    
    # Prédiction basée sur le texte
    result = clApp.classifier.predict(text)
    
    # Retourne le résultat de la prédiction
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) # pour AWS