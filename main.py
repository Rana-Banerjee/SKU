from fastapi import FastAPI
from predictor import Predictor

app=FastAPI()
predictor = Predictor('./models/best.pth')

@app.get('/')
def home():
    return 'Usage: /predict?url=url_to_image'

@app.get('predict')
def predict(url:str):
    return {'prediction': predictor.predict(url)}
    