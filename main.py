#from flask import Flask
from fastapi import FastAPI
from predictor import Predictor

app=FastAPI()
#app=Flask(__name__)
predictor = Predictor('./models/best.pth')

@app.get('/')
def home():
    return 'Usage: /predict?url=url_to_image'

@app.get('/predict')
def predict(url:str):
    return {'prediction': predictor.predict(url)}
    #return "in predict"
    
#if __name__ == '__main__':
#    # This is used when running locally. Gunicorn is used to run the
#    # application on Google App Engine. See entrypoint in app.yaml.
#    app.run(host='127.0.0.1', port=8080, debug=True)
