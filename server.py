import uvicorn
from fastapi import FastAPI
from modelim import model_load, data_parsing
import numpy as np
import json

app = FastAPI()
file = open("./config.json")
config = json.load(file)
n_mfcc = config["n_mfcc"]
n_window = config["n_window"]
n_hop = config["n_hop"]
model = model_load(config["model_name"])

@app.post('/predict')
def predict(data: list):
    """
    :data: input data from the post request
    :return: predicted type
    """
    features, meta_data = data_parsing( data, n_mfcc, n_window, n_hop )
    prediction = model.predict(features).tolist()
    meta_data["prediction"] =  str(prediction)
    return meta_data

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host=config["host"], port=config["port"])
