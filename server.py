import uvicorn
from fastapi import FastAPI
from modelim import model_load, data_parsing, data_proccesing
import numpy as np
import json
import requests

app = FastAPI()
file = open("./config.json")
config = json.load(file)
n_mfcc = config["n_mfcc"]
n_window = config["n_window"]
n_hop = config["n_hop"]
model = model_load(config["model_name"])
URL_DATA_HANDLER = config["URL_DATA_HANDLER"]
headers =  config["headers"]
@app.post('/predict')
def predict(data: list):
    """
    :data: input data from the post request
    :return: predicted type
    """
    print("******start predicting******")
    features, meta_data = data_parsing( data, n_mfcc )
    print("******data******")
    print(features.shape)
    print("************")
    prediction = data_proccesing(model, features, n_window, n_hop)
    print("******done!******")
    meta_data["record"] = str(prediction)
    #requests.post(URL_DATA_HANDLER, json=meta_data, headers=headers)
    return meta_data

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host=config["host"], port=config["port"])

