import joblib
import json
import numpy as np

def data_proccesing(data):
    finished_data = data
    return finished_data

def data_parsing(data, n_mfcc, n_window, n_hop):
    cep = np.zeros([0])
    meta_data = {"DataTimeCreate":data[0]["fields"]["DataTimeCreate"],
                    "user":data[0]["fields"]["user"],
                    "id_record":data[0]["fields"]["id_record"]}
    for d in data:
        for i in range(n_mfcc):
           cep= np.append(cep, d["fields"]['Cepstral_float_'+str(i+1)])
    cep = np.array([cep[i : i + n_mfcc*n_window] for i in range(0, len(cep)-n_window*n_mfcc+1, n_mfcc*n_hop)])
    return cep, meta_data

def model_load(model_name:str):
    model = joblib.load(model_name)
    return model