from joblib import load
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from librosa.core.spectrum import  util
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.fc1 = nn.Linear(20 * 12 * 9, 120)  # !!!
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 12 * 9)  # !!!
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelDropout(nn.Module):
    def __init__(self):
        super(ModelDropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.fc1 = nn.Linear(20 * 12 * 9, 120)  # !!!
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 20 * 12 * 9)  # !!!
        x = self.dropout1(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, dataloader,device=torch.device("cpu")):
    model.eval()
    prediction = np.array([])
    with torch.no_grad():
        for xb in dataloader:
            xb = xb[0].to(device)
            probs = model(xb)
            _, preds = torch.max(probs, axis=-1)
            prediction = np.append(prediction, preds.numpy())
    return prediction



def data_parsing(data, n_mfcc):
    ceps = np.zeros([0,n_mfcc])
    meta_data = {"DataTimeCreate":data[0]["fields"]["DataTimeCreate"],
                    "user":data[0]["fields"]["user"],
                    "id_record":data[0]["fields"]["id_record"]}
    for d in data:
        cep = np.zeros([0])
        for i in range(n_mfcc):
           cep= np.append(cep, d["fields"]['Cepstral_float_'+str(i+1)])
        ceps = np.append(ceps, [cep], axis=0)
    return ceps, meta_data

def model_load(model_name:str):
    model = LeNet()
    model.load_state_dict(torch.load(model_name))
    return model

def frame_analyse(marks):
    hbflag = False
    hb_start = []
    hb_end = []
    for k,mark in enumerate(marks):
        if mark:
            if hbflag==False:
                hb_start.append(k)
            hbflag = True
        elif hbflag:
            hb_end.append(k)
            hbflag = False
    if hbflag:
            hb_end.append(k)
            hbflag = False
    return hb_start,hb_end


file = open("./config.json")
config = json.load(file)
batch_size = 512
mfcc_hop = 512
mfcc_frame = 2048
n_mfcc = 15
sr = 16000
n_win = 18
n_hop = 1

def frame_to_sec(x):
    return ((x-1)*mfcc_hop + mfcc_frame)/sr

def mean_hr(prediction, mindur=1):
    starts, ends = frame_analyse(prediction)
    frame_pred = pd.DataFrame({"starts":starts,"ends":ends})
    frame_pred["duration"] = frame_pred["ends"]-frame_pred["starts"]
    new_df = frame_pred[frame_pred["duration"]>mindur]
    try:
        secst = new_df["starts"].apply(frame_to_sec)
        print(secst.iloc[-1],secst.iloc[0],secst.shape[0])
        return 60/((secst.iloc[-1]-secst.iloc[0])/secst.shape[0])
    except:
        return 0

def data_proccesing(model, data, n_window, n_hop):
    batch_size = 2
    sscaler = load('./std_scaler_1.bin')
    mfccs = sscaler.transform(data)
    mfccs_frame = util.frame(mfccs, frame_length=n_window, hop_length=n_hop, axis =0)
    val_dataset = TensorDataset(torch.FloatTensor(np.copy(mfccs_frame))[:,None,:,:])
    valid_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    prediction = evaluate(model, valid_dataloader)
    return mean_hr(prediction)


# if __name__=="__main__":
#     import os
#     import datagenerator
    
#     batch_size = 512
#     mfcc_hop = 512
#     mfcc_frame = 2048
#     n_mfcc = 15
#     sr = 16000
#     n_win = 18
#     n_hop = 1
#     dg = datagenerator.DataGenerator(n_hop,n_win,sr)
#     files = np.array(os.listdir('new_bront\data_denis'))
#     test = np.zeros([0,n_win,n_mfcc])
#     target_test = np.zeros([0])
#     sscaler = load('new_bront\src\std_scaler.bin')
#     for name in files[-1:]:
#         print(name)
#         mfccs = datagenerator._from_denis('new_bront\data_denis/'+name)
#         target_test = np.append(target_test, dg._get_target('new_bront\data\marks/'+name[:-4]+'__mark.txt', mfccs.shape[0]*mfcc_hop+(n_win+2)*mfcc_hop))
#     new_target_test = np.array([[x==0, x==1] for x in target_test])
#     file = open("new_bront\src\config.json")
#     config = json.load(file)
#     n_mfcc = config["n_mfcc"]
#     n_window = config["n_window"]
#     n_hop = config["n_hop"]
#     model = model_load(config["model_name"])
#     s,p = data_proccesing(model, mfccs,n_window,n_hop)
#     print(s.sum(), p)