from joblib import load
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from librosa.core.spectrum import  util
from torch.utils.data import TensorDataset, DataLoader
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
        #print(x.shape)
        x = x.view(-1, 20 * 12 * 9)  # !!!
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, dataloader):
    model.eval()
    use_gpu = False
    output = model(dataloader)
    _, predicted = torch.max(output, 1)
    # for data in dataloader:
    #     # получаем картинки и метки
    #     inputs = data
    #     print("******inputs******")
    #     print(len(inputs))
    #     # переносим на gpu, если возможно
    #     if use_gpu:
    #         inputs = inputs.cuda()

    #     # forard pass
    #     output = model(inputs)
    #     print("******output******")
    #     print(output)
    #     _, predicted = torch.max(output, 1)
    return predicted

def data_proccesing(model, data, n_window, n_hop):
    batch_size = 2
    sscaler = load('std_scaler.bin')
    mfccs = sscaler.transform(data)
    mfccs_frame = util.frame(mfccs, frame_length=n_window, hop_length=n_hop, axis =0)
    #print(torch.FloatTensor(mfccs_frame)[:,None,:,:].shape)
    val_dataset = TensorDataset(torch.FloatTensor(mfccs_frame)[:,None,:,:])
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #print(valid_dataloader)
    result = evaluate(model, torch.FloatTensor(mfccs_frame)[:,None,:,:])
    return result

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