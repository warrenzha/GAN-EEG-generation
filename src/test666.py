import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
from os.path import join
from modelCNN import ConvNet


def processdata(realdata, fakedata, nondata, realnum, fakenum):
    reala = realnum // 170
    realb = realnum % 170
    real1 = np.repeat(realdata, reala, 0)
    real2 = realdata[:realb]
    real3 = np.concatenate((real1, real2), 0)

    fake1 = fakedata[:fakenum]
    datatrain = np.concatenate((real3, fake1, nondata), 0)
    datatrain = datatrain[:, :, :144]
    # datatrain = np.expand_dims(datatrain, 1)
    targetlabel = np.ones((realnum + fakenum, 1))
    nontargetlabel = np.zeros((nondata.shape[0], 1))
    datalabel = np.concatenate((targetlabel, nontargetlabel), 0)

    assert len(datatrain) == len(datalabel)
    samples = np.array(list(zip(datatrain, datalabel)))

    return samples


flag = 1
filename1 = "GAN_fakeA_05-05-19-56.npy"
data_fake = np.load(filename1)

if flag == 1:
    filename2 = "targetoutA_8channels.txt"
    filename3 = "NontargetoutA_8channels.txt"
else:
    filename2 = "targetoutB_8channels.txt"
    filename3 = "NontargetoutB_8channels.txt"
data_real = np.loadtxt(filename2, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_real = np.reshape(data_real, (170, 8, 240), 'F')
data_nonreal = np.loadtxt(filename3, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_nonreal = np.reshape(data_nonreal, (850, 8, 240), 'F')

data1 = processdata(data_real, data_fake, data_nonreal, 170, 0)
data2 = processdata(data_real, data_fake, data_nonreal, 170, 255)
data3 = processdata(data_real, data_fake, data_nonreal, 425, 0)
data4 = processdata(data_real, data_fake, data_nonreal, 170, 680)
data5 = processdata(data_real, data_fake, data_nonreal, 850, 0)

print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)

data = (data1, data2, data3, data4, data5)
datanow = data[0]

aaa=666