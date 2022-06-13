from EEG_VAE2 import vae
import torch
import numpy as np
import time
import random
import argparse
import datetime
import winsound
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed_n = np.random.randint(500)

flag = 1
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)

gen_time = 0
if flag == 1:
    filename = "targetoutA_8channels.txt"
else:
    filename = "targetoutB_8channels.txt"
data = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data = np.reshape(data, (170, 8, 240), 'F')
# print("data's shape : ", data.shape)
data = np.expand_dims(data, axis=1)
# print("data's shape : ", data.shape)
# print(type(data))

# generative model
print("*********Training Generative Model*********")
start = time.time()

print("VAE")
gen_data = vae(data, seed_n, flag)  # VAE
gen_time = gen_time + (time.time() - start)

# save generated data

fake_data0 = gen_data
print("data's shape : ", fake_data0.shape)

time_str = datetime.datetime.now().strftime('%m-%d-%H-%M')
if flag == 1:
    str1 = "VAE_fakeA_"
    str2 = ".npy"
    filename1 = str1 + time_str + str2
else:
    str1 = "VAE_fakeB_"
    str2 = ".npy"
    filename1 = str1 + time_str + str2
np.save(filename1, fake_data0)

print("time for generative model: %f" % gen_time)
print(seed_n)
winsound.Beep(600, 3000)
