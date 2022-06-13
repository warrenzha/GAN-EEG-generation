import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import datetime

flag = 1
fake_data0 = np.random.randn(6, 6)
time_str = datetime.datetime.now().strftime('%m-%d-%H-%M')
if flag == 1:
    str1 = "fakeA_"
    str2 = ".npy"
    filename1 = str1 + time_str + str2
else:
    str1 = "fakeB_"
    str2 = ".npy"
    filename1 = str1 + time_str + str2
np.save(filename1, fake_data0)
print(str)
