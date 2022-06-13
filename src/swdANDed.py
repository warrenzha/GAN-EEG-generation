import numpy as np
import torch
from swdd import sliced_wasserstein_distance
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


flag = 1
filename1 = "GAN_fakeB_05-10-10-38.npy"
data_fake = np.load(filename1)
aa = np.random.permutation(800)
data_fake = data_fake[aa[:170]]
data_fake = np.squeeze(data_fake, 1)
if flag == 1:
    filename = "targetoutA_8channels.txt"
else:
    filename = "targetoutB_8channels.txt"
data_real = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_real = np.reshape(data_real, (170, 8, 240), 'F')
data_noise = np.random.randn(170, 8, 240)


def preprocess(data, flag):
    for ii in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_now = data[ii, j, :]
            # print(data_now.shape)
            # data_now = np.squeeze(data_now, 0)
            data_now = normalization(data_now)
            data[ii, j, :] = data_now

    if flag == 10:
        data = np.reshape(data, [data.shape[0] * data.shape[1], 240])
    else:
        data = np.squeeze(data[:, flag, :])

    return data


data_fake0 = preprocess(data_fake, 10)
data_real0 = preprocess(data_real, 10)
data_noise0 = preprocess(data_noise, 10)

data_fake1 = torch.from_numpy(data_fake0).float()
data_real1 = torch.from_numpy(data_real0).float()
data_noise1 = torch.from_numpy(data_noise0).float()

# print(data_fake1.shape)
out1 = sliced_wasserstein_distance(data_fake1, data_real1, num_projections=240)
out2 = sliced_wasserstein_distance(data_noise1, data_real1, num_projections=240)
# torch.manual_seed(231)  # fix seed
# out = swd(data_fake, data_real, device="cuda")  # Fast estimation if device="cuda"
print(out1)  # tensor(53.6950)
print(out2)

swdall = np.empty([8, 2])
for i in range(8):
    data_fake11 = torch.from_numpy(preprocess(data_fake, i)).float()
    data_real11 = torch.from_numpy(preprocess(data_real, i)).float()
    data_noise11 = torch.from_numpy(preprocess(data_noise, i)).float()
    swdall[i, 0] = sliced_wasserstein_distance(data_fake11, data_real11, num_projections=240)
    swdall[i, 1] = sliced_wasserstein_distance(data_noise11, data_real11, num_projections=240)
print(swdall)

# data_fake2 = preprocess(data_fake, 10)
# data_real2 = preprocess(data_real, 10)
# data_noise2 = preprocess(data_noise, 10)
# # print(data_fake2.shape)
# ijdis = np.zeros([data_fake2.shape[0], data_real2.shape[0]])
# # for i in range(data_fake2.shape[0]):
# #     now_i = data_fake2[i, :]
# #     for j in range(data_real2.shape[0]):
# #         now_j = data_real2[j, :]
# #         ijdis[i, j] = pairwise_distances(now_i, now_j)
# ijdis1 = distance_matrix(data_real2, data_fake2, 1)
# ijdis2 = distance_matrix(data_real2, data_noise2, 1)
# # ijdis3 = distance_matrix(data_fake2, data_noise2)
# edmin1 = np.min(ijdis1) / 240
# edmin2 = np.min(ijdis2) / 240
# # edmin3 = np.min(ijdis3)
# # a = torch.squeeze(torch.pairwise_distance(data_fake2, data_real2),1)
# print(edmin1)
# print(edmin2)
# # print(edmin3)
#
# disall = np.empty([8, 2])
# for i in range(8):
#     data_fake22 = torch.from_numpy(preprocess(data_fake, i)).float()
#     data_real22 = torch.from_numpy(preprocess(data_real, i)).float()
#     data_noise22 = torch.from_numpy(preprocess(data_noise, i)).float()
#     disall[i, 0] = np.min(distance_matrix(data_real22, data_fake22, 1)) / 240
#     disall[i, 1] = np.min(distance_matrix(data_real22, data_noise22, 1)) / 240
# print(disall)
