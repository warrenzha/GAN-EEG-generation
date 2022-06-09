import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from scipy import linalg
import winsound
from pytorch_fid.inception import InceptionV3


# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]

flag = 2
if flag == 1:
    filename = "targetoutA_8channels.txt"
else:
    filename = "targetoutB_8channels.txt"

filename1 = "GAN_fakeB_05-10-10-38.npy"
aa = np.random.permutation(800)
data_fake = np.load(filename1)
data_fake = data_fake[aa[:170]]
data_fake = np.squeeze(data_fake, 1)
data_real = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_real = np.reshape(data_real, (170, 8, 240), 'F')
# data_real = np.expand_dims(data_real, 1)
# data_real = data_real[:2000]
data_noise = np.random.randn(170, 8, 240)


def preprocessall(data, meanc, stdc):
    for ii in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_now = data[ii, j, :]
            # print(data_now.shape)
            # data_now = np.squeeze(data_now, 0)
            data_now = normalization(data_now)
            data[ii, j, :] = data_now

    data = np.expand_dims(data, 1).repeat(37, axis=2).repeat(3, axis=1)

    for ii in range(3):
        data_now = data[:, ii, :, :]
        data_now = data_now - meanc[ii] / stdc[ii]
        data[:, ii, :, :] = data_now

    return data


def preprocess(data, j, meanc, stdc):
    for ii in range(data.shape[0]):
        data_now = data[ii, j, :]
        # print(data_now.shape)
        # data_now = np.squeeze(data_now, 0)
        data_now = normalization(data_now)
        data[ii, 1, :] = data_now

    data = np.expand_dims(data, 1).repeat(299, axis=2).repeat(3, axis=1)

    for ii in range(3):
        data_now = data[:, ii, :, :]
        data_now = data_now - meanc[ii] / stdc[ii]
        data[:, ii, :, :] = data_now

    return data


def compute_FID(img1, img2, batch_size=5):
    device = torch.device("cuda:0")  # you can change the index of cuda

    N1 = len(img1)
    N2 = len(img2)
    n_act = 2048  # the number of final layer's dimension
    dtype = torch.cuda.FloatTensor
    # Set up dataloaders
    dataloader1 = torch.utils.data.DataLoader(img1, batch_size=batch_size)
    dataloader2 = torch.utils.data.DataLoader(img2, batch_size=batch_size)

    # Load inception model
    # inception_model = inception_v3(pretrained=True, transform_input=False).to(device)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[n_act]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    # get the activations
    def get_activations(x):
        x = inception_model(x)[0]
        return x.cpu().data.numpy().reshape(batch_size, -1)

    act1 = np.zeros((N1, n_act))
    act2 = np.zeros((N2, n_act))

    data = [dataloader1, dataloader2]
    act = [act1, act2]
    for n, loader in enumerate(data):
        for i, batch in enumerate(loader, 0):
            batch = batch.to(device).type(dtype)
            batch_size_i = batch.size()[0]
            activation = get_activations(batch)

            act[n][i * batch_size:i * batch_size + batch_size_i] = activation

    # compute the activation's statistics: mean and std
    def compute_act_mean_std(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)

    # compute FID
    def _compute_FID(mu1, mu2, sigma1, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return FID

    FID = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return FID


data_fake = preprocessall(data_fake, mean_inception, std_inception)
data_real = preprocessall(data_real, mean_inception, std_inception)
data_noise = preprocessall(data_noise, mean_inception, std_inception)
FID1 = compute_FID(data_fake, data_real)
FID2 = compute_FID(data_real, data_real)
FID3 = compute_FID(data_noise, data_real)
print('FID score is %.4f' % FID1)
print('FID score is %.4f' % FID2)
print('FID score is %.4f' % FID3)

# fidall = np.empty([8, 3])
# for i in range(8):
#     data_fakeone = preprocess(data_fake, i, mean_inception, std_inception)
#     data_realone = preprocess(data_real, i, mean_inception, std_inception)
#     data_noiseone = preprocess(data_noise, i, mean_inception, std_inception)
#     fidall[i, 0] = compute_FID(data_fakeone, data_realone)
#     fidall[i, 1] = compute_FID(data_realone, data_realone)
#     fidall[i, 2] = compute_FID(data_noiseone, data_realone)
#
# np.set_printoptions(4, suppress=True)
# print(fidall)

winsound.Beep(600, 3000)
