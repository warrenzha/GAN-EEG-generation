import pytorch_fid
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy


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

filename1 = "VAE_fakeB_04-30-23-25.npy"
data_fake = np.load(filename1)
data_fake = data_fake[:170]
data_fake = np.squeeze(data_fake, 1)
data_real = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_real = np.reshape(data_real, (170, 8, 240), 'F')
data_noise = np.random.randn(170, 8, 240)


# def preprocess(data, meanc, stdc):
#     for i in range(data.shape[0]):
#         data_now = data[i, :, :]
#         data_now = np.squeeze(data_now, 0)
#         data_now = normalization(data_now)
#         data[i, :, :] = data_now
#
#     data = np.expand_dims(data, 1).repeat(299, axis=2).repeat(3, axis=1)
#
#     for i in range(3):
#         data_now = data[:, i, :, :]
#         data_now = data_now - meanc[i] / stdc[i]
#         data[:, i, :, :] = data_now
#
#     return data

def preprocessall(data, meanc, stdc):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_now = data[i, j, :]
            # print(data_now.shape)
            # data_now = np.squeeze(data_now, 0)
            data_now = normalization(data_now)
            data[i, j, :] = data_now

    data = np.expand_dims(data, 1).repeat(37, axis=2).repeat(3, axis=1)

    for i in range(3):
        data_now = data[:, i, :, :]
        data_now = data_now - meanc[i] / stdc[i]
        data[:, i, :, :] = data_now

    return data


def preprocess(data, j, meanc, stdc):
    for i in range(data.shape[0]):
        data_now = data[i, j, :]
        # print(data_now.shape)
        # data_now = np.squeeze(data_now, 0)
        data_now = normalization(data_now)
        data[i, 1, :] = data_now

    data = np.expand_dims(data, 1).repeat(299, axis=2).repeat(3, axis=1)

    for i in range(3):
        data_now = data[:, i, :, :]
        data_now = data_now - meanc[i] / stdc[i]
        data[:, i, :, :] = data_now

    return data


def get_inception_score(imgs, cuda=True, batch_size=5, resize=True, splits=1):
    """
        Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


data_fake = preprocessall(data_fake, mean_inception, std_inception)
data_real = preprocessall(data_real, mean_inception, std_inception)
data_noise = preprocessall(data_noise, mean_inception, std_inception)
mean1, std1 = get_inception_score(data_real)
mean2, std2 = get_inception_score(data_fake)
mean3, std3 = get_inception_score(data_noise)
print('IS of real is %.4f' % mean1)
print('The std is %.4f' % std1)
print('IS of fake is %.4f' % mean2)
print('The std is %.4f' % std2)
print('IS of noise is %.4f' % mean3)
print('The std is %.4f' % std3)

# meanall = np.empty([8, 3])
# for i in range(8):
#     data_fakeone = preprocess(data_fake, i, mean_inception, std_inception)
#     data_realone = preprocess(data_real, i, mean_inception, std_inception)
#     data_noiseone = preprocess(data_noise, i, mean_inception, std_inception)
#     meanall[i, 0], std = get_inception_score(data_realone)
#     meanall[i, 1], std = get_inception_score(data_fakeone)
#     meanall[i, 2], std = get_inception_score(data_noiseone)
#
# print(meanall)
