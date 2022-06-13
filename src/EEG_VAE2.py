import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchsummary import summary
import datetime

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 2500
batch_size = 10


def batch_normalization(data, batchsize):
    data = np.squeeze(data, 1)
    for i in range(batchsize):
        _range = np.max(data[i, :]) - np.min(data[i, :])
        data[i, :] = (data[i, :] - np.min(data[i, :])) / _range
    data = torch.from_numpy(np.expand_dims(data, 1))
    return data


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=4):
        UF = input.view(input.size(0), size, 1, 25)
        return UF


class EEG_CNN_VAE(nn.Module):
    def __init__(self):
        super(EEG_CNN_VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 6), stride=2),
            # nn.BatchNorm1d(num_features=4),
            nn.PReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(2, 8), stride=2),
            nn.PReLU(),
            nn.MaxPool2d((1, 2)),
            Flatten(),
            nn.Linear(224, 100),
            nn.PReLU(),
        )

        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 100)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=(2, 8), stride=2),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=(2, 8), stride=2),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(2, 6), stride=2),
            # nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


aa = EEG_CNN_VAE().to(device)
summary(aa, (1, 8, 240))


def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x)
    mse = nn.MSELoss()
    BCE = mse(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def vae(datatrain, nseed, flag):
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)

    dataloader = torch.utils.data.DataLoader(dataset=datatrain, batch_size=batch_size, shuffle=True)

    model = EEG_CNN_VAE().to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def writeloss(loss, bce, kld):
        time_str = datetime.datetime.now().strftime('%m-%d-%H-%M')
        if flag == 1:
            str1 = "lossA_"
            str2 = "bceA_"
            str3 = "kldA_"
            str4 = ".npy"
            filename1 = str1 + time_str + str4
            filename2 = str2 + time_str + str4
            filename3 = str3 + time_str + str4
        else:
            str1 = "lossB_"
            str2 = "bceB_"
            str3 = "kldB_"
            str4 = ".npy"
            filename1 = str1 + time_str + str4
            filename2 = str2 + time_str + str4
            filename3 = str3 + time_str + str4
        np.save(filename1, loss)
        np.save(filename2, bce)
        np.save(filename3, kld)

    Loss = []
    Bce = []
    Kld = []
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            data = data.to(device)
            data = data.type(Tensor)

            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            # print(data)
            # print(recon_data)
            # data_nor = batch_normalization(data.cpu().numpy(), batch_size)
            # recondata_nor = batch_normalization(recon_data.cpu().detach().numpy(), batch_size)
            # recondata_nor.requires_grad_(True)
            loss, bce, kld = loss_fn(recon_data, data, mu, logvar)
            bce.requires_grad_(True)
            loss.backward()
            optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.6f} {:.6f} {:.6f}".format(epoch + 1, num_epochs, loss.item(), bce.item(),
                                                                    kld.item())
        print(to_print)
        Loss = np.append(Loss, loss.item())
        Bce = np.append(Bce, bce.item())
        Kld = np.append(Kld, kld.item())
    writeloss(Loss, Bce, Kld)

    # Generating new data
    new_data = []
    num_data_to_generate = 800
    with torch.no_grad():
        model.eval()
        for epoch in range(num_data_to_generate):
            z = torch.randn(1, 50).to(device)
            recon_data = model.decode(z).cpu().numpy()

            new_data.append(recon_data)

        new_data = np.concatenate(new_data)
        new_data = np.asarray(new_data)
        return new_data
