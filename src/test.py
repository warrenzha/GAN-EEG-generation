import numpy as np
import matplotlib.pyplot as plt

filename = "targetoutA_8channels.txt"
data0 = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data = np.reshape(data0, (170, 8, 240), 'F')

print(data0)
print(data)

for i in range(170):
    x = np.arange(240) / 240
    y = data[i, 1, :]
    plt.plot(x, y)

plt.show()

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.biasout = nn.Parameter(torch.tensor([0.5], dtype=torch.float32, device='cuda',requires_grad=True))
        self.nz = opt.nz
        self.layer1 = nn.Sequential(
            nn.Linear(self.nz, 800),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=8, stride=2),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=8, stride=2),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=8, kernel_size=6, stride=2, bias=False)
            # nn.PReLU()
        )

    def forward(self, z):
        out = self.layer1(z)
        out = out.view(out.size(0), 32, 25)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out + self.biasout
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=8, stride=2),
            # nn.BatchNorm1d(num_features=8),
            nn.PReLU(),
            nn.MaxPool1d(2))
        self.dense_layers = nn.Sequential(
            nn.Linear(1856, 900),
            nn.PReLU(),
            nn.Linear(900, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out