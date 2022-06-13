import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torchsummary import summary
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=6, stride=2),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=8, stride=2),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=8, stride=2),
            nn.PReLU()
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(100, 50),
            nn.PReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


discriminator = Discriminator()
discriminator.to(device)
summary(discriminator, (1, 240))
