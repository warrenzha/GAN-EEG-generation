import matplotlib.pyplot as plt
import numpy as np

flag = 1
filename1 = "GAN_fakeA_05-10-11-26.npy"
data_fake = np.load(filename1)
data_fake = np.squeeze(data_fake, 1)
if flag == 1:
    filename = "targetoutallA_8channels.txt"
else:
    filename = "targetoutallB_8channels.txt"
data_real = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_real = np.reshape(data_real, (2550, 8, 240), 'F')

filenameD = "DlossA_05-10-11-26.npy"
filenameG = "GlossA_05-10-11-26.npy"
Dloss = np.load(filenameD)
Gloss = np.load(filenameG)
# print(Dloss.shape)

plt.close('all')
for j in range(8):
    for i in range(200):
        x = np.arange(240) / 240
        y = data_fake[i, j, :]
        plt.plot(x, np.squeeze(y))

    plt.show()

    for i in range(200):
        x = np.arange(240) / 240
        y = data_real[i, j, :]
        plt.plot(x, np.squeeze(y))

    plt.show()

    realmean = np.mean(np.squeeze(data_real[:, j, :]), 0)
    fakemean = np.mean(np.squeeze(data_fake[:, j, :]), 0)
    x = np.arange(240) / 240
    plt.plot(x, realmean)
    plt.plot(x, fakemean)
    plt.show()

xx = np.arange(3000)
plt.plot(xx, Dloss)
plt.show()
plt.plot(xx, Gloss)
plt.show()
