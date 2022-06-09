import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

flag = 1
filename1 = "VAE_fakeA_05-10-13-13.npy"

data_fake = np.load(filename1)
flag1 = 1
if flag1 == 2:
    data_fake = np.squeeze(data_fake, 1)
print(data_fake.shape)
# data_fake = np.squeeze(data_fake)
if flag == 1:
    filename = "targetoutA_8channels.txt"
else:
    filename = "targetoutB_8channels.txt"
data_real = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_real = np.reshape(data_real, (170, 8, 240), 'F')
filenameB = "bceB_05-10-13-13.npy"
filenameK = "kldB_05-10-13-13.npy"
filenameL = "lossB_05-10-13-13.npy"
bce = np.load(filenameB)
kld = np.load(filenameK)
loss = np.load(filenameL)

fig, subs = plt.subplots(4, 2, figsize=(10, 16))
fig.suptitle('VAE generated data and real data——Subject B(1/2)', fontsize=20)

for j in range(4):
    aa = np.random.permutation(170)
    for i in range(170):
        x = np.arange(240) / 240
        y = data_fake[aa[i], j, :]
        subs[j][0].plot(x, np.squeeze(y))

    # plt.show()

    for i in range(170):
        x = np.arange(240) / 240
        y = data_real[i, j, :]
        subs[j][1].plot(x, np.squeeze(y))

subs[0, 0].set_title('generated data channel 4', fontsize=15)
subs[0, 0].set_xlabel('Time(s)', fontsize=12)
subs[0, 1].set_title('real data channel 4', fontsize=15)
subs[0, 1].set_xlabel('Time(s)', fontsize=12)
subs[1, 0].set_title('generated data channel 5', fontsize=15)
subs[1, 0].set_xlabel('Time(s)', fontsize=12)
subs[1, 1].set_title('real data channel 5', fontsize=15)
subs[1, 1].set_xlabel('Time(s)', fontsize=12)
subs[2, 0].set_title('generated data channel 6', fontsize=15)
subs[2, 0].set_xlabel('Time(s)', fontsize=12)
subs[2, 1].set_title('real data channel 6', fontsize=15)
subs[2, 1].set_xlabel('Time(s)', fontsize=12)
subs[3, 0].set_title('generated data channel 10', fontsize=15)
subs[3, 0].set_xlabel('Time(s)', fontsize=12)
subs[3, 1].set_title('real data channel 10', fontsize=15)
subs[3, 1].set_xlabel('Time(s)', fontsize=12)
# subs[0, 0].set_title('generated data channel 11', fontsize=15)
# subs[0, 0].set_xlabel('Time(s)', fontsize=12)
# subs[0, 1].set_title('real data channel 11', fontsize=15)
# subs[0, 1].set_xlabel('Time(s)', fontsize=12)
# subs[1, 0].set_title('generated data channel 12', fontsize=15)
# subs[1, 0].set_xlabel('Time(s)', fontsize=12)
# subs[1, 1].set_title('real data channel 12', fontsize=15)
# subs[1, 1].set_xlabel('Time(s)', fontsize=12)
# subs[2, 0].set_title('generated data channel 17', fontsize=15)
# subs[2, 0].set_xlabel('Time(s)', fontsize=12)
# subs[2, 1].set_title('real data channel 17', fontsize=15)
# subs[2, 1].set_xlabel('Time(s)', fontsize=12)
# subs[3, 0].set_title('generated data channel 18', fontsize=15)
# subs[3, 0].set_xlabel('Time(s)', fontsize=12)
# subs[3, 1].set_title('real data channel 18', fontsize=15)
# subs[3, 1].set_xlabel('Time(s)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

fig2, subs2 = plt.subplots(4, 2, figsize=(10, 16))
fig2.suptitle('VAE the mean of generated data and real data——Subject B', fontsize=20)
# fig2.set_tight_layout(True)
for j in range(8):
    realmean = np.mean(np.squeeze(data_real[:, j, :]), 0)
    fakemean = np.mean(np.squeeze(data_fake[:, j, :]), 0)
    x = np.arange(240) / 240
    s1, = subs2[j // 2, j % 2].plot(x, realmean)
    s2, = subs2[j // 2, j % 2].plot(x, fakemean)
    # plt.legend([s1])
    # plt.legend([s2], ['fakemean'])
    # s1.legend('realmean')

subs2[0, 0].set_title('channel 4', fontsize=15)
subs2[0, 0].legend(['realmean', 'fakemean'])
subs2[0, 0].set_xlabel('Time(s)', fontsize=12)
subs2[0, 1].set_title('channel 5', fontsize=15)
subs2[0, 1].legend(['realmean', 'fakemean'])
subs2[0, 1].set_xlabel('Time(s)', fontsize=12)
subs2[1, 0].set_title('channel 6', fontsize=15)
subs2[1, 0].legend(['realmean', 'fakemean'])
subs2[1, 0].set_xlabel('Time(s)', fontsize=12)
subs2[1, 1].set_title('channel 10', fontsize=15)
subs2[1, 1].legend(['realmean', 'fakemean'])
subs2[1, 1].set_xlabel('Time(s)', fontsize=12)
subs2[2, 0].set_title('channel 11', fontsize=15)
subs2[2, 0].legend(['realmean', 'fakemean'])
subs2[2, 0].set_xlabel('Time(s)', fontsize=12)
subs2[2, 1].set_title('channel 12', fontsize=15)
subs2[2, 1].legend(['realmean', 'fakemean'])
subs2[2, 1].set_xlabel('Time(s)', fontsize=12)
subs2[3, 0].set_title('channel 17', fontsize=15)
subs2[3, 0].legend(['realmean', 'fakemean'])
subs2[3, 0].set_xlabel('Time(s)', fontsize=12)
subs2[3, 1].set_title('channel 18', fontsize=15)
subs2[3, 1].legend(['realmean', 'fakemean'])
subs2[3, 1].set_xlabel('Time(s)', fontsize=12)
fig2.tight_layout(rect=(0, 0, 1, 0.95))
# fig2.update()
#
plt.show()
# plt.close('all')
# for j in range(8):
#     for i in range(200):
#         x = np.arange(240) / 240
#         y = data_fake[i, j, :]
#         plt.plot(x, np.squeeze(y))
#
#     plt.show()
#
#     for i in range(170):
#         x = np.arange(240) / 240
#         y = data_real[i, j, :]
#         plt.plot(x, np.squeeze(y))
#
#     plt.show()
#
#     realmean = np.mean(np.squeeze(data_real[:, j, :]), 0)
#     fakemean = np.mean(np.squeeze(data_fake[:, j, :]), 0)
#     x = np.arange(240) / 240
#     plt.plot(x, realmean)
#     plt.plot(x, fakemean)
#     plt.show()


fig = plt.figure(figsize=(10, 8))
fig.suptitle('loss of train——Subject B', fontsize=20)
gs = gridspec.GridSpec(2, 4)
# fig, subs = plt.subplots(2, 2,)
xx = np.arange(2500)
sub1 = plt.subplot(gs[0, :2])
sub1.plot(xx, bce)
sub1.set_title('MSE loss', fontsize=15)
sub1.set_xlabel('epoch', fontsize=12)
# plt.show()
sub2 = plt.subplot(gs[0, 2:])
sub2.plot(xx, kld[:2500])
sub2.set_title('KL divergence', fontsize=15)
sub2.set_xlabel('epoch', fontsize=12)
# plt.show()
sub3 = plt.subplot(gs[1, 1:3])
sub3.plot(xx, loss)
sub3.set_title('Total loss', fontsize=15)
sub3.set_xlabel('epoch', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
