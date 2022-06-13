import matplotlib.pyplot as plt
import numpy as np

flag = 1
filename1 = "GAN_fakeA_04-30-23-39.npy"
data_fake = np.load(filename1)

flag1 = 2
if flag1 == 2:
    data_fake = np.squeeze(data_fake, 1)

if flag == 1:
    filename = "targetoutA_8channels.txt"
else:
    filename = "targetoutB_8channels.txt"
data_real = np.loadtxt(filename, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
data_real = np.reshape(data_real, (170, 8, 240), 'F')
filenameD = "DlossA_04-30-23-39.npy"
filenameG = "GlossA_04-30-23-39.npy"
Dloss = np.load(filenameD)
Gloss = np.load(filenameG)
# print(Dloss.shape)

plt.close('all')
fig = plt.figure(figsize=(10, 10))
fig, subs = plt.subplots(4, 2, figsize=(10, 16))
fig.suptitle('WGAN-GP generated data and real data——Subject A(2/2)', fontsize=20)

for j in range(4):
    aa = np.random.permutation(800)
    for i in range(170):
        x = np.arange(240) / 240
        y = data_fake[aa[i], j+4, :]
        subs[j][0].plot(x, np.squeeze(y))

    # plt.show()

    for i in range(170):
        x = np.arange(240) / 240
        y = data_real[i, j+4, :]
        subs[j][1].plot(x, np.squeeze(y))

# subs[0, 0].set_title('generated data channel 4', fontsize=15)
# subs[0, 0].set_xlabel('Time(s)', fontsize=12)
# subs[0, 1].set_title('real data channel 4', fontsize=15)
# subs[0, 1].set_xlabel('Time(s)', fontsize=12)
# subs[1, 0].set_title('generated data channel 5', fontsize=15)
# subs[1, 0].set_xlabel('Time(s)', fontsize=12)
# subs[1, 1].set_title('real data channel 5', fontsize=15)
# subs[1, 1].set_xlabel('Time(s)', fontsize=12)
# subs[2, 0].set_title('generated data channel 6', fontsize=15)
# subs[2, 0].set_xlabel('Time(s)', fontsize=12)
# subs[2, 1].set_title('real data channel 6', fontsize=15)
# subs[2, 1].set_xlabel('Time(s)', fontsize=12)
# subs[3, 0].set_title('generated data channel 10', fontsize=15)
# subs[3, 0].set_xlabel('Time(s)', fontsize=12)
# subs[3, 1].set_title('real data channel 10', fontsize=15)
# subs[3, 1].set_xlabel('Time(s)', fontsize=12)
subs[0, 0].set_title('generated data channel 11', fontsize=15)
subs[0, 0].set_xlabel('Time(s)', fontsize=12)
subs[0, 1].set_title('real data channel 11', fontsize=15)
subs[0, 1].set_xlabel('Time(s)', fontsize=12)
subs[1, 0].set_title('generated data channel 12', fontsize=15)
subs[1, 0].set_xlabel('Time(s)', fontsize=12)
subs[1, 1].set_title('real data channel 12', fontsize=15)
subs[1, 1].set_xlabel('Time(s)', fontsize=12)
subs[2, 0].set_title('generated data channel 17', fontsize=15)
subs[2, 0].set_xlabel('Time(s)', fontsize=12)
subs[2, 1].set_title('real data channel 17', fontsize=15)
subs[2, 1].set_xlabel('Time(s)', fontsize=12)
subs[3, 0].set_title('generated data channel 18', fontsize=15)
subs[3, 0].set_xlabel('Time(s)', fontsize=12)
subs[3, 1].set_title('real data channel 18', fontsize=15)
subs[3, 1].set_xlabel('Time(s)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
#
fig2, subs2 = plt.subplots(4, 2, figsize=(10, 16))
fig2.suptitle('WGAN-GP the mean of generated data and real data——Subject A', fontsize=20)
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


plt.show()
# # #
fig3, subs3 = plt.subplots(2, 1, figsize=(6, 8))
xx = np.arange(2000)
subs3[0].plot(xx, Dloss)
subs3[0].set_title('Dloss of train', fontsize=12)
subs3[0].set_xlabel('epoch')
# plt.show()
subs3[1].plot(xx, Gloss)
subs3[1].set_title('Gloss of train', fontsize=12)
subs3[1].set_xlabel('epoch')
fig3.suptitle('WGAN-GP loss of train——Subject A', fontsize=18)
fig3.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
