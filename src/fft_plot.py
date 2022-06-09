import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

T = 1.0
# frequency class
nf = 0


def fft_data(input_data):
    ffts = []
    for block in range(input_data.shape[0]):
        for channel in range(input_data.shape[1]):
            # input_data[block, channel, :] -= input_data[block, channel, :].mean()
            # print data.shape
            ffts.append(np.abs(np.fft.rfft(input_data[block, channel, :])))
    mean_fft = np.stack(ffts).mean(axis=0)
    return mean_fft


def plot_fft(x_axis, data_fft, color, linestyle):
    plt.plot(x_axis, data_fft, color, linestyle=linestyle)

    plt.xlabel('Frequency', fontsize=17)
    plt.ylabel('Amplitude', fontsize=17)
    # plt.axis(xmin=5, xmax=70)
    # plt.axis(ymin=0, ymax=20)
    plt.tight_layout()
    # plt.grid()
    # plt.show()
    # filename = "fft_plot_class%i.pdf"
    # plt.savefig(filename % (nf), format='PDF', bbox_inches='tight')


if __name__ == "__main__":
    # data loading
    data_files1 = "targetoutA_8channels.txt"
    data_files2 = "GAN_fakeA_05-05-19-56.npy"
    data_files3 = "VAE_fakeA_05-09-14-24.npy"
    data = np.load(data_files2)
    data = np.asarray(data)
    # data = np.squeeze(data, 1)
    data3 = np.load(data_files3)
    data3 = np.asarray(data3)
    # data3 = np.squeeze(data3, 1)
    # data = np.concatenate(data)
    data2 = np.loadtxt(data_files1, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
    # print("data's shape : ", data.shape)
    data2 = np.reshape(data2, (170, 8, 240), 'F')

    data_fft = fft_data(data)
    data_fft2 = fft_data(data2)
    data_fft3 = fft_data(data3)
    x_axis = np.linspace(0, (data.shape[-1] / T) / 2, data_fft.shape[0])

    plot_fft(x_axis, data_fft, 'red', '--')
    plot_fft(x_axis, data_fft2, 'blue', '-')
    plot_fft(x_axis, data_fft3, 'green', '-.')
    plt.grid(linestyle='--')
    plt.legend(['GAN fake data', 'real data', 'VAE fake data'])
    plt.show()
