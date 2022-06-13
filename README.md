# GAN-based EEG Signal Generation

_Wenyuan Zhao, Lindong Ye, Ziyan Cui_

This repo provides code for implementing [GAN-based EEG Signal Generation](https://github.com/wyzhao030/WGAN-GP/blob/main/GAN-based_algorithm_for_EEG_signals.pdf).

* [📦 Install ](#install) -- Install relevant dependencies and the project
* [🔧 Usage ](#usage) -- Instructions on running different experiments in this project

## Install 
To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c pytorch
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install requirements
pip install -r requirements.txt

# Finally, clone the project
git clone https://github.com/wyzhao030/WGAN-GP.git
```

## Usage:
The default branch for the latest and stable changes is `release`. 

* To run WGAN-GP generative model
```bash
python setup_WGANGP.py
```

* To run VAE generative model
```bash
python setup_VAE.py
```

* Important parameter
```bash
flag = 1    % subjectA
flag = 0    % subjectB
filename1 = "GAN_fakeA_04-30-23-39.npy"     % filename is named by the time you run the code 
filename = "targetoutA_8channels.txt"       % Change "04-30-23-39" for necessity
filenameD = "DlossA_04-30-23-39.npy"
filenameG = "GlossA_04-30-23-39.npy"
```

* Plot Figures
```bash
python fft_plot.py
python plotresult.py
python plotresult_VAE.py
```

* Data Augmented CNN Classification
```bash
python train_CNN.py
```





