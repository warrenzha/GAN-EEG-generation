import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from modelCNN import ConvNet
import winsound


def processdata(realdata, fakedata, nondata, realnum, fakenum):
    reala = realnum // 170
    realb = realnum % 170
    real1 = np.repeat(realdata, reala, 0)
    real2 = realdata[:realb]
    real3 = np.concatenate((real1, real2), 0)

    fake1 = fakedata[:fakenum]
    datatrain = np.concatenate((real3, fake1, nondata), 0)
    datatrain = datatrain[:, :, :144]
    # datatrain = np.expand_dims(datatrain, 1)
    targetlabel = np.ones((realnum + fakenum, 1))
    nontargetlabel = np.zeros((nondata.shape[0], 1))
    datalabel = np.concatenate((targetlabel, nontargetlabel), 0)

    assert len(datatrain) == len(datalabel)
    samples = np.array(list(zip(datatrain, datalabel)))

    return samples


def minibatch(data, batch_size):
    start = 0
    while True:

        end = start + batch_size
        yield data[start:end]

        start = end
        if start >= len(data):
            break


# calculate acc
def cal_acc(pred, target):
    assert len(pred) == len(target)
    acc = np.sum(pred == target) / len(pred)
    return acc


def cal_f(pred, target):
    assert len(pred) == len(target)
    tp = 0
    for i in range(len(pred)):
        if pred[i] == target[i] and pred[i] == 1:
            tp += 1
    percision = tp / np.sum(pred == 1)
    recall = tp / np.sum(target == 1)
    f_score = (2 * percision * recall) / (percision + recall)
    return f_score, percision, recall


# train function
def train_batch(model, criterion, optimizer, data):
    model.zero_grad()

    # forward pass
    x = torch.FloatTensor([i for i in data[:, 0]]).cuda()
    _, height, width = x.size()
    x = x.view(len(x), 1, height, width)
    y = torch.FloatTensor([i for i in data[:, 1]]).cuda()
    data_set = torch.utils.data.TensorDataset(x, y)
    dataloader1 = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False,
                                              drop_last=True)
    predall = []
    for x1, y1 in dataloader1:
        pred = model(x1)
        loss = criterion(pred.view(-1), torch.squeeze(y1))
        loss.backward()
        optimizer.step()
        pred = pred.cpu().detach().numpy().reshape(-1)
        pred = np.array([1 if n >= 0.5 else 0 for n in pred])
        predall.append(pred)
        # back proporgation

    return predall


def val_batch(model, criterion, optimizer, data):
    with torch.no_grad():
        # forward pass
        x = torch.FloatTensor([i for i in data[:, 0]]).cuda()
        _, height, width = x.size()
        x = x.view(len(x), 1, height, width)
        y = torch.FloatTensor([i for i in data[:, 1]]).cuda()
        dataloader2 = torch.utils.data.DataLoader(dataset=x, batch_size=batch_size, shuffle=False,
                                                  drop_last=True)
        predall = []
        for x2 in dataloader2:
            pred = model(x2)
            pred = pred.cpu().detach().numpy().reshape(-1)
            pred = np.array([1 if n >= 0.5 else 0 for n in pred])
            predall.append(pred)
        return predall


accumean = np.empty((2, 5, 6))
accu = np.empty((2, 5, 10, 10, 6))
flag = 2
for nnn in range(2):

    if nnn == 0:
        filename1 = "VAE_fakeB_05-10-13-13.npy"
    else:
        filename1 = "GAN_fakeB_05-10-10-38.npy"
    data_fake = np.load(filename1)
    data_fake = np.squeeze(data_fake, 1)

    if flag == 1:
        filename2 = "targetoutA_8channels.txt"
        filename3 = "NontargetoutA_8channels.txt"
    else:
        filename2 = "targetoutB_8channels.txt"
        filename3 = "NontargetoutB_8channels.txt"
    data_real = np.loadtxt(filename2, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
    data_real = np.reshape(data_real, (170, 8, 240), 'F')
    data_nonreal = np.loadtxt(filename3, dtype=np.double, delimiter=',', skiprows=0, encoding='utf-8')
    data_nonreal = np.reshape(data_nonreal, (850, 8, 240), 'F')

    data1 = processdata(data_real, data_fake, data_nonreal, 170, 0)
    data2 = processdata(data_real, data_fake, data_nonreal, 170, 255)
    data3 = processdata(data_real, data_fake, data_nonreal, 425, 0)
    data4 = processdata(data_real, data_fake, data_nonreal, 170, 680)
    data5 = processdata(data_real, data_fake, data_nonreal, 850, 0)

    # print(data1.shape)
    # print(data2.shape)
    # print(data3.shape)
    # print(data4.shape)
    # print(data5.shape)

    data = (data1, data2, data3, data4, data5)
    batch_size = 10
    epoch = 30

    # for debug
    # np.random.seed(0)

    # divide data into minibatches

    for mm in range(5):

        ####### main logic #######
        # load the data
        # data format: [(x, y, y_stim)]
        for j in range(10):
            datanow = data[mm]
            data_size = len(datanow)

            # shuffle data
            shuffle_idx = np.random.permutation(data_size)
            datanow = datanow[shuffle_idx]

            # 80-20 split train/test
            cutoff = int(data_size * 80 // 100)
            train_data = datanow[:cutoff]
            test_data = datanow[cutoff:]

            # balance label in the train_data
            train_data_size = len(train_data)
            train_data_true_count = np.sum([x[1] for x in train_data])
            train_data_false_count = train_data_size - train_data_true_count

            # assert train_data_false_count >= train_data_true_count

            # train_data_dup_count = train_data_false_count - train_data_true_count
            # train_data_true_idx = np.array([i for i, x in enumerate(train_data) if x[1] == 1])
            # train_data_true_sample_idx = np.random.choice(train_data_true_idx, train_data_dup_count, replace=True)
            # train_data_addon = train_data[train_data_true_sample_idx]
            #
            # # make sure that all the addon have true labels
            # assert all([x[1] == 1 for x in train_data_addon])
            #
            # # stack the addon to the original trainning data and shuffle again
            # train_data = np.concatenate((train_data, train_data_addon), axis=0)
            # train_data_size = len(train_data)
            # shuffle_idx = np.random.permutation(train_data_size)
            # train_data = train_data[shuffle_idx]

            # init model
            model = ConvNet()
            model = model.cuda()

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5 * 2, weight_decay=1e-2)

            # train loop
            # use k-fold validation
            k_fold = 10
            fold_size = int(train_data_size // k_fold)
            # print(f"epoch:{j}")
            for i in range(k_fold):

                # split data into train/val
                val_data_curr_fold = train_data[i * fold_size:(i + 1) * fold_size]
                train_data_curr_fold_head = train_data[:i * fold_size]
                train_data_curr_fold_tail = train_data[(i + 1) * fold_size:]
                train_data_curr_fold = np.concatenate((train_data_curr_fold_head, train_data_curr_fold_tail))

                # epoch
                model = model.train()
                for curr_epoch in range(epoch):
                    # train minibatch
                    # train_pred = []
                    train_data_curr_fold = train_data_curr_fold[np.random.permutation(len(train_data_curr_fold))]

                    train_pred = train_batch(model, criterion, optimizer, train_data_curr_fold)
                    # train_pred.append(train_batch_pred)
                    train_pred = np.concatenate(train_pred, axis=0)

                    # val_pred = []

                    val_pred = val_batch(model, criterion, optimizer, val_data_curr_fold)
                    # val_pred.append(val_batch_pred)
                    val_pred = np.concatenate(val_pred, axis=0)

                    # calculate acc
                    train_target = train_data_curr_fold[:, 1].reshape(-1)
                    train_target = train_target[:len(train_pred)]
                    train_acc = cal_acc(train_pred, train_target)
                    val_target = val_data_curr_fold[:, 1].reshape(-1)
                    val_target = val_target[:len(val_pred)]
                    val_acc = cal_acc(val_pred, val_target)

                    # print stats

                # print(f"fold: {i}, train acc: {train_acc}, val acc: {val_acc}")
                accu[nnn, mm, j, i, 0] = train_acc
                accu[nnn, mm, j, i, 1] = val_acc
                # test acc
                model = model.eval()
                # test_pred = []

                test_pred = val_batch(model, criterion, optimizer, test_data)
                # test_pred.append(test_batch_pred)
                test_pred = np.concatenate(test_pred, axis=0)
                test_target = test_data[:, 1].reshape(-1)
                test_target = test_target[:len(test_pred)]
                test_acc = cal_acc(test_pred, test_target)
                test_f_score, test_percision, test_recall = cal_f(test_pred, test_target)
                accu[nnn, mm, j, i, 2] = test_acc
                accu[nnn, mm, j, i, 3] = test_percision
                accu[nnn, mm, j, i, 4] = test_recall
                accu[nnn, mm, j, i, 5] = test_f_score
                # print(f"fold: {i}, test acc: {test_acc}")
                # print(
                #     f"fold: {i}, test percision: {test_percision}, test recall: {test_recall}, test f score: {test_f_score}")

        for k in range(6):
            accunow = np.squeeze(accu[nnn, mm, :, :, k])
            accumean[nnn, mm, k] = np.mean(accunow)

np.set_printoptions(precision=4)
for nnn in range(2):
    for mm in range(5):
        if nnn == 0:
            print(f"type:VAE")
        else:
            print(f"type:GAN")

        print(f"dataset:{mm}")
        print(f"mean train acc:{accumean[nnn, mm, 0]}")
        print(f"mean val acc:{accumean[nnn, mm, 1]}")
        print(f"mean test acc:{accumean[nnn, mm, 2]}")
        print(f"mean test precision:{accumean[nnn, mm, 3]}")
        print(f"mean test recall:{accumean[nnn, mm, 4]}")
        print(f"mean test f score:{accumean[nnn, mm, 5]}")

time_str = datetime.datetime.now().strftime('%m-%d-%H-%M')
str1 = 'accuB_all'
str2 = '.npy'
str3 = 'accumeanB_all'
filename5 = str1 + time_str + str2
filename6 = str3 + time_str + str2
np.save(filename5, accu)
np.save(filename6, accumean)
winsound.Beep(600, 3000)
