import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
#import adabound
import numpy as np
import os
from readData import prepare
from model import My_loss_Lightning
from model import My_loss_Radar
from model import ConvLSTM3D
from model import RLNET

os.environ["CUDA_VISIBLE_DEVICES"] = "6" # "0, 1"
GPUID = 0

num_epochs = 0 #训练轮数
trainnum = 1408
batch_size = 16#32
fs = 80.0
radar_w = 0.2
lightning_w = 1.0

height = 3
row = 64
col = 64
radar_input_channel = 1
radar_hidden_channels = [4,8,4,1]
radar_kernel_size = 3
lightning_input_channel = 1
lightning_hidden_channels = [4,8,4,1]
lightning_kernel_size = 3
step = 10

datasetFilename = "data\\dataset.txt"
tmp = np.zeros((trainnum, 20, height+1, row, col))
trainlx = np.zeros((trainnum//batch_size+1, step, batch_size, lightning_input_channel, row, col))
trainly = np.zeros((trainnum//batch_size+1, step, batch_size, lightning_hidden_channels[-1], row, col))
trainrx = np.zeros((trainnum//batch_size+1, step, batch_size, radar_input_channel, height, row, col))
trainry = np.zeros((trainnum//batch_size+1, step, batch_size, radar_hidden_channels[-1], height, row, col))
sf = True


def show(np2d):
    for j in range(row):
        for k in range(col):
            fk = np2d[j][k] * fs
            fk = np.round(fk, decimals=2)
            print str(fk),
        print(' ')
    print('======================================================')

if __name__ == '__main__':
    torch.set_default_tensor_type("torch.DoubleTensor")
    convlstm3d = ConvLSTM3D(input_channels=radar_input_channel, hidden_channels=radar_hidden_channels, kernel_size=radar_kernel_size, step=step).cuda(GPUID)
    rlnet = RLNET(radarlstm=convlstm3d, input_channels=lightning_input_channel, hidden_channels=lightning_hidden_channels, kernel_size=lightning_kernel_size, step=step).cuda(GPUID)
    loss_fnr = My_loss_Radar().cuda(GPUID)
    loss_fnl = My_loss_Lightning().cuda(GPUID)

    #rlnet.load_state_dict(torch.load('model/oldrlnet'))
    optimizer = optim.Adam(rlnet.parameters(), lr=0.0001, betas=(0.9, 0.99))

    if num_epochs>0:
        prepare(0, datasetFilename, sf, tmp, trainlx, trainly, trainrx, trainry)
        trainlx = trainlx / fs
        trainly = trainly / fs
        trainrx = trainrx / fs
        trainry = trainry / fs
        print(trainlx.shape)
        print(trainly.shape)
        print(trainrx.shape)
        print(trainry.shape)

    batchNum = trainnum // batch_size
    for epoch in range(num_epochs*batchNum):
        if (epoch == 0):
            radar_w = 2.0
        if (epoch == 300):
            radar_w = 0.15
        tmpi = epoch%(trainnum//batch_size)
        radar_input = trainrx[tmpi]
        radar_target = trainry[tmpi]
        lightning_input = trainlx[tmpi]
        lightning_target = trainly[tmpi]
        radar_input = Variable(torch.from_numpy(radar_input)).double().cuda(GPUID)
        radar_target = Variable(torch.from_numpy(radar_target)).double().cuda(GPUID)
        lightning_input = Variable(torch.from_numpy(lightning_input)).double().cuda(GPUID)
        lightning_target = Variable(torch.from_numpy(lightning_target)).double().cuda(GPUID)

        radar_out, lightning_out = rlnet(radar_input, lightning_input)

        loss = radar_w*loss_fnr(radar_out[0].double(),radar_target[0]).cuda(GPUID)
        for i in range(1,10):
            loss += radar_w*loss_fnr(radar_out[i].double(),radar_target[i])
        for i in range(10):
            loss += lightning_w*loss_fnl(lightning_out[i].double(),lightning_target[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch) % 10 == 0:
            print(str(epoch) + "loss:{}".format(loss.data))
            torch.save(rlnet.state_dict(), 'model/rlnet')

    rlnet.eval()  #
    sf = False
    trainnum = 512
    prepare(1488, datasetFilename, sf, tmp, trainlx, trainly, trainrx, trainry)
    trainlx = trainlx / fs
    trainly = trainly / fs
    trainrx = trainrx / fs
    trainry = trainry / fs

    print("start")
    for bi in range(trainnum // batch_size):
        radar_input = trainrx[bi]
        radar_target = trainry[bi]
        lightning_input = trainlx[bi]
        lightning_target = trainly[bi]
        radar_input = Variable(torch.from_numpy(radar_input)).double().cuda(GPUID)
        radar_target = Variable(torch.from_numpy(radar_target)).double().cuda(GPUID)
        lightning_input = Variable(torch.from_numpy(lightning_input)).double().cuda(GPUID)
        lightning_target = Variable(torch.from_numpy(lightning_target)).double().cuda(GPUID)

        radar_out, lightning_out = rlnet(radar_input, lightning_input)

        for i in range(10):
            radar_out[i] = radar_out[i].cpu()
            lightning_out[i] = lightning_out[i].cpu()
        for i in range(batch_size):
            for t in range(10):
                show(trainlx[bi][t][i][0])
                # for h in range(height):
                # show(trainrx[0][t][i][0][h])
            for t in range(10):
                show(trainly[bi][t][i][0])
                # for h in range(height):
                # show(trainry[0][t][i][0][h])
            for t in range(10):
                show(lightning_out[t][i][0].data.numpy())
                # for h in range(height):
                # show(radar_out[t][i][0][h].data.numpy())
