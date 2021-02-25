import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
#import adabound
import numpy as np
import os
GPUID = 0


height = 3
row = 64
col = 64
step = 10

class My_loss_Lightning(nn.Module):
    def __init__(self):
        super(My_loss_Lightning, self).__init__()

    def forward(self, x, y):
        ny = fs*y
        l = torch.mean(torch.pow(x - y, 2.0))+torch.mean(torch.sigmoid(ny)*torch.pow(x - y, 2.0))
        return l

class My_loss_Radar(nn.Module):
    def __init__(self):
        super(My_loss_Radar, self).__init__()

    def forward(self, x, y):
        l = torch.mean(torch.pow(x - y, 2.0))
        return l

class ConvLSTMCell2D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell2D, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        #print(c.cpu().data.numpy().shape)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda(GPUID)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda(GPUID)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda(GPUID)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(GPUID),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(GPUID))


class ConvLSTMCell3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell3D, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)

        #self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wxi = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).cuda(GPUID)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).cuda(GPUID)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).cuda(GPUID)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
            assert shape[2] == self.Wci.size()[4], 'Input Width2 Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1], shape[2])).cuda(GPUID),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1], shape[2])).cuda(GPUID))


class ConvLSTM3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1):
        self.padding = int((kernel_size - 1) / 2)
        super(ConvLSTM3D, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        #self.Winit = nn.Conv3d(self.input_channels[0], self.input_channels[0], self.kernel_size, 2, self.padding, bias=True)
        for i in range(self.num_layers):
            name = 'cell3d{}'.format(i)
            cell = ConvLSTMCell3D(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, inputs):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = inputs[step]
            for i in range(self.num_layers):
                name = 'cell3d{}'.format(i)
                if step == 0:
                    bsize, _, height, width, width2 = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width, width2))
                    internal_state.append((h, c))
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            if(step==self.step-1):
                outputs.append(x)

        for step in range(self.step - 1):
            x = outputs[step]
            for i in range(self.num_layers):
                name = 'cell3d{}'.format(i)
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            outputs.append(x)

        return outputs

class RLNET(nn.Module):
    def __init__(self, radarlstm, input_channels, hidden_channels, kernel_size, step=1):
        self.padding = int((kernel_size - 1) / 2)
        super(RLNET, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.Wrl = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(height,1,1), stride=1, padding=0, bias=True)
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cellen2d{}'.format(i)
            cell = ConvLSTMCell2D(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
        for i in range(self.num_layers):
            name = 'cellde2d{}'.format(i)
            cell = ConvLSTMCell2D(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
        name = 'radarlstm'
        #cell =  ConvLSTM3D(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=3, step=step)
        setattr(self, name, radarlstm)

    def forward(self, radar_inputs, lightning_inputs):
        internal_state = []
        outputs = []
        radars = getattr(self, 'radarlstm')(radar_inputs)
        for step in range(self.step):
            #print("a" + str(step))
            x = lightning_inputs[step]
            for i in range(self.num_layers):
                name = 'cellen2d{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                    internal_state.append((h, c))
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

        for step in range(self.step):
            #print("b" + str(step))
            x = torch.sigmoid(self.Wrl(radars[step]))
            x = torch.squeeze(x,2)
            for i in range(self.num_layers):
                name = 'cellde2d{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            outputs.append(x)
        return radars, outputs