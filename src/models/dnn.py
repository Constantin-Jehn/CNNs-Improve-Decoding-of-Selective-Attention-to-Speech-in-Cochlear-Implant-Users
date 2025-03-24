import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class CNN_2(nn.Module):

    def __init__(self,
                 F1=16,
                 D=16,
                 F2=16,
                 dropout_rate=0.25,
                 input_length = 50,
                 num_input_channels = 31,
                 activation = 'ReLU',
                 conv_bias = True):
        """_summary_

        Args:
            F1 (int, optional): number of convolutional filters in first (temporal) layer. Defaults to 16.
            D (int, optional): #number of spatial filters in second layer. Defaults to 16.
            F2 (int, optional): _description_. Defaults to 16.
            dropout_rate (float, optional): _description_. Defaults to 0.25.
            input_length (int, optional): _description_. Defaults to 50.
            num_input_channels (int, optional): _description_. Defaults to 63.
            dtype (string, optional): Datatype to use for linear and Conv layers (None-> Standard, int -> torch.int16)
            activation(string, optinal): Activation function to use must be either 'ELU' or 'ReLU')
            conv_bias(bool, optional): Whether to use bias in Convolutional Layers befor Batchnorm
        """
        assert activation in ['ELU', 'ReLU', 'LeakyReLU'], f'Activation can only be ELU, ReLU or LeakyReLU but {activation} was given'
        super().__init__()

        self.input_length = input_length
        self.num_input_channels = num_input_channels
 
        self.F1=F1
        self.F2=F2
        self.D=D

        #kernel size on first temporal convolution
        self.conv1_kernel=3
        
        self.conv3_kernel=3
        self.temporalPool1 = 2
        self.temporalPool2 = 5

        # input shape is [1, C, T]
        #temporal convolution
        self.conv1 = torch.nn.Conv2d(1, self.F1, (1, self.conv1_kernel), padding=0, bias=conv_bias)
        #spatial convolution --> flattens out all channels
        self.conv2 = torch.nn.Conv2d(self.F1, self.F1*self.D, (self.num_input_channels, 1), padding=0, groups=self.F1, bias=conv_bias)
        #depthwise separable convolution (groups=self.F1*self.D) At groups= in_channels, each input channel is convolved with its own set of filters
        self.conv3 = torch.nn.Conv2d(self.F1*self.D, self.F1*self.D, (1, self.conv1_kernel), padding=0, groups=self.F1*self.D, bias=True)
        #dimension reduciton in filter space applied directly after conv3 without non-linearity in between
        self.conv4 = torch.nn.Conv2d(self.F1*self.D, self.F2, (1,1), bias=conv_bias)

        self.pool1 = torch.nn.AvgPool2d((1, self.temporalPool1))
        self.pool2 = torch.nn.AvgPool2d((1, self.temporalPool2))

        self.linear = torch.nn.Linear(self.F2*self.input_length//(self.temporalPool1*self.temporalPool2), 1)

        self.bnorm1 = torch.nn.BatchNorm2d(self.F1)
        self.bnorm2 = torch.nn.BatchNorm2d(self.F1*self.D)
        self.bnorm3 = torch.nn.BatchNorm2d(F2)

        self.dropout1 = torch.nn.Dropout2d(dropout_rate)
        self.dropout2 = torch.nn.Dropout2d(dropout_rate)

        if activation == 'ELU':
            self.activation1 = torch.nn.ELU()
            self.activation2 = torch.nn.ELU()
            self.activation3 = torch.nn.ELU()
        elif activation == 'ReLU':
            self.activation1 = torch.nn.ReLU()
            self.activation2 = torch.nn.ReLU()
            self.activation3 = torch.nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation1 = torch.nn.LeakyReLU(negative_slope=0.05)
            self.activation2 = torch.nn.LeakyReLU(negative_slope=0.05)
            self.activation3 = torch.nn.LeakyReLU(negative_slope=0.05)
    def __str__(self):
        return 'cnn'
     
    def forward(self, x):
        #x shape = [batch, C, T]
        x = x.unsqueeze(1)

        # Calculate padding dynamically
        padding_left = (self.conv1_kernel - 1) // 2
        padding_right = (self.conv1_kernel - 1) - padding_left
        # Apply padding
        x = nn.functional.pad(x, (padding_left, padding_right, 0, 0), mode='constant', value=0)

        out = self.conv1(x)
        out = self.bnorm1(out)

        out = self.conv2(out)
        out = self.bnorm2(out)
        out = self.activation1(out)
        out = self.pool1(out)
        out = self.dropout1(out)

        #shape is now [batch, DxF1, 1, T//TPool1]
        out = nn.functional.pad(out, (padding_left, padding_right, 0, 0), mode='constant', value=0)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bnorm3(out)
        out = self.activation2(out)
        out = self.pool2(out)
        out = self.dropout2(out)
        
        out = torch.flatten(out, start_dim = 1) # shape is now [batch, F2*T//(TPool1*TPool2)]
        out = self.linear(out)
        return out.flatten()