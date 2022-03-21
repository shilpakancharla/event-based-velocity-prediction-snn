import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset

class SpikingConvolutionalNeuralNetwork(nn.Model):
    def __init__(self, number_inputs, number_hidden, number_outputs):
        super().__init__()
        self.number_inputs = number_inputs
        self.number_hidden = number_hidden
        self.number_outputs = number_outputs
        self.spike_grad = surrogate.fast_sigmoid(slope = 75)
        self.beta = 0.5 

        # Initialize layers
        self.conv1 = nn.Conv2d(2, 12, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.leaky1 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad, init_hidden = True)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.leaky2 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad, init_hidden = True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 5 * 5, 10)
        self.leaky3 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad, init_hidden = True, output = True)