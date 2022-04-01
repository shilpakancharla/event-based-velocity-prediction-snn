import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from snntorch import surrogate

"""
  Author: Shilpa Kancharla
  Last updated: April 1, 2022
"""

class MultiOutputSNN(nn.Module):
  def __init__(self, beta, spike_grad):
    super(MultiOutputSNN, self).__init__()
    self.beta = beta
    self.spike_grad = spike_grad

    # Initialize the layers
    self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 5)
    self.lif1 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.lif2 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.lif3 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5)
    self.lif4 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.lif5 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.lif6 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(in_features = 16 * 267 * 477, out_features = 3)

  def forward(self, x):
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()
    mem3 = self.lif3.init_leaky()
    mem4 = self.lif4.init_leaky()
    mem5 = self.lif5.init_leaky()
    mem6 = self.lif6.init_leaky()

    vel_xyz = None

    for step in range(x.size(0)):
      res = F.max_pool2d(self.conv1(x[step]), (2, 2))
      spk1, mem1 = self.lif1(res, mem1)
      spk2, mem2 = self.lif2(spk1, mem2)
      spk3, mem3 = self.lif3(spk2, mem3)
      res2 = F.max_pool2d(self.conv2(spk3), (2, 2))
      spk4, mem4 = self.lif4(res2, mem4)
      spk5, mem5 = self.lif5(spk4, mem5)
      spk6, mem6 = self.lif4(spk5, mem6)
      flat = self.flat(spk6)
      vel_xyz = self.fc1(flat)
      
    return vel_xyz