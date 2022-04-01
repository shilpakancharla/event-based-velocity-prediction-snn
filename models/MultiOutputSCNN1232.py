import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from snntorch import surrogate

"""
  Author: Shilpa Kancharla
  Last updated: April 1, 2022
"""

class MultiOutputSCNN1232(nn.Module):
  def __init__(self, beta, spike_grad):
    super(MultiOutputSCNN1232, self).__init__()
    self.beta = beta
    self.spike_grad = spike_grad

    # Initialize the layers
    self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 12, kernel_size = 5)
    self.lif1 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 32, kernel_size = 5)
    self.lif2 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(in_features = 32 * 267 * 477, out_features = 3)

  def forward(self, x):
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()

    vel_xyz = None

    for step in range(x.size(0)):
      res = F.max_pool2d(self.conv1(x[step]), (2, 2))
      spk1, mem1 = self.lif1(res, mem1)
      res2 = F.max_pool2d(self.conv2(spk1), (2, 2))
      spk2, mem2 = self.lif2(res2, mem2)
      flat = self.flat(spk2)
      vel_xyz = self.fc1(flat)
      
    return vel_xyz