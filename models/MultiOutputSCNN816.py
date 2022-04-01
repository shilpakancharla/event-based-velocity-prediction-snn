import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from snntorch import surrogate

"""
  Author: Shilpa Kancharla
  Last updated: April 1, 2022
"""

class MultiOutputSCNN816(nn.Module):
  def __init__(self, alpha, beta, spike_grad):
    super(MultiOutputSCNN816, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.spike_grad = spike_grad

    # Initialize the layers
    self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 5)
    self.lif1 = snn.Synaptic(alpha = self.alpha, beta = self.beta, spike_grad = self.spike_grad)
    self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5)
    self.lif2 = snn.Synaptic(alpha = self.alpha, beta = self.beta, spike_grad = self.spike_grad)
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(in_features = 16 * 267 * 477, out_features = 3)

  def forward(self, x):
    syn1, mem1 = self.lif1.init_synaptic()
    syn2, mem2 = self.lif2.init_synaptic()

    vel_xyz = None

    for step in range(x.size(0)):
      res = F.max_pool2d(self.conv1(x[step]), (2, 2))
      spk1, syn1, mem1 = self.lif1(res, syn1, mem1)
      res2 = F.max_pool2d(self.conv2(spk1), (2, 2))
      spk2, syn2, mem2 = self.lif2(res2, syn2, mem2)
      flat = self.flat(spk2)
      vel_xyz = self.fc1(flat)
      
    return vel_xyz