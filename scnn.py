import gc
import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from snntorch import surrogate
from torch.utils.data import DataLoader

"""
  Author: Shilpa Kancharla
  Last updated: March 24, 2022
"""

class MultiOutputSNN(nn.Module):
  def __init__(self, beta, spike_grad):
    super(MultiOutputSNN, self).__init__()
    self.beta = beta
    self.spike_grad = spike_grad

    # Initialize the layers
    self.conv1 = nn.Conv2d(2, 12, 5)
    #self.maxpool1 = nn.MaxPool2d(2)
    self.lif1 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.conv2 = nn.Conv2d(12, 32, 5)
    #self.maxpool2 = nn.MaxPool2d(2)
    self.lif2 = snn.Leaky(beta = self.beta, spike_grad = self.spike_grad)
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(32 * 267 * 477, 1) # vel_x output
    self.fc2 = nn.Linear(32 * 267 * 477, 1) # vel_y output
    self.fc3 = nn.Linear(32 * 267 * 477, 1) # vel_x output

  def forward(self, x):
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()

    # Record final layer
    spk2_rec = []
    mem2_rec = []

    vel_x = None
    vel_y = None
    vel_z = None

    for step in range(x.size(0)):
      res = F.max_pool2d(self.conv1(x[step]), (2, 2))
      spk1, mem1 = self.lif1(res, mem1)
      res2 = F.max_pool2d(self.conv2(spk1), (2, 2))
      spk2, mem2 = self.lif2(res2, mem2)
      spk2_rec.append(spk2)
      mem2_rec.append(mem2)
      flat = self.flat(spk2)
      vel_x = self.fc1(flat)
      vel_y = self.fc2(flat)
      vel_z = self.fc3(flat)
      
    return vel_x, vel_y, vel_z

def training_loop(net, train_loader, test_loader, optimizer, loss_fn):
  num_epochs = 1
  
  # Store loss history for future plotting
  loss_history_x = []
  loss_history_y = []
  loss_history_z = []
  loss_history = []
  test_loss_history_x = []
  test_loss_history_y = []
  test_loss_history_z = []
  test_loss_history = []
  loss = 0
  test_loss = 0 
  history = dict()
  counter = 0

  for epoch in range(num_epochs):
    batch = iter(train_loader)
    for data, targets in batch: # Training loop
      data = data.cuda()
      targets = targets.cuda()
      net.train() # Forward pass
      vel_x, vel_y, vel_z = net(data) # Predictions
      loss_val_x = torch.sqrt(loss_fn(vel_x[0][0].float(), targets[0][0].float()))
      loss_history_x.append(loss_val_x.item())
      loss_val_y = torch.sqrt(loss_fn(vel_y[0][0].float(), targets[0][1].float()))
      loss_history_y.append(loss_val_y.item())
      loss_val_z = torch.sqrt(loss_fn(vel_z[0][0].float(), targets[0][2].float()))
      loss_history_z.append(loss_val_z.item())

      loss = loss_val_x + loss_val_y + loss_val_z
      loss_history.append(loss.item())

      # Gradient calculation and weight update
      optimizer.zero_grad()
      loss.backward() 
      optimizer.step()

      with torch.no_grad(): # Test loop
        net.eval() # Test forward pass
        test_data, test_targets = next(iter(test_loader))
        test_data = test_data.cuda()
        test_targets = test_targets.cuda()
        test_vel_x, test_vel_y, test_vel_z = net(test_data) # Predictions
        test_loss_val_x = torch.sqrt(loss_fn(test_vel_x[0][0].float(), test_targets[0][0].float()))
        test_loss_history_x.append(test_loss_val_x.item())
        test_loss_val_y = torch.sqrt(loss_fn(test_vel_y[0][0].float(), test_targets[0][1].float()))
        test_loss_history_y.append(test_loss_val_y.item())
        test_loss_val_z = torch.sqrt(loss_fn(test_vel_z[0][0].float(), test_targets[0][2].float()))
        test_loss_history_z.append(test_loss_val_z.item())

        test_loss = test_loss_val_x + test_loss_val_y + test_loss_val_z
        test_loss_history.append(test_loss.item())

        if counter % 10 == 0: # Print every 10 results
          print(f"Training Iteration {counter}: vel_x RMSE: {loss_val_x}, vel_y RMSE: {loss_val_y}, vel_z RMSE: {loss_val_z}, total RMSE: {loss}")
          print(f"Test Iteration {counter}: vel_x RMSE: {test_loss_val_x}, vel_y RMSE: {test_loss_val_y}, vel_z RMSE: {test_loss_val_z}, total RMSE: {test_loss}\n")
        counter += 1

  history['Training Velocity X RMSE'] = loss_history_x
  history['Training Velocity Y RMSE'] = loss_history_y
  history['Training Velocity Z RMSE'] = loss_history_z
  history['Training Total RMSE'] = loss_history
  history['Test Velocity X RMSE'] = test_loss_history_x
  history['Test Velocity Y RMSE'] = test_loss_history_y
  history['Test Velocity Z RMSE'] = test_loss_history_z
  history['Test Total RMSE'] = test_loss_history
  
  return history
  
if __name__ == "__main__":
  gc.collect()
  torch.cuda.empty_cache()
  sample_csv = SRC + 'GT_HW1_1_DROPPED.csv'
  datasets = SyntheticRecording(sample_csv) # Create a PyTorch dataset object
  train_dataset, test_dataset = train_val_dataset(datasets) # Train/test split
  # Create DataLoader objects
  train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True,
                            collate_fn = PadMultiOutputTensor(),
                            num_workers = 2, drop_last = False)
  test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True,
                          collate_fn = PadMultiOutputTensor(),
                          num_workers = 2, drop_last = False)
  
  # Neuron and simulation parameters
  spike_grad = surrogate.fast_sigmoid(slope = 75)
  beta = 0.5
  net = MultiOutputSNN(beta, spike_grad).cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr = 2e-2)
  # total loss = lx + ly + lz
  loss_fn = nn.MSELoss()
  
  history = training_loop(net, train_loader, test_loader, optimizer, loss_fn)
