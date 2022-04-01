import torch
import torch.nn as nn
from torch.utils.data import DataLoader

"""
  Author: Shilpa Kancharla
  Last updated: April 1, 2022
"""

def training_loop(net, train_loader, val_loader, optimizer, loss_fn):
  num_epochs = 1
  loss_l1 = nn.L1Loss()

  # Store loss history for future plotting
  loss_history, test_loss_history  = [], []
  loss_history_mae, test_loss_history_mae = [], []
  loss, test_loss = 0, 0
  history = dict()
  history_val = dict()
  counter = 0
  val_counter = 0

  for epoch in range(num_epochs):
    batch = iter(train_loader)
    for data, targets in batch: # Training loop
      optimizer.zero_grad() # Clear gradients for next train

      data = data.cuda()
      targets = targets.cuda()
      net.train() # Forward pass
      prediction = net(data) # Predictions

      loss = torch.sqrt(loss_fn(prediction, targets))
      loss_mae = loss_l1(prediction, targets)
      loss_history.append(loss.item())
      loss_history_mae.append(loss_mae.item())

      # Gradient calculation and weight update
      loss.backward() # Backpropagation, compute gradients
      optimizer.step() # Performs the update (apply gradients)
   
      if counter % 10 == 0: # Print every 10 results
        print(f"Training Iteration {counter}: Training RMSE: {loss.item()}, Training MAE: {loss_mae.item()}")
      counter += 1

    with torch.no_grad(): # Test loop - do not track history for backpropagation
      net.eval() # Test forward pass
      test_batch = iter(val_loader)
      for test_data, test_targets in test_batch:
        test_data = test_data.cuda()
        test_targets = test_targets.cuda()
        test_pred = net(test_data) # Predictions

        test_loss = torch.sqrt(loss_fn(test_pred, test_targets))
        test_loss_mae = loss_l1(test_pred, test_targets)
        test_loss_history.append(test_loss.item())
        test_loss_history_mae.append(test_loss_mae.item())

        if val_counter % 10 == 0:
          print(f"Validation Iteration {val_counter}: Validation RMSE: {test_loss.item()}, Validation MAE: {test_loss_mae.item()}")
        val_counter += 1

  history['Training RMSE'] = loss_history
  history['Training MAE'] = loss_history_mae
  history_val['Validation RMSE'] = test_loss_history
  history_val['Validation MAE'] = test_loss_history_mae
  torch.save(net.state_dict(), SRC + 'results/model2.pt')
  return history, history_val