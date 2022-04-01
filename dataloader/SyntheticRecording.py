import ast
import tonic
import numpy as np 
import pandas as pd
from torch.utils.data import Subset
import tonic.transforms as transforms
from sklearn.model_selection import train_test_split

"""
  Author: Shilpa Kancharla
  Last updated: March 24, 2022
"""

class SyntheticRecording(tonic.Dataset):
  """
      Synthetic event camera recordings dataset.
  """
  def __init__(self, df):
    super(SyntheticRecording, self).__init__()
    self.df = df.reset_index(drop = True) # Address index out of order issue
    self.events = self.df['Events'] # Select only last column of dataframe
    self.target = self.df[['Vel_x', 'Vel_y', 'Vel_z']] # Select every column except last column of dataframe
    self.sensor_size = (1920, 1080, 2)
    # Denoise removes isolated, one-off events
    self.frame_transform = transforms.Compose([transforms.Denoise(filter_time = 1000000),
                                               transforms.ToFrame(sensor_size = (1920, 1080, 2), n_event_bins = 5)]) 
    
  """
      Retrieve the index i to get the ith sample from the dataset. Apply the appropriate transformations.
  """
  def __getitem__(self, index):
    list_ = ast.literal_eval(self.events[index]) # Convert string literal to list
    t, x, y, p = [], [], [], []
    for e in list_:
      t.append(e[0] * 1e6) # Convert to microseconds
      x.append(e[1])
      y.append(e[2])
      p.append(e[3])
    structured_events = tonic.io.make_structured_array(x, y, t, p) # Ordering is xytp now
    transformed_frames = self.frame_transform(structured_events)
    vel_xyz = []
    vel_x = self.target.loc[index][0]
    vel_xyz.append(vel_x)
    vel_y = self.target.loc[index][1]
    vel_xyz.append(vel_y)
    vel_z = self.target.loc[index][2]
    vel_xyz.append(vel_z)

    frames_tensor = torch.tensor(transformed_frames)
    vel_tensor = torch.tensor(vel_xyz, dtype = torch.float32)
    return frames_tensor, vel_tensor

  """
      Returns the size of the dataset.
  """
  def __len__(self):
    return len(self.df)