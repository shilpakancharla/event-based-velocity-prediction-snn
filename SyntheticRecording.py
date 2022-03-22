import ast
import tonic
import numpy as np 
import pandas as pd
from tonic.transforms as transforms

"""
  Author: Shilpa Kancharla
  Last updated: March 22, 2022
"""

DATA_PATH = "data/processed/"

class SyntheticRecording(tonic.Dataset):
    """
        Synthetic event camera recordings dataset.
    """
    def __init__(self, csv_file, transform = None):
        super(SyntheticRecording, self).__init__()
        self.csv_file = csv_file
        df = pd.read_csv(self.csv_file, index_col = False)
        self.events = df['Events'] # Select only last column of dataframe
        self.target = df[['Vel_x', 'Vel_y', 'Vel_z']] # Select every column except last column of dataframe
        assert(self.target.shape[0] == len(self.events))
        self.sensor_size = (1920, 1080, 2)
    
    """
        Retrieve the index i to get the ith sample from the dataset. Apply the appropriate transformations.
    """
    def __getitem__(self, index):
      events = None
      list_ = ast.literal_eval(self.events[index])
      t = []
      x = []
      y = []
      p = []
      for e in list_:
        t.append(item[0] * 1e6) # Microseconds
        x.append(item[1])
        y.append(item[2])
        p.append(item[3])
      events = tonic.io.make_structured_array(x, y, t, p) # Ordering is xytp now
      # Denoise removes isolated, one-off events
      frame_transform = transforms.Compose([transforms.Denoise(filter_time = 10000)],
                                            [transforms.ToFrame(sensor_size = self.sensor_size, 
                                                                time_window = 1000)])
      frames = frame_transform(events)
      sample = {'events': frames, 'vel': self.target[index]}
      return sample

    """
        Returns the size of the dataset.
    """
    def __len__(self):
        return self.target.shape[0]