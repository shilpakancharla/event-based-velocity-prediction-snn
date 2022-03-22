import ast
import tonic
import numpy as np 
import pandas as pd
from tonic.transforms as transforms

"""
  Author: Shilpa Kancharla
  Last updated: March 22, 2022
"""

class SyntheticRecording(tonic.Dataset):
    """
        Synthetic event camera recordings dataset.
    """
    def __init__(self, csv_file):
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
      list_ = ast.literal_eval(self.events[index])
      t = []
      x = []
      y = []
      p = []
      for e in list_:
        t.append(e[0] * 1e6) # Convert to microseconds
        x.append(e[1])
        y.append(e[2])
        p.append(e[3])
      events = tonic.io.make_structured_array(x, y, t, p) # Ordering is xytp now
      
      # Denoise removes isolated, one-off events
      frame_transform = transforms.Compose([transforms.Denoise(filter_time = 10000), 
                                            transforms.ToFrame(sensor_size = self.sensor_size, 
                                                         time_window = 1000)])
      transformed_frames = frame_transform(events)
      sample = {'events': transformed_frames, 
                'vel_x': self.target.loc[index][0],
                'vel_y': self.target.loc[index][1],
                'vel_z': self.target.loc[index][2]}
      return sample

    """
        Returns the size of the dataset.
    """
    def __len__(self):
        return self.target.shape[0]