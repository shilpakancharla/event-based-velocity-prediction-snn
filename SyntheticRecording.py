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
                                                         time_window = 1000)
                                            ])
      transformed_frames = frame_transform(events)
      vel_x = np.array(self.target.loc[index][0]).astype('float')
      vel_y = np.array(self.target.loc[index][1]).astype('float')
      vel_z = np.array(self.target.loc[index][2]).astype('float')
      
      vel_xyz = []
      tensor_vel_x = torch.from_numpy(vel_x)
      vel_xyz.append(tensor_vel_x)
      tensor_vel_y = torch.from_numpy(vel_y)
      vel_xyz.append(tensor_vel_y)
      tensor_vel_z = torch.from_numpy(vel_z)
      vel_xyz.append(tensor_vel_z)
      
      # Map-style dataset
      sample = {'frames': torch.tensor(transformed_frames),
                'vel_xyz': torch.FloatTensor(vel_xyz)}
      return sample

    """
        Returns the size of the dataset.
    """
    def __len__(self):
        return self.target.shape[0]