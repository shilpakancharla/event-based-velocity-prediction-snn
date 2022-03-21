import os
import numpy as np 
import pandas as pd
from tonic.datasets import Dataset

"""
  Author: Shilpa Kancharla
  Last updated: March 21, 2022
"""

DATA_PATH = "data/processed/"

class SyntheticRecording(Dataset):
    """
        Synthetic event camera recordings dataset.
    """
    sensor_size = (1920, 1080)
    ordering = "txyp" # (timestamp, x, y, polarity)

    def __init__(self, train = True, transform = None, target_transform = None):
        super(SyntheticRecording, self).__init__(transform = transform, target_transform = target_transform)

        self.train = train 

        if train:
            self.filename = TRAINING_PATH
        else:
            self.filename = TEST_PATH

    """
        Retrieve the index i to get the ith sample from the dataset.
    """
    def __getitem__(self, index):
        # Read contents from .csv files
        events = None
        target = None
        
        return events, target

    """
        Returns the size of the dataset.
    """
    def __len__(self):
        file = h5py.File(self.filename, 'r')
        return len(file['labels'])