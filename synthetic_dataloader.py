import numpy as np 
import pandas as pd
from tonic.datasets import Dataset

DATA_PATH = "data/processed/"

class SyntheticRecording(Dataset):
    sensor_size = (1920, 1080)
    ordering = "txyp" # (timestamp, x, y, polarity)

    def __init__(self, train = True, transform = None, target_transform = None):
        super(SyntheticRecording, self).__init__(transform = transform, target_transform = target_transform)

        self.train = train 

        if train:
            self.filename = TRAINING_PATH
        else:
            self.filename = TEST_PATH

    def __getitem__(self, index):
        # Read contents from .csv files
        events = None
        target = None
        
        return events, target

    def __len__(self):
        file = h5py.File(self.filename, 'r')
        return len(file['labels'])