import h5py
import numpy as np 
from tonic.datasets import Dataset

TRAINING_PATH = ""
TEST_PATH = ""

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
        file = h5py.File(self.filename, 'r') # Read contents from .h5 file
        events = None
        target = None
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        file = h5py.File(self.filename, 'r')
        return len(file['labels'])