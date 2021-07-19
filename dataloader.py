import random
import numpy as np
from pathlib import Path
import sys


class DataLoader:
    def __init__(self):
        train_data_paths = list(Path("./train_data/").rglob("*.npy"))
        test_data_paths = list(Path("./test_data/").rglob("*.npy"))

        if len(train_data_paths) == 0:
            print("Training data not found. Please ensure your training data is in the \"train_data\" subfolder.")
            sys.exit()
        if len(test_data_paths) == 0:
            print("Testing data not found. Please ensure your training data is in the \"test_data\" subfolder.")
            sys.exit()

        self.train_data = []
        self.test_data = []
        for path in train_data_paths:
            self.train_data.append(np.load(path))
        for path in test_data_paths:
            self.test_data.append(np.load(path))

        self.train_max = 0
        self.test_max = 0
        for data in self.train_data:
            self.train_max = max(self.train_max, data.shape[1])
        for data in self.test_data:
            self.test_max = max(self.test_max, data.shape[1])

    def sample_train(self, length):
        if self.train_max < length:
            return None

        data = None
        while data is None:
            sample = random.sample(self.train_data, 1)[0]
            if sample.shape[1] >= length:
                start_frame = random.randint(0, sample.shape[1] - length)
                data = sample[:, start_frame:start_frame + length]

        return data

    def sample_test(self, length):
        if self.test_max < length:
            return None

        data = None
        while data is None:
            sample = random.sample(self.test_data, 1)[0]
            if sample.shape[1] >= length:
                start_frame = random.randint(0, sample.shape[1] - length)
                data = sample[:, start_frame:start_frame + length]

        return data
