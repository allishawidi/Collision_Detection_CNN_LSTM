import config as conf
import os
import random
import numpy as np
import matplotlib.pyplot as plt


train_folder = os.path.join("all_datasets", "train")
test_folder = os.path.join("all_datasets", "test")
valid_folder = os.path.join("all_datasets", "valid")
one_hot_labels = np.eye(conf.n_classes, dtype="uint8")


class data_tools:
    def __init__(self, data_folder, split_name):
        self.name = split_name
        self.data_folder = data_folder
        self._data = os.listdir(self.data_folder)
        if split_name == "train":
            self.it = int(conf.batch_size / 8)
        else:
            self.it = int(32 / 8)

    def batch_dispatch(self):
        counter = 0
        random.shuffle(self._data)
        while counter <= len(self._data):
            image_seqs = np.empty((0, 8, 140, 210, 3))
            labels = np.empty((0, 2))
            for i in range(self.it):
                npz_path = os.path.join(self.data_folder, self._data[counter])
                np_data = np.load(npz_path, "r")
                image_seqs = np.vstack((image_seqs, np_data["name1"] / 255))
                labels = np.vstack((labels, np_data["name2"]))
                counter += 1
                if counter >= len(self._data):
                    counter = 0
                    random.shuffle(self._data)
            yield image_seqs, labels