import os
from glob import glob
from typing import Tuple

import numpy as np
import torch
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(
            self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy"
        )
        X = torch.from_numpy(np.load(X_path))

        subject_idx_path = os.path.join(
            self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy"
        )
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            y_path = os.path.join(
                self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy"
            )
            y = torch.from_numpy(np.load(y_path))

            return X, y, subject_idx
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
