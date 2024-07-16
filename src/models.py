import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


class BasicConvClassifier(nn.Module):
    def __init__(
        self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)


class NewConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)


class NewConvClassifier(nn.Module):
    def __init__(
        self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            NewConvBlock(in_channels, hid_dim, p_drop=0.4),
            NewConvBlock(hid_dim, hid_dim * 2, p_drop=0.4),
            NewConvBlock(hid_dim * 2, hid_dim * 2, p_drop=0.4),
            NewConvBlock(hid_dim * 2, hid_dim * 3, p_drop=0.4),
            NewConvBlock(hid_dim * 3, hid_dim * 3, p_drop=0.4),
            NewConvBlock(hid_dim * 3, hid_dim * 4, p_drop=0.4),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 4, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        # baseline correction
        X -= X[:, :, :30].mean(dim=2, keepdim=True)

        X = self.blocks(X)

        return self.head(X)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    r"""
    https://github.com/torcheeg/torcheeg/blob/9c2c2dd333ca5a92ea7d255dc07a9525d2df803f/torcheeg/models/cnn/eegnet.py

    Args:
        seq_len (int): Number of data points included in each EEG chunk, i.e., T in the paper.
        in_channels (int): The number of electrodes, i.e., C in the paper.
        F1 (int): The filter number of block 1, i.e., F_1 in the paper.
        F2 (int): The filter number of block 2, i.e., F_2 in the paper.
        D (int): The depth multiplier (number of spatial filters), i.e., D in the paper.
        num_classes (int): The number of classes to predict, i.e., N in the paper.
        kernel_1 (int): The filter size of block 1.
        kernel_2 (int): The filter size of block 2.
        dropout (float): Probability of an element to be zeroed in the dropout layers.
    """

    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_1: int = 64,
        kernel_2: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_1),
                stride=1,
                padding=(0, kernel_1 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(in_channels, 1),
                max_norm=1,
                stride=1,
                padding=(0, 0),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, kernel_2),
                stride=1,
                padding=(0, kernel_2 // 2),
                bias=False,
                groups=F1 * D,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=1,
                padding=(0, 0),
                groups=1,
                bias=False,
                stride=1,
            ),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout),
        )

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.in_channels, self.seq_len)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[1] * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        logger.debug(f"block1: {x.shape}")
        x = self.block2(x)
        logger.debug(f"block2: {x.shape}")
        x = x.flatten(start_dim=1)
        logger.debug(f"flattened: {x.shape}")
        x = self.lin(x)
        logger.debug(f"lin: {x.shape}")

        return x


if __name__ == "__main__":
    from torchinfo import summary

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())

    batch_size = 128
    num_classes = 1854
    seq_len = 281
    in_channels = 271

    model = BasicConvClassifier(
        num_classes=num_classes, seq_len=seq_len, in_channels=in_channels
    )

    summary(
        model,
        input_size=(batch_size, in_channels, seq_len),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3,
        row_settings=["var_names"],
    )

    model = NewConvClassifier(
        num_classes=num_classes, seq_len=seq_len, in_channels=in_channels
    )

    summary(
        model,
        input_size=(batch_size, in_channels, seq_len),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3,
        row_settings=["var_names"],
    )

    model = EEGNet(num_classes=num_classes, seq_len=seq_len, in_channels=in_channels)

    summary(
        model,
        input_size=(batch_size, in_channels, seq_len),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3,
        row_settings=["var_names"],
    )
