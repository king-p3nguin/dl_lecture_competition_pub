import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from einops.layers.torch import Rearrange

from transform import CWT

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
        kernel_size: int = 5,
        p_drop: float = 0.1,
    ) -> None:
        logger.debug(f"NewConvBlock: in_dim={in_dim}, out_dim={out_dim}")
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv2d(in_dim, out_dim, kernel_size)
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size)

        self.batchnorm0 = nn.BatchNorm2d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_dim)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_dim)

        self.avgpool = nn.AvgPool2d(2)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.avgpool(X)
        X = self.dropout(X)
        logger.debug(f"NewConvBlock layer1: {X.shape}")

        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))
        X = self.avgpool(X)
        X = self.dropout(X)
        logger.debug(f"NewConvBlock layer2: {X.shape}")

        X = self.conv2(X)
        X = F.gelu(self.batchnorm2(X))
        X = self.avgpool(X)
        X = self.dropout(X)
        logger.debug(f"NewConvBlock layer3: {X.shape}")

        return X


class NewConvClassifier(nn.Module):
    def __init__(
        self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 64
    ) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.seq_len = seq_len
        self.in_channels = in_channels

        # dim = [channel, 94, 94]
        self.spec = nn.Sequential(
            # T.Spectrogram(n_fft=186, win_length=32, hop_length=3, normalized=True),
            # T.AmplitudeToDB(),
            CWT(hop_length=2),
        )

        self.blocks = NewConvBlock(in_channels, hid_dim)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.feature_dim(), 1024),
            nn.GELU(),
            nn.Linear(1024, num_classes),
        )

    def feature_dim(self):
        with torch.no_grad():
            mock_tensor = torch.zeros(1, self.in_channels, self.seq_len)

            mock_tensor = self.spec(mock_tensor)
            mock_tensor = self.blocks(mock_tensor)

        return mock_tensor.shape[1] * mock_tensor.shape[2] * mock_tensor.shape[3]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.spec(X)  # (b, c, win_length+1, t/hop_length)
        logger.debug("spec: %s", X.shape)

        X = self.blocks(X)
        logger.debug("blocks: %s", X.shape)

        X = self.head(X)
        logger.debug("head: %s", X.shape)
        return X


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
    Args:
        seq_len (int): Number of data points included in each EEG chunk, i.e., T in the paper.
        num_electrodes (int): The number of electrodes, i.e., C in the paper.
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
        num_electrodes: int,
        hidden_dim: int = 128,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_1: int = 64,
        kernel_2: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.seq_len = seq_len
        self.F2 = F2

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
                kernel_size=(num_electrodes, 1),
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
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.seq_len)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is [n, 60, 151]. Here, n corresponds to the batch size, 60 corresponds to num_electrodes, and 151 corresponds to seq_len.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

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

    # model = EEGNet(num_classes=num_classes, seq_len=seq_len, num_electrodes=in_channels)

    # summary(
    #     model,
    #     input_size=(batch_size, in_channels, seq_len),
    #     col_names=["input_size", "output_size", "num_params", "mult_adds"],
    #     depth=3,
    #     row_settings=["var_names"],
    # )
