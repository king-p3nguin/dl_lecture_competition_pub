import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
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
        X = F.relu(self.batchnorm0(X))
        X = self.avgpool(X)
        X = self.dropout(X)
        logger.debug(f"NewConvBlock layer1: {X.shape}")

        X = self.conv1(X)
        X = F.relu(self.batchnorm1(X))
        X = self.avgpool(X)
        X = self.dropout(X)
        logger.debug(f"NewConvBlock layer2: {X.shape}")

        X = self.conv2(X)
        X = F.relu(self.batchnorm2(X))
        X = self.avgpool(X)
        X = self.dropout(X)
        logger.debug(f"NewConvBlock layer3: {X.shape}")

        return X


class NewConvClassifier(nn.Module):
    def __init__(
        self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128
    ) -> None:
        super().__init__()

        # dim = [channel, 94, 94]
        self.spec = T.Spectrogram(n_fft=186, win_length=32, hop_length=3)

        self.blocks = NewConvBlock(in_channels, hid_dim)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hid_dim * 8 * 8, num_classes),
        )

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
