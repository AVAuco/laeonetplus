"""
PyTorch reimplementation of the LAEO-Net++ classifier head ("heads + head-map" variant,
i.e. `mj_genNetHeadsGeoCropMap(..., useself64=True, useMap=True, useGeom=False,
useFCrop=False, windowLenMap=10)` from the original Keras codebase).

The model is described by two small, framework-agnostic artifacts instead of a Keras
".json + .hdf5" pair:

  * a JSON config file with the architecture hyperparameters, produced by
    `tools/convert_keras_weights.py`;
  * an HDF5 weights file whose datasets are named exactly like the PyTorch
    `state_dict` keys (so loading needs nothing beyond `h5py` + `torch`).

See `tools/convert_keras_weights.py` for the one-time conversion from the legacy
Keras ".hdf5"/"weights.h5" files shipped in models/bestAVA and models/bestUCO.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import h5py
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LaeoNetConfig:
    window_len: int = 10
    window_len_map: int = 10
    head_size: int = 64
    map_size: int = 64
    dense_size: int = 32

    @classmethod
    def from_dict(cls, d: dict) -> "LaeoNetConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


class HeadBranchSelf64(nn.Module):
    """Shared 3D-CNN encoder applied independently to each head's crop sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(64, eps=1e-5)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(128, eps=1e-5)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(256, eps=1e-5)
        self.embedding = nn.Conv3d(256, 256, kernel_size=(2, 4, 4), stride=(1, 2, 2), padding=0)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, 64, 64)
        x = self.act(self.conv1(x))
        x = self.act(self.bn1(self.conv2(x)))
        x = self.act(self.bn2(self.conv3(x)))
        x = self.act(self.bn3(self.conv4(x)))
        x = self.embedding(x)
        return torch.flatten(x, 1)


class MapBranch3D(nn.Module):
    """3D-CNN encoder for the temporal sequence of pairwise head-position maps."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(16, 24, kernel_size=(3, 3, 3), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(24, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2))
        self.conv4 = nn.Conv3d(32, 12, kernel_size=(1, 6, 6), stride=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2.0, dim=1)


class LaeoNetPP(nn.Module):
    """LAEO-Net++ classifier: two head branches (shared weights) + map branch + MLP head."""

    def __init__(self, config: LaeoNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or LaeoNetConfig()
        self.head_branch = HeadBranchSelf64()
        self.map_branch = MapBranch3D()

        concat_size = 256 + 256 + 48  # two head codes + flattened map encoding
        self.fc1 = nn.Linear(concat_size, self.config.dense_size)
        self.output_layer = nn.Linear(self.config.dense_size, 2)

    def forward(self, head1: torch.Tensor, head2: torch.Tensor, headmap: torch.Tensor) -> torch.Tensor:
        """
        head1, head2: (B, 3, T, 64, 64) RGB head-crop sequences in [0, 1]
        headmap:      (B, 3, T, 64, 64) pairwise head-position map sequence
        returns: (B, 2) softmax probabilities [p_not_laeo, p_laeo]
        """
        h1 = self.head_branch(head1)
        h2 = self.head_branch(head2)
        m = self.map_branch(headmap)

        x = torch.cat([h1, h2, m], dim=1)
        x = F.relu(self.fc1(x))
        logits = self.output_layer(x)
        return F.softmax(logits, dim=1)


def load_laeo_model(json_path: str, weights_path: str, device: str = "cpu") -> LaeoNetPP:
    """Build a `LaeoNetPP` from a config JSON file and load its weights from an HDF5 file."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"LAEO model config not found: {json_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"LAEO model weights not found: {weights_path}")

    with open(json_path, "r") as f:
        raw_config = json.load(f)
    config = LaeoNetConfig.from_dict(raw_config)

    model = LaeoNetPP(config)
    state_dict = model.state_dict()

    with h5py.File(weights_path, "r") as h5f:
        # BatchNorm's `num_batches_tracked` is a PyTorch-only training bookkeeping
        # buffer with no Keras equivalent; it's irrelevant in eval mode, so it's the
        # only key allowed to be absent from the weights file.
        missing = [k for k in state_dict if k not in h5f and not k.endswith("num_batches_tracked")]
        if missing:
            raise KeyError(f"Weights file is missing tensors: {missing[:5]} (and {max(0, len(missing) - 5)} more)")
        for key in state_dict:
            if key not in h5f:
                continue
            array = h5f[key][()]
            state_dict[key] = torch.as_tensor(array, dtype=state_dict[key].dtype)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
