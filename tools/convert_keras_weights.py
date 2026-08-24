"""
One-time converter: legacy Keras LAEO-Net++ weights (models/best{AVA,UCO}/*_weights.h5)
-> the new framework-agnostic format consumed by `laeo_pipeline.laeo_model.load_laeo_model`
(a JSON config file + an HDF5 file whose datasets are named like PyTorch state_dict keys).

Only `h5py` and `numpy` are required -- no TensorFlow/Keras installation is needed,
since Keras ".h5" weight files are plain HDF5 files with a conventional group layout
that we can read directly.

Usage:
    python tools/convert_keras_weights.py \\
        --keras_weights models/bestAVA/model-hmaps-trava_weights.h5 \\
        --out_json models_pt/laeonet_ava.json \\
        --out_weights models_pt/laeonet_ava_weights.h5
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

# (keras dataset path, pytorch state_dict key, transpose axes or None)
CONV_TRANSPOSE = (4, 3, 0, 1, 2)  # keras (kd,kh,kw,in,out) -> torch (out,in,kd,kh,kw)
DENSE_TRANSPOSE = (1, 0)  # keras (in,out) -> torch (out,in)

WEIGHT_MAP = [
    # -- shared head branch ("sequential_1" in the Keras model) --
    ("sequential_1/hconv3d_1/kernel:0", "head_branch.conv1.weight", CONV_TRANSPOSE),
    ("sequential_1/hconv3d_1/bias:0", "head_branch.conv1.bias", None),
    ("sequential_1/hconv3d_2/kernel:0", "head_branch.conv2.weight", CONV_TRANSPOSE),
    ("sequential_1/hconv3d_2/bias:0", "head_branch.conv2.bias", None),
    ("sequential_1/bn_1/gamma:0", "head_branch.bn1.weight", None),
    ("sequential_1/bn_1/beta:0", "head_branch.bn1.bias", None),
    ("sequential_1/bn_1/moving_mean:0", "head_branch.bn1.running_mean", None),
    ("sequential_1/bn_1/moving_variance:0", "head_branch.bn1.running_var", None),
    ("sequential_1/hconv3d_3/kernel:0", "head_branch.conv3.weight", CONV_TRANSPOSE),
    ("sequential_1/hconv3d_3/bias:0", "head_branch.conv3.bias", None),
    ("sequential_1/bn_2/gamma:0", "head_branch.bn2.weight", None),
    ("sequential_1/bn_2/beta:0", "head_branch.bn2.bias", None),
    ("sequential_1/bn_2/moving_mean:0", "head_branch.bn2.running_mean", None),
    ("sequential_1/bn_2/moving_variance:0", "head_branch.bn2.running_var", None),
    ("sequential_1/hconv3d_4/kernel:0", "head_branch.conv4.weight", CONV_TRANSPOSE),
    ("sequential_1/hconv3d_4/bias:0", "head_branch.conv4.bias", None),
    ("sequential_1/bn_3/gamma:0", "head_branch.bn3.weight", None),
    ("sequential_1/bn_3/beta:0", "head_branch.bn3.bias", None),
    ("sequential_1/bn_3/moving_mean:0", "head_branch.bn3.running_mean", None),
    ("sequential_1/bn_3/moving_variance:0", "head_branch.bn3.running_var", None),
    ("sequential_1/embedding/kernel:0", "head_branch.embedding.weight", CONV_TRANSPOSE),
    ("sequential_1/embedding/bias:0", "head_branch.embedding.bias", None),
    # -- map branch ("mapBranch" in the Keras model) --
    ("mapBranch/mconv3d_1/kernel:0", "map_branch.conv1.weight", CONV_TRANSPOSE),
    ("mapBranch/mconv3d_1/bias:0", "map_branch.conv1.bias", None),
    ("mapBranch/mconv3d_2/kernel:0", "map_branch.conv2.weight", CONV_TRANSPOSE),
    ("mapBranch/mconv3d_2/bias:0", "map_branch.conv2.bias", None),
    ("mapBranch/mconv3d_3/kernel:0", "map_branch.conv3.weight", CONV_TRANSPOSE),
    ("mapBranch/mconv3d_3/bias:0", "map_branch.conv3.bias", None),
    ("mapBranch/mconv3d_4/kernel:0", "map_branch.conv4.weight", CONV_TRANSPOSE),
    ("mapBranch/mconv3d_4/bias:0", "map_branch.conv4.bias", None),
    # -- classifier head --
    ("fc_1/fc_1/kernel:0", "fc1.weight", DENSE_TRANSPOSE),
    ("fc_1/fc_1/bias:0", "fc1.bias", None),
    ("output_laeo/output_laeo/kernel:0", "output_layer.weight", DENSE_TRANSPOSE),
    ("output_laeo/output_laeo/bias:0", "output_layer.bias", None),
]


def convert(keras_weights_path: str, out_json: str, out_weights: str) -> None:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_weights) or ".", exist_ok=True)

    with h5py.File(keras_weights_path, "r") as src, h5py.File(out_weights, "w") as dst:
        for keras_key, torch_key, transpose in WEIGHT_MAP:
            if keras_key not in src:
                raise KeyError(f"Expected dataset '{keras_key}' not found in {keras_weights_path}")
            array = src[keras_key][()].astype(np.float32)
            if transpose is not None:
                array = np.transpose(array, transpose).copy()
            dst.create_dataset(torch_key, data=array)

    config = {
        "window_len": 10,
        "window_len_map": 10,
        "head_size": 64,
        "map_size": 64,
        "dense_size": 32,
    }
    with open(out_json, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {out_json} and {out_weights}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras_weights", required=True, help="Path to the legacy Keras *_weights.h5 file")
    parser.add_argument("--out_json", required=True, help="Output path for the new config JSON file")
    parser.add_argument("--out_weights", required=True, help="Output path for the new PyTorch-ready HDF5 weights")
    args = parser.parse_args()

    convert(args.keras_weights, args.out_json, args.out_weights)


if __name__ == "__main__":
    main()
