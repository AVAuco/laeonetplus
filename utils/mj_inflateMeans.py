# (c) MJMJ/2020

import os.path as osp
# from os.path import expanduser
# homedir = expanduser("~")

import numpy as np


def mj_inflateMeanMat(data, winlen_new=10):
    """
    Expands dimensions of input matrix
    :param data: numpy matrix
    :param winlen_new: integer used to expand the first dimension of matrix 'data'
    :return: extended matrix
    """

    side = data.shape[1]
    wlen = int(data.shape[0] / side)

    if wlen >= winlen_new:
        print("WARN: Nothing to do!")
        return data

    data_new = np.zeros((winlen_new * side, side, data.shape[2]), data.dtype)

    df = winlen_new - wlen
    df2 = int(df / 2)

    for t in range(0, df2):
        data_new[t * side:(t + 1) * side, :, :] = data[0:side, ]

    for t in range(0, wlen):
        data_new[(t + df2) * side:(t + df2 + 1) * side, :, :] = data[t * side:(t + 1) * side, ]

    for t in range(0, df2 + 1):
        data_new[(df2 + wlen + t) * side:(df2 + wlen + t + 1) * side, :, :] = data[(wlen - 1) * side:wlen * (side), ]

    return data_new

