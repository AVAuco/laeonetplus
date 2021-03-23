"""
(c) MJMJ/2018
"""

import os
import copy
import numpy as np
from tensorflow.keras import layers


def mj_findLatestFileModel(inputdir, pattern, epoch_max=1000):
    '''
    Searchs for check-points during training
    :param inputdir: path
    :param pattern: string compatible with format()
    :return: path to the best file, if any, "" otherwise
    '''

    if epoch_max < 0:
        maxepochs = 1000
    else:
        maxepochs = epoch_max

    bestfile = ""

    for epoch in range(1, maxepochs+1):
        modelname = os.path.join(inputdir, pattern.format(epoch))
        if os.path.isfile(modelname):
            bestfile = copy.deepcopy(modelname)


    return bestfile


def mj_computeMeanImageSample(samples):
    '''

    :param samples: tensor [nsamples, nrows, ncols, 3]
    :return:
    '''

    nsamples = samples.shape[0]
    mean = np.zeros((samples.shape[1], samples.shape[2], samples.shape[3]))
    ninv = 1.0/nsamples
    for i in range(0,nsamples):
        for c in range(0,3):
           mean[:,:,c] += (samples[i,:,:,c])*ninv

    return mean


def mj_inflateFilter(filters_in, new_depth=3):

    f_shape = filters_in.shape

    new_weights = np.zeros((new_depth,) + f_shape)
    for ix in range(0, f_shape[3]):
        for d_ in range(0, new_depth):
            new_weights[d_, :, :, :, ix] = filters_in[:, :, :, ix]

    return new_weights

def mj_inflateConvLayer(layer_in, new_depth=3, new_name=""):
    '''
    A new layer Conv3D is created
    :param layer_in:
    :param new_depth:
    :param new_name:
    :return:
    '''

    w = layer_in.get_weights()
    filters = w[0]
    f_shape = filters.shape
    f_strides = layer_in.strides

    if new_name == "":
        new_name = layer_in.name

    # Create new layer: 2D --> 3D
    new_layer = layers.Conv3D(f_shape[3], (new_depth, f_shape[0], f_shape[1]), strides=(1, f_strides[0], f_strides[1]), padding='valid',
                  data_format='channels_last', activation='relu', name=new_name)  # input_shape=the_input_shape,

    new_filters = mj_inflateFilter(filters, new_depth=new_depth)

    biases = w[1]
    new_biases = biases  # No need to inflate
    new_weights = [new_filters, new_biases]
    new_layer.set_weights(new_weights)

    return new_layer

def mj_inflateConvWeights(w, new_depth=3, toBGR=False):
    '''
    No new layer is created, just the list of weights (filters and biases)
    :param w:
    :param new_depth:
    :return:
    '''

    filters = w[0]

    if toBGR:
        filters = filters[:, :, ::-1, ]

    new_filters = mj_inflateFilter(filters, new_depth=new_depth)

    biases = w[1]
    new_biases = biases  # No need to inflate

    new_weights = [new_filters, new_biases]

    return new_weights