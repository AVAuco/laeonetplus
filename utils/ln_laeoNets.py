"""
Functions to create a LAEO-Net++ model

Warning: right now, there are missing functions, not needed anyway.

Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2020
"""

__author__ = "Manuel J Marin-Jimenez"

import os
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model, Sequential


# Needed
def mj_l2normalize(x, axis=1):
    dnorm = K.l2_normalize(x, axis=axis)
    return dnorm


# This is the main LAEO-Net++ model
def mj_genNetHeadsGeoCropMap(windowLen=10, densesize_top=32, ndense_top=1,
                             dropoutval=0.5, batchnorm=False, nfilters_map=8,
                             usel2=True, initfileFM="", initfileHG="",
                             initfileHead="",
                             freezehead=False, freezecrop=False,
                             useFCrop=False, useGeom=False,
                             useMap=True, useself64=True,
                             windowLenMap=10):
    """
    Creates the main LAEO-Net++ model
    :param windowLen:
    :param densesize_top:
    :param ndense_top:
    :param dropoutval:
    :param batchnorm:
    :param nfilters_map:
    :param usel2:
    :param initfileFM:
    :param initfileHG:
    :param initfileHead: this is used *only* if other files are not provided
    :param freezehead:
    :param freezecrop:
    :param useFCrop:
    :param useGeom:
    :param useMap:
    :param useself64:
    :param windowLenMap:
    :return:
    """

    trainable = not freezehead

    # Some parameters
    head_input_shape = (windowLen, 64, 64, 3)
    the_input_shape = (windowLen, 128, 128, 3)
    if windowLenMap > 1:
        the_input_map_shape = (windowLenMap, 64, 64, 3)
    else:
        the_input_map_shape = (64, 64, 3)

    # Frame crop and heads map
    # ===========================================
    if useFCrop:
        frameBranch = mj_genNetFrameCrop(windowLen=windowLen, dropoutval=dropoutval, batchnorm=batchnorm, usel2=usel2)
        # create mapBranch
        mapBranch = mj_genNetMap(nfilters=nfilters_map, dropoutval=dropoutval, batchnorm=batchnorm, usel2=usel2)

        if initfileFM != "":
            basename = os.path.basename(initfileFM)
            nameparts = os.path.splitext(basename)
            if nameparts[1] == ".hdf5":  # is a whole model
                m0 = load_model(initfileFM)
                bF = m0.get_layer('sequential_1')
                bM = m0.get_layer('sequential_2')
                tmpfile_bF = "/tmp/mjmarin_00_w_bf.h5"
                bF.save_weights(tmpfile_bF)
                frameBranch.load_weights(tmpfile_bF, by_name=True)

                tmpfile_bM = "/tmp/mjmarin_00_w_bm.h5"
                bM.save_weights(tmpfile_bM)
                mapBranch.load_weights(tmpfile_bM, by_name=True)
            else:
                frameBranch.load_weights(initfileFM, by_name=True)
                mapBranch.load_weights(initfileFM, by_name=True)

            if freezecrop:
                trainable = False
            else:
                trainable = True
            if not trainable:
                for layer in frameBranch.layers:
                    layer.trainable = False
                for layer in mapBranch.layers:
                    layer.trainable = False

        frameinput = layers.Input(shape=the_input_shape, name='frameinput')
        mapinput = layers.Input(shape=the_input_map_shape, name='mapinput')

        frame_encoding = frameBranch(frameinput)
        map_encoding = mapBranch(mapinput)
    else:
        # create mapBranch
        if windowLenMap > 1:
            mapBranch = mj_genNetMap3D(windowLen=windowLenMap, dropoutval=dropoutval, batchnorm=batchnorm, usel2=usel2)
        else:
            mapBranch = mj_genNetMap(nfilters=nfilters_map, dropoutval=dropoutval, batchnorm=batchnorm, usel2=usel2)

        if initfileFM != "":
            basename = os.path.basename(initfileFM)
            nameparts = os.path.splitext(basename)
            if nameparts[1] == ".hdf5":  # is a whole model
                m0 = load_model(initfileFM)

                bM = m0.get_layer('sequential_2')

                tmpfile_bM = "/tmp/mjmarin_00_w_bm.h5"
                bM.save_weights(tmpfile_bM)
                mapBranch.load_weights(tmpfile_bM, by_name=True)
            else:
                mapBranch.load_weights(initfileFM, by_name=True)

            if freezecrop:
                trainable = False
            else:
                trainable = True
            if not trainable:
                # for layer in frameBranch.layers:
                #     layer.trainable = False
                for layer in mapBranch.layers:
                    layer.trainable = False

        # frameinput = layers.Input(shape=the_input_shape, name='frameinput')
        mapinput = layers.Input(shape=the_input_map_shape, name='mapinput')

        # frame_encoding = frameBranch(frameinput)
        map_encoding = mapBranch(mapinput)

    # Heads and geometry
    # ===========================================
    if useself64:
        if initfileHead != "":
            headBranch = load_model(initfileHead)

            if freezehead:
                for ly in headBranch.layers:
                    ly.trainable = False

        else:
            # print("ERROR: requested using self-supervised head branch, but no file has been provided. Check parameter -initfileHead-")
            headBranch = mj_genNetHeadChannel3Dself64Inflated(winlen=windowLen)

    else:
        headBranch = mj_genNetHeadChannel(windowLen, dropoutval, batchnorm, usel2)

    if useGeom:
        # Extra branch: geometry
        geoinput = layers.Input(shape=(3,), name='geoinput')
        geoBranch = mj_genNetGeoBranch(usel2=usel2)
        geoenc = geoBranch(geoinput)

        if initfileHG != "":
            basename = os.path.basename(initfileHG)
            nameparts = os.path.splitext(basename)
            if nameparts[1] == ".hdf5":  # is a whole model
                m0 = load_model(initfileHG)
                bH = m0.get_layer('sequential_3')
                tmpfile_bH = "/tmp/mjmarin_00_w_h.h5"
                bH.save_weights(tmpfile_bH)
                headBranch.load_weights(tmpfile_bH, by_name=True)

                tmpfile_bG = "/tmp/mjmarin_00_w_g.h5"
                m0.save_weights(tmpfile_bG)
                geoBranch.load_weights(tmpfile_bG, by_name=True)
            else:
                geoBranch.load_weights(initfileHG, by_name=True)
                headBranch.load_weights(initfileHG, by_name=True)

            if freezehead:
                trainable = False
            else:
                trainable = True
            if not trainable:
                for layer in headBranch.layers:
                    layer.trainable = False
                for layer in geoBranch.layers:
                    layer.trainable = False

    else:
        if initfileHG != "":
            basename = os.path.basename(initfileHG)
            nameparts = os.path.splitext(basename)
            if nameparts[1] == ".hdf5":  # is a whole model
                m0 = load_model(initfileHG)
                bH = m0.get_layer('sequential_3')
                tmpfile_bH = "/tmp/mjmarin_00_w_h.h5"
                bH.save_weights(tmpfile_bH)
                headBranch.load_weights(tmpfile_bH, by_name=True)

            else:
                headBranch.load_weights(initfileHG, by_name=True)

            if freezehead:
                trainable = False
            else:
                trainable = True
            if not trainable:
                for layer in headBranch.layers:
                    layer.trainable = False
                # for layer in geoBranch.layers:
                #     layer.trainable = False
        elif initfileHead != "" and not useself64:
            basename = os.path.basename(initfileHead)
            nameparts = os.path.splitext(basename)
            if nameparts[1] == ".hdf5":  # is a whole model
                m0 = load_model(initfileHead, custom_objects={'mj_smoothL1sigmaPsign': losses.mean_squared_error})
                tmpfile = "/tmp/mjmarin_00_w.h5"
                m0.save_weights(tmpfile)
                headBranch.load_weights(tmpfile, by_name=True)
            else:
                headBranch.load_weights(initfileHead, by_name=True)

            if not trainable:
                for layer in headBranch.layers:
                    layer.trainable = False

    rgbinput1 = layers.Input(shape=head_input_shape, name='rgbinput1')
    rgbinput2 = layers.Input(shape=head_input_shape, name='rgbinput2')

    h1 = headBranch(rgbinput1)
    h2 = headBranch(rgbinput2)

    if useself64:
        h1 = layers.Flatten(name="hcode01")(h1)
        h2 = layers.Flatten(name="hcode02")(h2)

    # Concatenate branches
    # ===========================================
    the_concats = [h1, h2]
    the_inputs = [rgbinput1, rgbinput2]

    if useGeom:
        the_concats.append(geoenc)
        the_inputs.append(geoinput)

    if useFCrop:
        the_concats.append(frame_encoding)
        the_inputs.append(frameinput)

    if useMap:
        the_concats.append(map_encoding)
        the_inputs.append(mapinput)

    concat = layers.concatenate(the_concats, name="concat_encods")

    # Add intermediate layers
    x = layers.Dense(densesize_top, activation='relu', name='fc_1')(concat)
    if ndense_top > 1:
        for ndix in range(1, ndense_top):
            x = layers.Dropout(dropoutval)(x)
            nunits = int(densesize_top / (2 ** ndix))
            x = layers.Dense(nunits, activation='relu', name='fc_{:d}'.format(ndix+1))(x)

    drp2 = layers.Dropout(dropoutval, name="top_dropout")(x)
    laeooutput = layers.Dense(2, activation='softmax', name='output_laeo')(drp2)

    model = Model(inputs=the_inputs, outputs=[laeooutput], name="LAEONetPP")

    return model


def mj_genNetMap3D(windowLen=10, dropoutval=0.5, batchnorm=False, usel2=False):
    # Some parameters
    the_input_shape = (windowLen, 64, 64, 3)

    mapBranch = Sequential(name="mapBranch")

    mapBranch.add(layers.Conv3D(16, (3,5,5), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                       activation='relu', input_shape=the_input_shape, name="mconv3d_1"))

    mapBranch.add(layers.Conv3D(24, (3,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                       activation='relu', name="mconv3d_2"))

    if windowLen >= 10:

        if batchnorm:
            mapBranch.add(layers.BatchNormalization())

        mapBranch.add(layers.Conv3D(32, (3,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                           activation='relu', name="mconv3d_3"))

        if batchnorm:
            mapBranch.add(layers.BatchNormalization())

        mapBranch.add(layers.Conv3D(12, (1,6,6), strides=(1, 1, 1), padding='valid', data_format='channels_last',
                           activation='relu', name="mconv3d_4"))
    else:
        mapBranch.add(layers.Conv3D(32, (1,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                           activation='relu', name="mconv3d_3"))

        mapBranch.add(layers.Conv3D(12, (1,6,6), strides=(1, 1, 1), padding='valid', data_format='channels_last',
                           activation='relu', name="mconv3d_4"))

    mapBranch.add(layers.Dropout(dropoutval, name="mtop_dropout"))

    mapBranch.add(layers.Flatten(name="mflat"))

    if usel2:
        from tensorflow.keras.layers import Lambda

        # L2 normalization
        mapBranch.add(Lambda(mj_l2normalize, arguments={'axis': 1}, name="ml2norm"))

    return mapBranch


def mj_genNetHeadChannel3Dself64Inflated(winlen=10):
    the_input_shape = (winlen, 64, 64, 3)

    headBranch = Sequential(name="headBranchSelf64")

    headBranch.add(layers.ZeroPadding3D(padding=(0,1,1), input_shape=the_input_shape,  name="zp3d_1"))
    headBranch.add(layers.Conv3D(32, (3,4,4), strides=(1,2,2), padding='valid', data_format='channels_last',
                   activation='linear', name="hconv3d_1"))
    headBranch.add(layers.LeakyReLU(alpha=0.2, name="lkrelu_1"))

    headBranch.add(layers.ZeroPadding3D(padding=(0, 1, 1), name="zp3d_2"))
    headBranch.add(layers.Conv3D(64, (3, 4, 4), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                 activation='linear', name="hconv3d_2"))
    headBranch.add(layers.BatchNormalization(momentum=0.99, epsilon=1e-05, name='bn_1'))
    headBranch.add(layers.LeakyReLU(alpha=0.2, name="lkrelu_2"))

    if winlen >= 10:
        headBranch.add(layers.ZeroPadding3D(padding=(0, 1, 1), name="zp3d_3"))
        headBranch.add(layers.Conv3D(128, (3, 4, 4), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                     activation='linear', name="hconv3d_3"))
        headBranch.add(layers.BatchNormalization(momentum=0.99, epsilon=1e-05, name='bn_2'))
        headBranch.add(layers.LeakyReLU(alpha=0.2, name="lkrelu_3"))

        headBranch.add(layers.ZeroPadding3D(padding=(0, 1, 1), name="zp3d_4"))
        headBranch.add(layers.Conv3D(256, (3, 4, 4), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                     activation='linear', name="hconv3d_4"))
        headBranch.add(layers.BatchNormalization(momentum=0.99, epsilon=1e-05, name='bn_3'))
        headBranch.add(layers.LeakyReLU(alpha=0.2, name="lkrelu_4"))

        headBranch.add(layers.Conv3D(256, (2, 4, 4), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                     activation='linear', name="embedding"))
    else:
        headBranch.add(layers.ZeroPadding3D(padding=(0, 1, 1), name="zp3d_3"))
        headBranch.add(layers.Conv3D(128, (1, 4, 4), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                     activation='linear', name="hconv3d_3"))
        headBranch.add(layers.BatchNormalization(momentum=0.99, epsilon=1e-05, name='bn_2'))
        headBranch.add(layers.LeakyReLU(alpha=0.2, name="lkrelu_3"))

        headBranch.add(layers.ZeroPadding3D(padding=(0, 1, 1), name="zp3d_4"))
        headBranch.add(layers.Conv3D(256, (1, 4, 4), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                     activation='linear', name="hconv3d_4"))
        headBranch.add(layers.BatchNormalization(momentum=0.99, epsilon=1e-05, name='bn_3'))
        headBranch.add(layers.LeakyReLU(alpha=0.2, name="lkrelu_4"))

        headBranch.add(layers.Conv3D(256, (1, 4, 4), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                                     activation='linear', name="embedding"))

    return headBranch