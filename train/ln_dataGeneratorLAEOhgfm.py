"""
Based on the following example:
   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
(c) MJMJ/2018
"""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import tensorflow.keras
from   tensorflow.keras.preprocessing.image import ImageDataGenerator
#import multiprocessing

#import model_utils as mu
import random
import copy
from time import time

from mj_dataHelper import mj_gatherSampleFromId, mj_gatherSampleFromListIds, mj_gatherFrameCropMapSampleFromListIds
from mj_laeoUtils import mj_genNegLAEOwithGeom


# ---------------------------------------------------------------------------------------------------

class DataGeneratorLAEOhgfm(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, allSamples, laeoIdxs, batch_size=32, dim=(10,64,128),
                 n_channels=3, n_classes=2, shuffle=True, augmentation=True,
                 withHMap=True,
                 withFCrop=True, withGeom=True,  isTest=False, splitHeads=False, augmentation_x=1,
                 meanSampleH=[0.0], meanSampleFM=[0.0], winlenMap=1):
        'Initialization'
        self.dim = dim
        self.time_dim = dim[0]
        self.dim_map = (self.time_dim,64,64,3)
        self.winlenMap = winlenMap
        self.dim_crop = (self.time_dim,128,128,3)
        self.batch_size = batch_size

        self.list_IDs = copy.deepcopy(list_IDs)
        if augmentation:
            for aix in range(0,augmentation_x):
                self.list_IDs.extend(-np.array(list_IDs))  # Minus means to be perturbated
        if isTest == False:
            np.random.shuffle(self.list_IDs)   # Done always!
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.allSamples = allSamples
        self.laeoIdxs = laeoIdxs
        self.withFCrop = withFCrop
        self.withGeom = withGeom
        self.withMap = withHMap
        self.augmenProb = 0.2     # Was 0.1
        self.isTest = isTest
        self.splitHeads = splitHeads
        self.meanSampleH = meanSampleH   # To be subtracted from samples
        self.meanSampleFM = meanSampleFM  # To be subtracted from samples

        # Needed for initialization
        self.img_gen = ImageDataGenerator(width_shift_range=[-2,0,2], height_shift_range=[-2,0,2],
                                          brightness_range=[0.5,1.4], zoom_range=0.04,
                                          channel_shift_range=0.35, horizontal_flip=True)
        self.__splitInPosAndNegs()
        self.on_epoch_end()

        self.classes = np.zeros((len(self.indexes),2))
        #self.kk = -1

    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def getitemwithinfo(self, index):
        X, y = self.__getitem__(index)

        info = [self.allSamples[k] for k in self.list_IDs_temp]

        return X, y, info

    def __getitem__(self, index):
        'Generate one batch of data'
        #Indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if len(indexes) < 1:
           rnd = random.randint(0,len(self.indexes)-self.batch_size)
           indexes = self.indexes[rnd:rnd+self.batch_size]

        # Find list of ids
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        # Store classes
#        self.classes[index*self.batch_size:(index+1)*self.batch_size,] = copy.deepcopy(y)
        self.kk = index
        self.list_IDs_temp = list_IDs_temp

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.list_IDs))
        #
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
        if self.isTest:
            self.indexes = np.arange(len(self.list_IDs))
        else:
            if self.shuffle:
                np.random.shuffle(self.pos)
                np.random.shuffle(self.neg)

            # Balanced batch
            npos = len(self.pos)
            nneg = len(self.neg)
            df = nneg-npos  # Difference
            # if nneg > npos:
            #     self.neg = self.neg[0:npos]
            # else:
            #     self.pos = self.pos[0:nneg]
            tmppos = copy.deepcopy(self.pos)
            tmpneg = copy.deepcopy(self.neg)
            if nneg > npos:
                tmppos.extend(self.pos[0:abs(df)])
            else:
                tmpneg.extend(self.neg[0:abs(df)])
            halfbatch =  int(self.batch_size/2)
            nbatches = int(np.floor((len(tmppos)+len(tmpneg))/self.batch_size))
            indexes = []
            for i in range(0,nbatches):
                indexes.extend(tmppos[i*halfbatch:(i+1)*halfbatch] )
                indexes.extend(tmpneg[i* halfbatch:(i+1)* halfbatch])

            self.indexes = indexes

        # if hasattr(self, 'classes'):
        #     # Compute number of positive and negative samples
        #     nneg = self.classes[:,0].sum() / self.classes.shape[0]
        #     npos = self.classes[:, 1].sum()/ self.classes.shape[0]
        #     print("* INFO: number of samples used: neg={} vs pos={}".format(nneg, npos))

    def __splitInPosAndNegs(self):
        self.pos = []
        self.neg = []

        #for ix in self.list_IDs:
        for ix_ in range(0,len(self.list_IDs)):
            ix = abs(self.list_IDs[ix_])
            if self.allSamples[ix][3] == 0:
                self.neg.append(ix_)
            elif self.allSamples[ix][3] == 1:
                self.pos.append(ix_)


    def __gatherSampleFromId(self, trix_):
        pair, label, geom = mj_gatherSampleFromId(trix_, self.laeoIdxs,
         self.allSamples, self.dim,
         self.meanSample,
         self.augmentation,
         self.img_gen)

        return pair, label, geom


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
#        X = np.random.random((self.batch_size, *self.dim, self.n_channels))
#        y = np.zeros((self.batch_size, 1, 1, 1, 2), dtype=int)
#        G = np.zeros((self.batch_size, 3))
        yG = np.zeros((len(list_IDs_temp), 2), dtype=int)

#        labels = [0] * self.batch_size

        # TODO: make sure that same data augmentation strategy is used for same sample

        transformation = self.img_gen.get_random_transform(self.dim[1:2])

        # Gather the different inputs
        Xheads, labels_heads, G_heads = mj_gatherSampleFromListIds(list_IDs_temp,self.laeoIdxs,
                               self.allSamples, self.dim,
                              self.meanSampleH,
                              self.augmentation,
                              self.img_gen, transform=transformation)


        if self.withMap or self.withFCrop:
            Xfm, Mfm, labels_fm = mj_gatherFrameCropMapSampleFromListIds(list_IDs_temp, self.laeoIdxs,
                               self.allSamples, self.dim_crop, self.dim_map,
                              self.meanSampleFM,
                              self.augmentation,
                              self.img_gen, transform=transformation, winlenMap=self.winlenMap)
        else:
            labels_fm = labels_heads


        for i in range(0, len(list_IDs_temp)):
            if labels_fm[i] != 9:
                #y[i, 0, 0, 0, int(labels_fm[i])] = 1
                yG[i,int(labels_fm[i])] = 1
            else:
                #y[i, 0, 0, 0,] = 1
                yG[i,] = 1

        # splitHeads:
        imgh = self.dim[1]
        Xs = []
        for ix in range(0,2):
            X_ = Xheads[:,:,:,ix*imgh:(ix+1)*imgh,:]
            if self.dim[0] == 1:
                X_ = np.squeeze(X_)
            Xs.append(X_)

        if self.withGeom:
            Xs.append(G_heads)

        if self.withFCrop:
            Xs.append(Xfm)

        if self.withMap:
            Xs.append(Mfm)

        return Xs, yG
