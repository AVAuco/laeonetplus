"""
Based on the following example:
   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

This version is for AVA-Google dataset

(c) MJMJ/2019
"""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from os.path import expanduser
homedir = expanduser("~")

import numpy as np
import tensorflow.keras
from   tensorflow.keras.preprocessing.image import ImageDataGenerator
#import multiprocessing

#import model_utils as mu
import random
import copy
from time import time
import os, tarfile

from mj_dataHelper import mj_gatherSampleFromId, mj_gatherSampleFromListIds, mj_gatherFrameCropMapSampleFromListIds
# from mj_laeoUtils import mj_genNegLAEOwithGeom
# from mj_avagoogleImages import mj_getImagePairSeqFromTracks
from ln_avagoogleConfig import AvaGoogledb
from ln_avagoogleImages import mj_getImagePairSeqFromPklTrack, mj_getImagePairSeqFromPklTrackTar

# ---------------------------------------------------------------------------------------------------

class DataGeneratorAVALAEO(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, avalaeo_annots, list_IDs, allSamples, batch_size=32, dim=(10,64,128),
                 n_channels=3, n_classes=2, shuffle=True, augmentation=True,
                 withHMap=True,
                 withFCrop=True, withGeom=True,  isTest=False, splitHeads=False, augmentation_x=1,
                 meanSampleH=[0.0], meanSampleFM=[0.0], case_wanted = "train", tarpath="", winlenMap=1):
        'Initialization'
        self.avalaeo_annots = avalaeo_annots
        self.dim = dim
        self.time_dim = dim[0]
        self.dim_map = (64,64,3)
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
        #self.laeoIdxs = laeoIdxs
        self.withFCrop = withFCrop
        self.withGeom = withGeom
        self.withMap = withHMap
        self.augmenProb = 0.2     # Was 0.1
        self.isTest = isTest
        self.splitHeads = splitHeads
        self.meanSampleH = meanSampleH   # To be subtracted from samples
        self.meanSampleFM = meanSampleFM  # To be subtracted from samples
        if isinstance(meanSampleFM, dict):
            idx_5 = 5
            self.meanMap5 = meanSampleFM["meanmap"][idx_5*self.dim_map[1]:(idx_5+1)*self.dim_map[1],]
        else:
            self.meanMap5 = [0.0]

        # Paths
        self.avadb = AvaGoogledb(case_wanted=case_wanted, basedir=homedir + "/experiments/ava",
                            framesdirbase=homedir + "/databases/ava_google/Frames")

        # Data in tar file
        if tarpath == "":
            outdir = os.path.join(homedir, "experiments/ava/preprocdata/w10/", case_wanted)
            tarname = os.path.join(outdir, "allsamples.tar")
        else:
            tarname = tarpath

        if not os.path.exists(tarname):
            print("ERROR: cannot find tar file: {}".format(tarname))
            exit(-1)
        self.tar = tarfile.open(tarname, 'r')
        if self.tar is None:
            print("Error reading tar file!!!")
            exit(-1)

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
            if self.allSamples[ix][2] == 0:
                self.neg.append(ix_)
            elif self.allSamples[ix][2] == 1:
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
        G_heads = []
        Xfm = []


#        labels = [0] * self.batch_size


        # Gather the different inputs
        # Xheads, labels_heads, G_heads = mj_gatherSampleFromListIds(list_IDs_temp,self.laeoIdxs,
        #                        self.allSamples, self.dim,
        #                       self.meanSampleH,
        #                       self.augmentation,
        #                       self.img_gen, transform=transformation)
        #
        #
        # if self.withMap or self.withFCrop:
        #     Xfm, Mfm, labels_fm = mj_gatherFrameCropMapSampleFromListIds(list_IDs_temp, self.laeoIdxs,
        #                        self.allSamples, self.dim_crop, self.dim_map,
        #                       self.meanSampleFM,
        #                       self.augmentation,
        #                       self.img_gen, transform=transformation)
        # else:
        #     labels_fm = labels_heads

        # TODO: assign correct labels and get samples
        Xheads01 = np.zeros((self.batch_size, self.time_dim, self.dim[1], self.dim[1], self.n_channels))
        Xheads02 = np.zeros((self.batch_size, self.time_dim, self.dim[1], self.dim[1], self.n_channels))
        if self.winlenMap > 1:
            Mfm = np.zeros((self.batch_size, self.winlenMap, self.dim_map[0], self.dim_map[1], self.dim_map[2]))
        else:
            Mfm = np.zeros((self.batch_size, self.dim_map[0], self.dim_map[1],self.dim_map[2] ))

        for i in range(0, len(list_IDs_temp)):
            ix = abs(list_IDs_temp[i])
            vidname = self.allSamples[ix][0]
            pairkey = self.allSamples[ix][1]
            laeo_lab = self.allSamples[ix][2]
            tracks = self.avalaeo_annots.get_tracks(vidname, pairkey)

            if False:
                sampleL, sampleR, G, M = mj_getImagePairSeqFromPklTrack(tracks, self.avadb.frames, vidname, targetsize=(64, 64),
                                                                    winlen=self.time_dim,
                                                                    mean_head=self.meanSampleH, with_maps= self.withMap, mean_map=self.meanMap5,
                                                                    lbbs_others=[])

            sampleL, sampleR, G, M = mj_getImagePairSeqFromPklTrackTar(self.tar, tracks, self.avadb.frames, vidname, pairkey, targetsize=(64, 64),
                                                  winlen=self.time_dim,
                                                  mean_head=self.meanSampleH, with_maps=self.withMap, mean_map=self.meanMap5, lbbs_others=[],
                                                                       winlenMap=self.winlenMap)

            # Augmentation?
            if self.augmentation and list_IDs_temp[i] < 0:
                # Transform head pair
                normalize = sampleL.max() <= 1.0
                transformation = self.img_gen.get_random_transform((sampleL.shape[1:2]))
                if transformation["flip_horizontal"] == 1:
                    flipMap = True
                else:
                    flipMap = False

                for wix in range(0, sampleL.shape[0]):
                    sampleL[wix,] = self.img_gen.apply_transform(sampleL[wix,]+self.meanSampleH, transformation)/255.0
                    sampleL[wix,] -= self.meanSampleH # Normalization is needed after transformation!

                    sampleR[wix,] = self.img_gen.apply_transform(sampleR[wix,]+self.meanSampleH, transformation)/255.0
                    sampleR[wix,] -= self.meanSampleH  # Normalization is needed after transformation!

                # Transform map
                if flipMap:
                    if self.winlenMap > 1:
                        #map_tmp = np.zeros(M.shape)
                        for mix in range(0, self.winlenMap):
                            map_tmp_ = np.fliplr(M[mix,])
                            M[mix,] = map_tmp_[:, :, [1, 0, 2]]  # Swap channels, as left head and right head have been flipped as well
                    else:
                        map_tmp = np.fliplr(M)
                        M = map_tmp[:, :, [1, 0, 2]]  # Swap channels, as left head and right head have been flipped as well

                # Transform geometry
                if flipMap:
                    G[0] = -G[0]  # Difference is opposite
                    G[2] = 1.0 / G[2]  # Ratio of scales is inverted

            Xheads01[i,] = sampleL
            Xheads02[i,] = sampleR
            time_half = int(np.floor(self.time_dim/2))
            if self.winlenMap > 1:
                Mfm[i,] = M
            else:
                Mfm[i, ] = M[time_half,] # / 255.0 # TODO check if normalization is needed
            if laeo_lab != 9:
                yG[i, int(laeo_lab)] = 1
            else:
                yG[i,] = 1   # We shouldn't reach this point, but just in case


                # for ix in list_IDs_temp:
        #     this_label = 0
        #     labels_fm.append(this_label)
        #     # Variables to be filled
        #     tracks = []
        #     trix1 = 0
        #     trix2 = 1
        #     init_frame = 23
        #     winlen = self.time_dim
        #     framesdir = self.avadb.frames
        #     targetsize = (self.dim[1], self.dim[1])
        #     sampleL, sampleR, G, M = mj_getImagePairSeqFromTracks(tracks, (trix1, trix2),
        #                                                       init_frame, winlen, framesdir,
        #                                                       targetsize, self.meanSampleH, with_maps=self.withMap,
        #                                                       mean_map=self.meanMap5, strict_mode=True)

        # for i in range(0, len(list_IDs_temp)):
        #     if labels_fm[i] != 9:
        #         #y[i, 0, 0, 0, int(labels_fm[i])] = 1
        #         yG[i,int(labels_fm[i])] = 1
        #     else:
        #         #y[i, 0, 0, 0,] = 1
        #         yG[i,] = 1

        # splitHeads:
        # imgh = self.dim[1]
        Xs = []
        # for ix in range(0,2):
        #     X_ = Xheads[:,:,:,ix*imgh:(ix+1)*imgh,:]
        #     if self.dim[0] == 1:
        #         X_ = np.squeeze(X_)
        #     Xs.append(X_)
        Xs.append(Xheads01)
        Xs.append(Xheads02)

        if self.withGeom:
            Xs.append(G_heads)

        if self.withFCrop:
            Xs.append(Xfm)

        if self.withMap:
            Xs.append(Mfm)

        return Xs, yG
