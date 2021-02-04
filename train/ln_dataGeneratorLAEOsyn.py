'''
Based on the following example:
   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
(c) MJMJ/2018
'''

import numpy as np
import tensorflow.keras
from   tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

import ln_model_utils as mu
import mj_laeoUtils as LU
from ln_laeoImage import mj_drawHeadMapFromBBs, mj_drawHeadMapFromGeom
from mj_genericUtils import mj_isDebugging
import random
import copy
import sys

class DataGeneratorLAEOsyn(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, allSamples, targets=[], batch_size=32, dim=(10,64,64,3),
                 dim_map = (64,64,3),
                 shuffle=True, augmentation=True,
                 isTest=False, augmentation_x=1,
                 meanSample=[0.0], probdown=0.1,
                 withHMap=False, withGeom=True, meanSampleFM=[0.0], winlenMap=1):
        'Initialization'
        self.dim = dim
        self.time_dim = dim[0]
        self.dim_map = dim_map
        self.winlenMap = winlenMap
        self.batch_size = batch_size
        self.n_channels = 3

        self.list_IDs = np.arange(allSamples.shape[3])

        # if augmentation:
        #     for aix in range(0,augmentation_x):
        #         self.list_IDs = np.concatenate((self.list_IDs, -np.array(self.list_IDs)))  # Minus means to be perturbated
        # if isTest == False:
        #     np.random.shuffle(self.list_IDs)   # Done always!

        self.shuffle = shuffle
        self.augmentation = augmentation
        self.probdown = probdown
        self.allSamples = allSamples
        self.targets = targets
        self.meansample = meanSample
        self.meanSampleFM = meanSampleFM  # To be subtracted from samples

        self.withHMap = withHMap
        self.withGeom = withGeom

        self.augmenProb = 0.2     # Was 0.1
        self.isTest = isTest

        # Needed for initialization
        self.img_gen = ImageDataGenerator(width_shift_range=[-2,0,2], height_shift_range=[-2,0,2],
                                          brightness_range=[0.95,1.05], channel_shift_range=0.05,
                                          zoom_range=0.015, horizontal_flip=True)
        self.__splitSets()
        self.on_epoch_end()

        self.isDebug = mj_isDebugging()

    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        #Indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if len(indexes) < 1:
           #rnd = random.randint(0,len(self.indexes)-self.batch_size)
           #indexes = self.indexes[rnd:rnd+self.batch_size]
            indexes = range(0, min(self.batch_size, len(self.pairs)) )

        # Find list of ids
        list_IDs_temp = [self.pairs[k] for k in indexes]
        list_labels_temp = [self.pair_labs[k] for k in indexes]

        if self.withGeom:
            geom_temp = self.geom[indexes,]
        else:
            geom_temp = []

        if self.withHMap:
            list_hmaps_temp = [self.hmaps[k] for k in indexes]
        else:
            list_hmaps_temp = []

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp, geom_temp, list_hmaps_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.list_IDs))
        # #
        # # if self.shuffle == True:
        # #     np.random.shuffle(self.indexes)
        # if self.isTest:
        #     self.indexes = np.arange(len(self.list_IDs))
        # else:
        #     if self.shuffle:
        #         self.indexes = np.arange(len(self.list_IDs))
        #         np.random.shuffle(self.indexes)
        #         # np.random.shuffle(self.pos)
        #         # np.random.shuffle(self.neg)
        self.__generatePairs()
        self.indexes = np.arange(len(self.pairs))

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

        attempts = 0   # For very weird situations!
        while (len(tmppos) != len(tmpneg)) and (attempts < 10000):
            attempts += 1
            df = len(tmpneg) - len(tmppos)
            if len(tmpneg) > len(tmppos):
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

    def __splitSets(self):

        allyaws = abs(self.targets[:,2])
        allpitch = abs(self.targets[:, 1])

        yawthr_zero = 0.17
        pitch_zero = 0.11   # Restrictive

        pzerologic = allpitch <= pitch_zero

        cond45 = np.logical_and(allyaws >=0.6, allyaws < 0.96)
        self.y45idx = np.where(np.logical_and(cond45, pzerologic))[0]   # 35-55 degrees

        cond90 = np.logical_and(allyaws >=1.40, allyaws < 1.75)
        self.y90idx = np.where(np.logical_and(cond90, pzerologic))[0]   # 80-100 degrees

        cond135 = np.logical_and(allyaws >= 2.18, allyaws < 2.53)
        self.y135idx = np.where(np.logical_and(cond135, pzerologic))[0]  # 125-145 degrees

        self.yg90idx = np.where(np.logical_and(allyaws > 1.66, pzerologic))[0]  # Just greater than 90 degrees

        self.zeroidx = np.where(allyaws <= yawthr_zero)[0]


    def __splitInPosAndNegs(self):
        self.pos = []
        self.neg = []

        #for ix in self.list_IDs:
        for ix_ in range(0,len(self.pair_labs)):
            lab = self.pair_labs[ix_]
            if lab == 0:
                self.neg.append(ix_)
            else:
                self.pos.append(ix_)


    def __generatePairs(self):
        print(" ~ Preparing positive pairs...")
        if self.withHMap:
            pospairs, posgeom, poshmaps = self.__generatePositivePairs()
        else:
            pospairs, posgeom = self.__generatePositivePairs()

        pospair_labs = np.ones((len(pospairs)))  # Update me!

        print(" ~ Preparing negative pairs...")
        if self.withHMap:
            negpairs, neggeom, neghmaps = self.__generateNegativePairs(maxpercase=int(len(pospairs) / 6))
        else:
            negpairs, neggeom = self.__generateNegativePairs(maxpercase=int(len(pospairs)/6))

        negpair_labs = np.zeros((len(negpairs)))

        self.pairs = pospairs
        self.pairs.extend(negpairs)

        if self.withHMap:
            self.hmaps = poshmaps
            self.hmaps.extend(neghmaps)

        self.pair_labs = np.concatenate((pospair_labs, negpair_labs))

        if self.withGeom:
            try:
                self.geom = np.concatenate((posgeom, neggeom), axis=0)
            except:
                print("Exception in {}".format(__name__))
                print("{}".format(posgeom.shape))
                print("{}".format(neggeom.shape))
                exit(-1)
        else:
            self.geom = []

        self.__splitInPosAndNegs()


    def __generatePositivePairs(self):
        pairs = []
        geom = []
        maps = []


        # Profile case
        posidx = self.y90idx
        for i in range(0,len(posidx)-1):
            px1 = posidx[i]
            if self.targets[px1,2] > 0: # Looking at left
               px1 = -px1

            for j in range(i+1,len(posidx)):
                px2 = posidx[j]
                if self.targets[px2,2] < 0: # Looking at right
                    px2 = -px2
                pairs.append((px1,px2))

                g = np.array([random.normalvariate(0.5, 0.1), random.normalvariate(0, 0.01), random.normalvariate(1, 0.01)])
                g[0] = min(g[0], 0.8)
                geom.append(g)

                if self.withHMap:
                    hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                    maps.append(hmap)

        print("\t Current pos length: {}".format(len(geom)))

        # Intermediate case
        posidx1 = self.y45idx
        posidx2 = self.y135idx
        if len(posidx2) < 1:
            posidx2 = self.yg90idx

        for i in range(0,len(posidx2)-1):
            px1 = posidx2[i]
            if self.targets[px1,2] > 0: # Looking at left
               px1 = -px1

            for j in range(i+1,len(posidx1)):
                px2 = posidx1[j]
                if self.targets[px2,2] < 0: # Looking at right
                    px2 = -px2
                pairs.append((px1,px2))

                g = np.array([random.normalvariate(0.5, 0.1), random.normalvariate(0, 0.05), random.normalvariate(1, 0.05)])
                g[0] = min(g[0], 0.8)
                geom.append(g)

                if self.withHMap:
                    hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                    maps.append(hmap)

                # We could also add the reversed case
                if random.random() > 0.6:
                    pairs.append((-px2, -px1))
                    g = np.array([random.normalvariate(0.5, 0.1), random.normalvariate(0, 0.05), random.normalvariate(1, 0.05)])
                    g[0] = min(g[0], 0.8)
                    geom.append(g)

                    if self.withHMap:
                        hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                        maps.append(hmap)

        print("\t Current pos length: {}".format(len(geom)))
        sys.stdout.flush()

        if self.withHMap:
            return pairs, np.asarray(geom), maps
        else:
            return pairs, np.asarray(geom)


    def __generateNegativePairs(self, maxpercase=5000):
        pairs = []
        geom = []
        maps = []
        #maxpercase = 1000

        # Profile same direction
        negidx = np.concatenate((self.y90idx, self.y45idx))
        np.random.shuffle(negidx)
        terminate_loop = False
        for i in range(0, len(negidx) - 1):
            sys.stdout.write("{:06d} of {:06d}".format(i, len(negidx)))
            sys.stdout.write("\r")
            sys.stdout.flush()

            px1 = negidx[i]

            for j in range(i + 1, len(negidx)):
                px2 = negidx[j]
                if np.sign(self.targets[px1, 2])*np.sign(self.targets[px2, 2]) < 0:  # Different viewpoint?
                    px2 = -px2
                pairs.append((px1, px2))
                if random.random() > 0.5:
                    g = np.array([np.random.uniform(0.01, 1.0), random.normalvariate(0, 0.01), random.normalvariate(1, 0.02)])
                else:
                    g = np.array(
                        [np.random.uniform(0.01, 1.0), random.normalvariate(0.4, 0.01), random.normalvariate(1, 0.2)])

                g[0] = min(g[0], 0.8)
                g[1] = min(g[1], 0.8)
                g[1] = max(g[1], -0.8)
                g[2] = max(g[2], 0.5)    # DEVELOP!!!
                geom.append(g)

                if self.withHMap:
                    hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                    maps.append(hmap)

                if len(pairs) >= maxpercase:
                    terminate_loop = True
                    break

            if terminate_loop:
                break

        print("\t Current neg length: {}".format(len(geom)))

        # Profile with frontal
        frontidx = self.zeroidx
        terminate_loop = False
        for i in range(0, len(negidx) - 1):
            sys.stdout.write("{:06d} of {:06d}".format(i, len(negidx)))
            sys.stdout.write("\r")
            sys.stdout.flush()

            px1 = negidx[i]

            for j in range(i + 1, len(frontidx)):
                px2 = frontidx[j]
                if np.sign(self.targets[px1, 2])*np.sign(self.targets[px2, 2]) < 0:  # Different viewpoint?
                    px2 = -px2
                pairs.append((px1, px2))

                if random.random() > 0.5:
                    g = np.array([np.random.uniform(0.01,1.0), random.normalvariate(0, 0.01), random.normalvariate(1, 0.01)])
                else:
                    g = np.array([np.random.uniform(0.01,1.0), random.normalvariate(0.4, 0.01), random.normalvariate(1, 0.2)])

                g[0] = min(g[0], 0.8)
                g[1] = min(g[1], 0.8)
                g[1] = max(g[1], -0.8)
                g[2] = max(g[2], 0.5)  # DEVELOP!!!
                geom.append(g)
                if self.withHMap:
                    hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                    maps.append(hmap)

                if len(pairs) >= 2*maxpercase:
                    terminate_loop = True
                    break

            if terminate_loop:
                break

        print("\t Current neg length: {}".format(len(geom)))

        # Profile opposite direction
        terminate_loop = False
        for i in range(0, len(negidx) - 1):
            sys.stdout.write("{:06d} of {:06d}".format(i, len(negidx)))
            sys.stdout.write("\r")
            sys.stdout.flush()

            px1 = negidx[i]
            if self.targets[px1,2] < 0: # Looking at right?
               px1 = -px1

            for j in range(i + 1, len(negidx)):
                px2 = negidx[j]
                if np.sign(self.targets[px2, 2]) > 0:  # Looking at left?
                    px2 = -px2
                pairs.append((px1, px2))

                if random.random() > 0.5:
                    g = np.array([np.random.uniform(0.01,1.0), random.normalvariate(0, 0.01),
                                  random.normalvariate(1, 0.01)])
                else:
                    g = np.array([np.random.uniform(0.01,1.0), random.normalvariate(0.4, 0.01),
                                  random.normalvariate(1, 0.02)])

                g[0] = min(g[0], 0.8)
                g[1] = min(g[1], 0.8)
                g[1] = max(g[1], -0.8)
                g[2] = max(g[2], 0.5)  # DEVELOP!!!
                geom.append(g)

                if self.withHMap:
                    hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                    maps.append(hmap)

                if len(pairs) >= 3 * maxpercase:
                    terminate_loop = True
                    break

            if terminate_loop:
                break

        print("\t Current neg length: {}".format(len(geom)))

        # Frontal with frontal
        np.random.shuffle(frontidx)
        terminate_loop = False
        cte_limit = 2500 # DEVELOP!!!
        for i in range(0, min(cte_limit, len(frontidx) - 1)):
            sys.stdout.write("{:06d} of {:06d}x{:06d}".format(i, len(frontidx), len(frontidx)))
            sys.stdout.write("\r")
            sys.stdout.flush()

            px1 = frontidx[i]

            for j in range(i + 1, len(frontidx)):
                px2 = frontidx[j]
                if np.sign(self.targets[px1, 2])*np.sign(self.targets[px2, 2]) < 0:  # Different viewpoint?
                    px2 = -px2
                pairs.append((px1, px2))
                if random.random() > 0.6:
                    g = np.array([np.random.uniform(0.01, 1.0), random.normalvariate(0, 0.05), random.normalvariate(1, 0.02)])
                else:
                    g = np.array(
                        [np.random.uniform(0.01, 1.0), random.normalvariate(0, 0.4), random.normalvariate(1, 0.3)])
                g[0] = min(g[0], 0.8)
                g[1] = min(g[1], 0.8)
                g[1] = max(g[1], -0.8)
                g[2] = max(g[2], 0.5)  # DEVELOP!!!
                geom.append(g)

                if self.withHMap:
                    hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                    maps.append(hmap)

                if len(pairs) >= 4*maxpercase:
                    terminate_loop = True
                    sys.stdout.write("\n")
                    break

            if terminate_loop:
                break

        print("\t Current neg length: {}".format(len(geom)))
        sys.stdout.flush()

        # Same head, facing but wrong geometry
        np.random.shuffle(negidx)
        terminate_loop = False
        for i in range(0, len(negidx) - 1):
            sys.stdout.write("{:06d} of {:06d}".format(i, len(negidx)))
            sys.stdout.write("\r")
            sys.stdout.flush()

            px1 = negidx[i]
            px2 = px1

            if np.sign(self.targets[px1, 2]) > 0:
                px1 = -px1

            if np.sign(self.targets[px2, 2]) < 0:
                px2 = -px2

            pairs.append((px1, px2))

            # dy should be invalid
            g = np.array([np.random.uniform(0.01, 0.5), random.normalvariate(0.65, 0.1), random.normalvariate(1, 0.1)])
            g[0] = min(g[0], 0.8)
            g[1] = min(g[1], 0.8)
            g[1] = max(g[1], -0.8)
            g[2] = max(g[2], 0.5)  # DEVELOP!!!
            geom.append(g)

            if self.withHMap:
                hmap = mj_drawHeadMapFromGeom(g, target_size=(64, 64))
                maps.append(hmap)

            if len(pairs) >= 5*maxpercase:
                terminate_loop = True
                sys.stdout.write("\n")
                break

        print("\t Current neg length: {}".format(len(geom)))
        sys.stdout.flush()

        if self.withHMap:
            return pairs, np.asarray(geom), maps
        else:
            return pairs, np.asarray(geom)


    def __unrollImage(self, img, timelen, skipnorm=False, is_map=False):
        if not is_map:
            sample = np.zeros((self.dim)+ (self.n_channels, ), dtype=type(img))
        else:
            sample = np.zeros((timelen,)+ (self.dim_map), dtype=type(img))

        img2 = copy.deepcopy(img)

        rndnum = np.random.random()

        if (not is_map) and rndnum <= self.probdown:
            if rndnum < self.probdown / 3:
                img2 = cv2.resize(img2, (48, 48))
            else:
                img2 = cv2.resize(img2, (56, 56))

            img2 = cv2.resize(img2, (self.dim[1], self.dim[2]))

        if (not is_map) and self.dim[1] != img2.shape[0]:
            img2 = cv2.resize(img2, (self.dim[1], self.dim[2]))

        for tix in range(0, timelen):
            transf = self.img_gen.get_random_transform(img2.shape)
            transf["flip_horizontal"] = False
            s = self.img_gen.apply_transform(img2, transf)
            if s.max() > 1.01 and not skipnorm:  # Is 255-valued
                s /= 255.0
            sample[tix, :, :, :] = s

        return sample

    def __data_generation(self, list_IDs_temp, list_labels_temp, geom_temp, list_hmaps_temp=[]):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.random.random((self.batch_size, ) + self.dim + (self.n_channels,))
        X2 = np.random.random((self.batch_size, ) + self.dim + (self.n_channels,))

        G = np.zeros((self.batch_size, 3))
        if self.winlenMap > 1:
            M = np.zeros((self.batch_size, self.winlenMap, 64, 64, 3))
        else:
            M = np.zeros((self.batch_size, 64, 64, 3))
        y = np.zeros((self.batch_size, 2), dtype=int)
        #
        # yG = np.zeros((self.batch_size, 2), dtype=int)

        useMean = (type(self.meansample) == type(self.allSamples)) and (self.meansample.shape[0] == self.allSamples.shape[0])
        if self.withHMap and len(self.meanSampleFM) > 1:
            useMeanMap = self.meanSampleFM['meanmap'].shape[1] == self.hmaps[0].shape[1]
        else:
            useMeanMap = False

        nsamples = 0
        windowLen = self.dim[0]


        # Loop on ids
        for trix in range(0,len(list_IDs_temp)):
            pair = list_IDs_temp[trix]
            label = list_labels_temp[trix]

            trix1 = abs(pair[0])
            trix2 = abs(pair[1])
            img1 = copy.deepcopy(self.allSamples[:,:,:,trix1])
            img2 = copy.deepcopy(self.allSamples[:, :, :, trix2])

            if pair[0] < 0:
                transf = self.img_gen.get_random_transform(img1.shape)
                transf["flip_horizontal"] = True
                img1 = self.img_gen.apply_transform(img1, transf)

            if pair[1] < 0:
                transf = self.img_gen.get_random_transform(img2.shape)
                transf["flip_horizontal"] = True
                img2 = self.img_gen.apply_transform(img2, transf)

            _DEBUG_ = self.isDebug
            if _DEBUG_:
                outdir = "/tmp"

                if label > 0:
                    imagename = os.path.join(outdir, "possample{:06d}.jpg".format(trix))
                else:
                    imagename = os.path.join(outdir, "negsample{:06d}.jpg".format(trix))

                mosaic = np.concatenate((img1, img2), axis=1)
                cv2.imwrite(imagename, mosaic)


            # Do this after any transform!!! As the function rescales to 255!!!
            img1 /= 255.0
            img2 /= 255.0

            s1 = self.__unrollImage(img1, windowLen)
            s2 = self.__unrollImage(img2, windowLen)

            # As unrollImage applies transformations, the range changes to [0,255], so if the mean value
            # has been subtracted before, the data range is not centered at 0 anymore. Solution: remove mean
            # after unroll
            if useMean:
                for t in range(0, windowLen):
                    s1[t, :, :, :] -= self.meansample
                    s2[t, :, :, :] -= self.meansample

            X1[nsamples,] = s1
            X2[nsamples,] = s2
            #G[nsamples,] = np.array([random.normalvariate(0.5,0.1), random.normalvariate(0,0.01), random.normalvariate(1,0.01)])
            if self.withGeom:
                G[nsamples,] = geom_temp[trix,]

            if self.withHMap:
                hmap = self.__unrollImage(list_hmaps_temp[trix], self.winlenMap, skipnorm=True, is_map=True) # Do not do normalization here, but after subtracting the mean

                if useMeanMap:
                    for t in range(0, self.winlenMap):
                        if t < 10:
                            hmap[t, :, :, :] -= self.meanSampleFM['meanmap'][t*64:(t+1)*64,]
                        else:
                            hmap[t, :, :, :] -= self.meanSampleFM['meanmap'][9*64:(9+1)*64,]  # Patch for longer than 10

                if self.winlenMap > 1:
                    M[nsamples, ] = hmap / 255.0
                else:
                    time_half = int(np.floor(self.time_dim/2))
                    if hmap.shape[0] != 1:
                        M[nsamples,] = hmap[time_half,] / 255.0  # We need just the 'central' map
                    else:
                        M[nsamples,] = np.squeeze(hmap) /255.0

            #X[nsamples,] = pair
            y[nsamples, int(label)] = 1

            nsamples = nsamples+1

        if self.dim[0] == 1:
            X1 = np.squeeze(X1)
            X2 = np.squeeze(X2)

        Xout = []
        Xout.append(X1)
        Xout.append(X2)
        if self.withGeom:
            Xout.append(G)


        if self.withHMap:
            Xout.append(M)

        return Xout, y

