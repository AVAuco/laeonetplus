'''
(c) MJMJ/2018
'''

import mj_laeoUtils as LU
from tensorflow.keras import preprocessing
from joblib import Parallel, delayed
import numpy as np
import cv2

from os.path import expanduser

homedir = expanduser("~")


def mj_gatherSampleFromId(trix_, laeoIdxs, allSamples, dim, meanSample,
                         augmentation, img_gen):
    trix = abs(trix_)
    tracks = laeoIdxs[allSamples[trix][0]]["tracks"]
    pair, label, geom = LU.mj_readTempLAEOsample(allSamples[trix], tracks, dim[0], meanSample)

    if augmentation and trix_ < 0:  # random.random() < self.augmenProb:

        transf = img_gen.get_random_transform(pair[0,].shape)
        for slix in range(0, pair.shape[0]):
            pair[slix,] = LU.mj_applyTransformToPair(pair[slix,], img_gen, transf)

        if transf["flip_horizontal"] == 1:  # Do the geom correction?
            geom[0] = -geom[0]  # Difference is opposite
            geom[2] = 1.0 / geom[2]  # Ratio of scales is inverted

    return pair, label, geom



def mj_gatherSampleFromListIds(listIds, laeoIdxs, allSamples, dim, meanSample,
                         augmentation, img_gen, transform=None):
    num_cores = 4
    lresults_pool = Parallel(n_jobs=num_cores)(delayed(LU.mj_readTempLAEOsample)
                                               (allSamples[abs(trix_)],
                                                laeoIdxs[allSamples[abs(trix_)][0]]["tracks"],
                                                dim[0], meanSample) for trix_ in listIds)

    X = np.random.random((len(listIds), dim[0], dim[1], dim[2], 3))
    G = np.zeros((len(listIds), 3))

    labels = [0] * len(listIds)

    for i in range(0,len(listIds)):
        trix_ = listIds[i]
        # trix = abs(trix_)
        # tracks = laeoIdxs[allSamples[trix][0]]["tracks"]
        # pair, label, geom = LU.mj_readTempLAEOsample(allSamples[trix], tracks, dim[0], meanSample)
        pair = lresults_pool[i][0]
        label = lresults_pool[i][1]
        geom = lresults_pool[i][2]

        if augmentation and trix_ < 0:  # random.random() < self.augmenProb:
            if transform is None:
                transf = img_gen.get_random_transform(pair[0,].shape)
            else:
                transf = transform

            for slix in range(0, pair.shape[0]):
                pair[slix,] = LU.mj_applyTransformToPair(pair[slix,], img_gen, transf)

            if transf["flip_horizontal"] == 1:  # Do the geom correction?
                geom[0] = -geom[0]  # Difference is opposite
                geom[2] = 1.0 / geom[2]  # Ratio of scales is inverted

        if pair.shape[1] != X.shape[2]:
            pair_ = cv2.resize(np.squeeze(pair), (X.shape[3], X.shape[2]))  # resize needs (width, height)
            pair = np.expand_dims(pair_, axis=0)

        X[i,] = pair
        labels[i] = label
        G[i,] = geom

    return X, labels, G

# ==================================================================================

def mj_readTempFrameCropLAEOsample(config, tracks, winlen=10, meanSample=0.0, scaleDiv=255.0):
    import os
    from mj_laeoImage import mj_cropFrameByTracks
    import model_utils as MU

    videoname = config[0]
    frameId = config[1]
    pair = config[2]
    label = config[3]

    bbdirbase = homedir+"/databases/avalaeo/annotations_head/"
    bbfile = os.path.join(bbdirbase, videoname+"_heads.txt")
    lBBs = MU.parseFileAnnotations(bbfile)

    track1 = tracks[pair[0]]
    track2 = tracks[pair[1]]

    # Define image dir
    framesdirbase = homedir+"/databases/avalaeo/frames_per_video"
    framesdir = os.path.join(framesdirbase, videoname)

    # Define time interval
    lcrops = mj_cropFrameByTracks(framesdir, lBBs, track1, track2, winlen, (128,128))

    return lcrops/scaleDiv, label

# -------------------------------------------------------------------------------


def mj_readTempFrameCropLAEOsamplePresaved(config, tracks, winlen=10, meanSample=0.0, scaleDiv=255.0):


    import os
    #from mj_laeoImage import mj_cropFrameByTracks
    #import model_utils as MU
    import cv2

    videoname = config[0]
    frameId = config[1]
    pair = config[2]
    label = config[3]

    # bbdirbase = "/home/mjmarin/databases/avalaeo/annotations_head/"
    # bbfile = os.path.join(bbdirbase, videoname+"_heads.txt")
    # lBBs = MU.parseFileAnnotations(bbfile)
    #
    # track1 = tracks[pair[0]]
    # track2 = tracks[pair[1]]

    # Define image dir
    framesdirbase = homedir+"/databases/avalaeo/frame_crops"


    # samplename = "{:s}_{:04d}_{:02d}_{:02d}_{:d}.jpg".format(videoname,
    #                                             frameId,
    #                                             pair[0], pair[1],
    #                                             label)

    samplename = "{:s}_{:04d}_{:02d}_{:02d}.jpg".format(videoname,
                                                frameId,
                                                pair[0], pair[1])

    framename = os.path.join(framesdirbase, samplename)
    img = cv2.imread(framename)

    if img is None:
        print("Error reading image: "+framename)
        return None, 0

    # Convert to sample format
    lcrops = np.zeros((winlen, int(img.shape[0]/winlen), img.shape[1], img.shape[2]), np.uint8)
    step = lcrops.shape[1]
    for t in range(0,winlen):
        lcrops[t,] = img[t*step:(t+1)*step,]

    # Define time interval
#    lcrops = mj_cropFrameByTracks(framesdir, lBBs, track1, track2, winlen, (128,128))

    return lcrops/scaleDiv, label

# -------------------------------------------------------------------------------

def mj_readTempFrameCropMapLAEOsamplePresaved(config, tracks, winlen=10, meanSample=0.0, scaleDiv=255.0):
    from mj_genericUtils import  mj_isDebugging

    import os
    import cv2

    videoname = config[0]
    frameId = config[1]
    pair = config[2]
    label = config[3]

    # Define image dir
    framesdirbase = homedir+"/databases/avalaeo/frame_crops_v2"

    samplename = "{:s}_{:04d}_{:02d}_{:02d}.jpg".format(videoname,
                                                frameId,
                                                pair[0], pair[1])

    framename = os.path.join(framesdirbase, samplename)
    img = cv2.imread(framename)

    if img is None:
        print("Error reading image: "+framename)

        if mj_isDebugging():
            return np.random.rand(winlen, 128, 128,3), np.zeros((winlen, 64, 64, 3)), int(np.random.sample() > 0.5)
        else:
            return None, None, 0

    # Convert to sample format
    if winlen == 10:
        lcrops = np.zeros((winlen, int(img.shape[0]/winlen), img.shape[1], img.shape[2]), np.float32)
    else:
        lcrops = np.zeros((winlen, img.shape[1], img.shape[1], img.shape[2]), np.float32)

    step = lcrops.shape[1]
    curlen_crop = int(img.shape[0]/step)
    for t in range(0,winlen):
        if t < curlen_crop:
            lcrops[t,] = img[t*step:(t+1)*step,] - meanSample['meancrop'][t*step:(t+1)*step,]
        else:
            lcrops[t,] = img[(curlen_crop-1)*step:curlen_crop*step,] - meanSample['meancrop'][t*step:(t+1)*step,]

	# Same for maps	
    samplename = "{:s}_{:04d}_{:02d}_{:02d}_maps.jpg".format(videoname,
                                                frameId,
                                                pair[0], pair[1])

    framename = os.path.join(framesdirbase, samplename)
    img = cv2.imread(framename)

    if img is None:
        print("Error reading image: "+framename)

        if mj_isDebugging():
            return np.zeros((winlen, 128, 128,3)), np.zeros((winlen, 64, 64, 3)), label
        else:
            return None, None, 0

    # Convert to sample format
    if winlen == 10:
        lmaps = np.zeros((winlen, int(img.shape[0]/winlen), img.shape[1], img.shape[2]), np.float32)
    else:
        lmaps = np.zeros((winlen, img.shape[1], img.shape[1], img.shape[2]), np.float32)

    step = lmaps.shape[1]
    curlen_map = int(img.shape[0]/step)
    for t in range(0,winlen):
        if t < curlen_map:
            lmaps[t,] = img[t*step:(t+1)*step,] - meanSample['meanmap'][t*step:(t+1)*step,]
        else:
            lmaps[t,] = img[(curlen_map-1)*step:curlen_map*step,] - meanSample['meanmap'][t*step:(t+1)*step,]
	
		
    return lcrops/scaleDiv, lmaps/scaleDiv, label

	
# -------------------------------------------------------------------------------
def mj_gatherFrameCropSampleFromListIds(listIds, laeoIdxs, allSamples, dim, meanSample,
                         augmentation, img_gen):
    num_cores = 1

    if num_cores > 1:
        lresults_pool = Parallel(n_jobs=num_cores)(delayed(mj_readTempFrameCropLAEOsample)
                                               (allSamples[abs(trix_)],
                                                laeoIdxs[allSamples[abs(trix_)][0]]["tracks"],
                                                dim[0], meanSample) for trix_ in listIds)
    else:
        lresults_pool = []
        for trix_ in listIds:
            A, B = mj_readTempFrameCropLAEOsamplePresaved(allSamples[abs(trix_)],
                                                laeoIdxs[allSamples[abs(trix_)][0]]["tracks"],
                                                dim[0], meanSample)
            lresults_pool.append((A,B))


    X = np.zeros((len(listIds), dim[0], dim[1], dim[2], 3))
    #G = np.zeros((len(listIds), 3))

    labels = [0] * len(listIds)

    for i in range(0,len(listIds)):
        trix_ = listIds[i]
        # trix = abs(trix_)
        # tracks = laeoIdxs[allSamples[trix][0]]["tracks"]
        # pair, label, geom = LU.mj_readTempLAEOsample(allSamples[trix], tracks, dim[0], meanSample)
        pair = lresults_pool[i][0]
        label = lresults_pool[i][1]
        #geom = lresults_pool[i][2]

        if augmentation and trix_ < 0:  # random.random() < self.augmenProb:
            normalize = pair[0,].max() <= 1.0
            transf = img_gen.get_random_transform(pair[0,].shape)
            for slix in range(0, pair.shape[0]):
                #pair[slix,] = LU.mj_applyTransformToPair(pair[slix,], img_gen, transf)
                pair[slix,] = img_gen.apply_transform(pair[slix,], transf)
                if normalize:
                    pair[slix,] /= 255.0

            # if transf["flip_horizontal"] == 1:  # Do the geom correction?
            #     geom[0] = -geom[0]  # Difference is opposite
            #     geom[2] = 1.0 / geom[2]  # Ratio of scales is inverted

        X[i,] = pair
        labels[i] = label
        #G[i,] = geom

    return X, labels #, G
	
	
def mj_gatherFrameCropMapSampleFromListIds(listIds, laeoIdxs, allSamples, dim, dim_map, meanSample,
                         augmentation, img_gen, transform=None, winlenMap=1):
    num_cores = 1

    time_dim = dim[0]
    time_dim_center = int(np.floor(time_dim/2))

    if num_cores > 1:  # TODO: this is not working
        lresults_pool = Parallel(n_jobs=num_cores)(delayed(mj_readTempFrameCropLAEOsample)
                                               (allSamples[abs(trix_)],
                                                laeoIdxs[allSamples[abs(trix_)][0]]["tracks"],
                                                dim[0], meanSample) for trix_ in listIds)
    else:
        lresults_pool = []
        for trix_ in listIds:
            A, A2, B = mj_readTempFrameCropMapLAEOsamplePresaved(allSamples[abs(trix_)],
                                                laeoIdxs[allSamples[abs(trix_)][0]]["tracks"],
                                                dim[0], meanSample)
            lresults_pool.append((A,A2,B))


    winlen = dim[0]
    X = np.zeros((len(listIds), dim[0], dim[1], dim[2], 3))
    if winlenMap > 1:
        M = np.zeros((len(listIds), winlenMap, dim_map[1], dim_map[2], 3))
    else:
        M = np.zeros((len(listIds), dim_map[1], dim_map[2], 3))
	#G = np.zeros((len(listIds), 3))

    labels = [0] * len(listIds)

    for i in range(0,len(listIds)):
        trix_ = listIds[i]
        # trix = abs(trix_)
        # tracks = laeoIdxs[allSamples[trix][0]]["tracks"]
        # pair, label, geom = LU.mj_readTempLAEOsample(allSamples[trix], tracks, dim[0], meanSample)
        pair = lresults_pool[i][0]
        label = lresults_pool[i][2]
        
        flipMap = False
        if augmentation and trix_ < 0:  # random.random() < self.augmenProb:
            normalize = pair.max() <= 1.0

            if transform is None:
                transf = img_gen.get_random_transform(pair[0,].shape)   # TODO: use right shape [128,128]
            else:
                transf = transform

            #nrows = int(pair.shape[0] / winlen)
            if transf["flip_horizontal"] == 1:
                flipMap = True

            for wix in range(0, winlen):
                pair[wix,] = img_gen.apply_transform(pair[wix,], transf)

            if normalize:
                pair /= 255.0

        X[i,] = pair

        if winlenMap > 1:
            if flipMap:
                for mix in range(0, winlenMap):
                    dd_ = int((10 - winlenMap) / 2)
                    map_tmp_ = np.fliplr(lresults_pool[i][1][mix+dd_,])
                    M[i, mix,] = map_tmp_[:, :, [1, 0, 2]]  # Swap channels, as left head and right head have been flipped as well
            else:
                if winlenMap == 10:
                    M[i,] = lresults_pool[i][1]
                else:
                    for mix in range(0, winlenMap):
                        dd_ = int((10-winlenMap) / 2)
                        M[i,mix,] = lresults_pool[i][1][mix+dd_,]
        else:
            if flipMap:
                map_tmp = np.fliplr(np.squeeze(lresults_pool[i][1][time_dim_center,]))
                map = map_tmp[:, :, [1, 0, 2]]  # Swap channels, as left head and right head have been flipped as well
                M[i,] = map
            else:
                M[i,] = np.squeeze(lresults_pool[i][1][time_dim_center,])    # Head map

        labels[i] = label
        #G[i,] = geom

    return X, M, labels #, G