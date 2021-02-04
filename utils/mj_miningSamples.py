"""
File: mj_miningSamples.py

(c) MJMJ/2018
"""

import os
import scipy.io
import cv2
import numpy as np

import mj_laeoUtils as LU
from ln_laeoImage import mj_cropImageFromBBs, mj_drawHeadMapFromBBs
import copy


def mj_detBB2squared(bb, imagesize, margin=0.05):
    '''
    Obtains a squared BB from a detection BB
    :param bb: [xmin ymin xmax ymax]
    :param imagesize: [nrows, ncols]
    :param margin: float number in [0,1]
    :return: new BB
    '''
    # (c) MJMJ/2018

    bbsq = copy.deepcopy(bb)

    w = int(bb[2]) - bb[0] + 1
    h = int(bb[3]) - bb[1] + 1

    s = max([w, h])

    # df = abs(w-h)
    # df2 = np.floor(df/2)

    center = [(int(bb[2]) + bb[0]) / 2, (int(bb[3]) + bb[1]) / 2]

    w2 = s * (1 + margin)
    h2 = w2

    bb2 = [center[0] - w2 / 2,
           center[1] - h2 / 2,
           center[0] + w2 / 2,
           center[1] + h2 / 2
           ]

    # Check limits
    bbsq[0] = round(max(1, bb2[0]))
    bbsq[1] = round(max(1, bb2[1]))
    bbsq[2] = round(min(imagesize[1], bb2[2]))
    bbsq[3] = round(min(imagesize[0], bb2[3]))

    return bbsq


# ---------------------------------------------------------
def mj_getDetsSeq(videoname, detsdir, framesdir):
    matfile = os.path.join(detsdir, videoname + "_ssd_head.mat")

    # Open file
    f = scipy.io.loadmat(matfile)

    # Get detection struct
    R = f["R"]

    imnames = R["imname"]  # Filenames
    dets = R["dets"]  # Dets

    nsamples = imnames.shape[1]
    #print(nsamples)

    lCenters = []
    lCrops = []

    # For each image, get detections
    for ix in range(0, nsamples):
        imdets = dets[0, ix]
        ndets = imdets.shape[0]

        imname = imnames[0, ix][0]
        framenum = int(imname[0:3])

        #print(imname)

        imfullpath = os.path.join(framesdir, videoname, "{:06d}.jpg".format(framenum))

        img = cv2.imread(imfullpath)

        # cv2.imshow("Frame", img)
        # cv2.waitKey(-1)
        imagesize = img.shape
        margin = 0.05

        centers = []
        crops = []

        for dix in range(0, ndets):
            score = imdets[dix,4]

            if ndets > 2 and score < 0.25:
                continue

            bb_ = np.around(imdets[dix,0:4])


            bb = mj_detBB2squared(bb_, imagesize, margin)

            bb = bb.astype(int)

            crop = img[bb[1]:bb[3], bb[0]:bb[2], ]

            centers.append((((bb[0] + bb[2]) / 2,
                             (bb[1] + bb[3]) / 2,
                             max(crop.shape))))

            crop = cv2.resize(crop, (64, 64))

            # cv2.imshow("Crop", crop)
            # cv2.waitKey(-1)
            crops.append(crop)

        #print(len(centers))
        lCenters.append(centers)
        lCrops.append(crops)

    return lCrops, lCenters, imagesize

# ---------------------------------------------------------
def mj_getDetsSeqWithFM(videoname, detsdir, framesdir, frame_crop_size=(128,128)):
    matfile = os.path.join(detsdir, videoname + "_ssd_head.mat")

    # Open file
    f = scipy.io.loadmat(matfile)

    # Get detection struct
    R = f["R"]

    imnames = R["imname"]  # Filenames
    dets = R["dets"]  # Dets

    nsamples = imnames.shape[1]
    #print(nsamples)

    lCenters = []
    lCrops = []
    lFrames = []

    # For each image, get detections
    for ix in range(0, nsamples):
        imdets = dets[0, ix]
        ndets = imdets.shape[0]

        imname = imnames[0, ix][0]
        framenum = int(imname[0:3])

        #print(imname)

        imfullpath = os.path.join(framesdir, videoname, "{:06d}.jpg".format(framenum))

        img = cv2.imread(imfullpath)

        # cv2.imshow("Frame", img)
        # cv2.waitKey(-1)
        imagesize = img.shape
        margin = 0.05

        centers = []
        crops = []
        frames = []
        lBBs = []

        for dix in range(0, ndets):
            score = imdets[dix,4]

            if ndets > 2 and score < 0.25:
                continue

            bb_ = np.around(imdets[dix,0:4])


            bb = mj_detBB2squared(bb_, imagesize, margin)

            bb = bb.astype(int)

            crop = img[bb[1]:bb[3], bb[0]:bb[2], ]

            centers.append((((bb[0] + bb[2]) / 2,
                             (bb[1] + bb[3]) / 2,
                             max(crop.shape))))

            crop = cv2.resize(crop, (64, 64))

            # cv2.imshow("Crop", crop)
            # cv2.waitKey(-1)
            crops.append(crop)

            # Add BB for later processing
            lBBs.append(((bb[0], bb[1]), (bb[2],bb[3])))

        # Crop frame based on detections
        # BBs in list format: ((x1,y1),(x2,y2))

        if len(lBBs) > 0:
            frame_crop = mj_cropImageFromBBs(img, lBBs)
            # Resize frame crop
            frames = cv2.resize(frame_crop, frame_crop_size)
        else:
            frames = []

        #print(len(centers))
        lCenters.append(centers)
        lCrops.append(crops)
        lFrames.append(frames)

    return lCrops, lCenters, lFrames, imagesize


# --------------------------------------------------------
def mj_groupDetsIntoTracks(lCenters):

    nframes = len(lCenters)
    lPairs = []

    for frix in range(0,nframes-1):
        centers01 = lCenters[frix]
        #for frix2 in range(frix+1,nframes):
        frix2 = frix+1
        centers02 = lCenters[frix2]

        pairs = []
        # Compare all dets from 01 with all from 02
        for i in range(0,len(centers01)):
            det01 = centers01[i]

            mindist = 9999999.0
            maxds = 0
            pair = (0, 0)

            for j in range(0, len(centers02)):
                det02 = centers02[j]

                # Distance
                d = (det01[0]-det02[0])**2 + (det01[1]-det02[1])**2
                ds = min(det01[2],det02[2]) / max(det01[2],det02[2])
                if d < mindist:
                    mindist = d
                    maxds = ds
                    pair = (i,j)

            pairs.append(pair)

        lPairs.append(pairs)

    # For each detection in the first frame, follow its track
    lTracks = []
    pairs0 = lPairs[0]
    for dix in range(0,len(pairs0)):

        track = [pairs0[dix][0]]
        prev = pairs0[dix][0]
        goto = pairs0[dix][1]
        for i in range(1,len(lPairs)):
            if len(lPairs[i]) < 1:
                continue
            pair_t1 = lPairs[i][goto]
            goto = pair_t1[1]

            track.append(pair_t1[0])

        lTracks.append(track)

    return lTracks

# --------------------------------------------------------
def mj_groupDetsIntoTracksFromEachFrame(lCenters, winlen):
    lTracks = []

    nframes = len(lCenters)

    for frix in range(0, nframes-winlen):
        lT = mj_groupDetsIntoTracks(lCenters[frix:nframes])
        lTracks.append(lT)

    return lTracks


# ------------------------------------------------------------------------

def mj_getAllLAEOnegativeSamples(detsdir, framesdir, winlen):

    negtracks = {"videoname": [],
                 "tracks": [],
                 "crops": [],
                 "geom": [],
                 "imsize": []}
    for nix in range(1,30):   #range(1,30):
        videoname = "negatives{:02d}".format(nix)
        print(videoname)

        lCrops, lCenters, imagesize = mj_getDetsSeq(videoname, detsdir, framesdir)

        # lTracks = mj_groupDetsIntoTracks(lCenters)
        lTracks = mj_groupDetsIntoTracksFromEachFrame(lCenters, winlen)

        negtracks["videoname"].append(videoname)
        negtracks["tracks"].append(lTracks)
        negtracks["crops"].append(lCrops)
        negtracks["geom"].append(lCenters)
        negtracks["imsize"].append(imagesize)

    return negtracks

# ------------------------------------------------------------------------

def mj_getAllLAEOnegativeSamplesWithFC(detsdir, framesdir, winlen):

    negtracks = {"videoname": [],
                 "tracks": [],
                 "crops": [],
                 "geom": [],
                 "frames": [],
                 #"maps": [],
                 "imsize": []}
    for nix in range(1,30):   #range(1,30):
        videoname = "negatives{:02d}".format(nix)
        print(videoname)

        lCrops, lCenters, lFrames, imagesize = mj_getDetsSeqWithFM(videoname, detsdir, framesdir)

        # lTracks = mj_groupDetsIntoTracks(lCenters)
        lTracks = mj_groupDetsIntoTracksFromEachFrame(lCenters, winlen)

        negtracks["videoname"].append(videoname)
        negtracks["tracks"].append(lTracks)
        negtracks["crops"].append(lCrops)
        negtracks["geom"].append(lCenters)
        negtracks["frames"].append(lFrames)
        #negtracks["maps"].append(lMaps)
        negtracks["imsize"].append(imagesize)

    return negtracks

# ------------------------------------------------------------------------

def mj_getNegLAEOpair(negsamples, videoname, timepos, winlen, meanSample=[0], imgsize=(64,64)):
    '''
    Gets just one pair of negative samples
    :param negsamples:
    :param videoname:
    :param timepos:
    :param winlen:
    :param imgsize: (rows, cols) of output crop
    :return: output images are already normalized (x/255)
    '''
    foo = 0
    nvids = len(negsamples["videoname"])
    vix = -1
    for vix_ in range(0,nvids):
        if negsamples["videoname"][vix_] == videoname:
            vix = vix_

    if vix >= 0:
        lTr = negsamples["tracks"][vix]
        if timepos >= len(lTr):
            return None, None
        ltrx = lTr[timepos]

        ntracks = len(ltrx)

        if ntracks < 2:
            return None, None

        # TODO: parametrize these values, currently, random
        if ntracks > 2:
            rnp = np.random.permutation(range(0, ntracks))
            t1 = rnp[0]
            t2 = rnp[1]
        else:
            t1 = 0
            t2 = 1

        cropsvid = negsamples["crops"][vix]
        geomvid = negsamples["geom"][vix]
        pair = np.zeros((winlen, imgsize[0], 2*imgsize[1],3))  # Allocate memory for the temporal sequence

        if len(ltrx[t1]) < winlen or len(ltrx[t2]) < winlen:   # Just in case
            return None, None

        # Define an image transformation
        from keras.preprocessing.image import ImageDataGenerator
        img_gen = ImageDataGenerator(width_shift_range=[-2, 0, 2], height_shift_range=[-2, 0, 2],
                                     brightness_range=[0.95, 1.05], channel_shift_range=0.05,
                                     zoom_range=0.015, horizontal_flip=True)
        transf = img_gen.get_random_transform(cropsvid[0][0].shape)
        transf["flip_horizontal"] = False

        G = 0

        for tix in range(timepos,timepos+winlen):
            ix1 = ltrx[t1][tix-timepos]
            ix2 = ltrx[t2][tix-timepos]

            if tix >= len(geomvid):
                return None, None

            # Check which one is on the left
            if geomvid[tix][ix1][0] > geomvid[tix][ix2][0]:
                ix1, ix2 = ix2, ix1  # Swap

            crop1 = copy.deepcopy(cropsvid[tix][ix1])
            crop1 = img_gen.apply_transform(crop1, transf)

            crop2 = copy.deepcopy(cropsvid[tix][ix2])
            crop2 = img_gen.apply_transform(crop2, transf)

            geo1 = geomvid[tix][ix1]
            geo2 = geomvid[tix][ix2]

            dx = geo2[0]-geo1[0]
            dy = geo2[1]-geo1[1]
            rscale = geo1[2] / geo2[2]

            crop1 = crop1/255.0
            crop2 = crop2/255.0

            if crop1.shape[0] != imgsize[0]:
                crop1 = cv2.resize(crop1, imgsize)
                crop2 = cv2.resize(crop2, imgsize)

            if type(meanSample) == np.ndarray and meanSample.shape[0] == crop1.shape[0]:
                crop1 -= meanSample
                crop2 -= meanSample

            p = np.concatenate((crop1,crop2), axis=1)

            pair[tix - timepos,] = p
            #cv2.imshow("Pair", p/255)
            #cv2.waitKey(-1)

            G = G + np.array([dx, dy, rscale])

        imgsize_ = negsamples["imsize"][vix]
        G = G / winlen
        G[0] = G[0] / imgsize_[1]
        G[1] = G[1] / imgsize_[0]

        if winlen == 1:
            pair = np.squeeze(pair)

        return pair, G
    else:
        return None, None

# ------------------------------------------------------------------------

def mj_getNegLAEOpairWithFM(negsamples, videoname, timepos, winlen, meanSample=[0], meanSampleFM=None, imgsize=(64,64),
                            windowLenMap=1):
    '''
    Gets just one pair of negative samples
    :param negsamples:
    :param videoname:
    :param timepos:
    :param winlen:
    :return: output images are already normalized (x/255)
    '''
    foo = 0
    nvids = len(negsamples["videoname"])
    vix = -1
    for vix_ in range(0,nvids):
        if negsamples["videoname"][vix_] == videoname:
            vix = vix_

    if vix >= 0:
        lTr = negsamples["tracks"][vix]
        ltrx = lTr[timepos]

        ntracks = len(ltrx)

        if ntracks < 2:
            return None, None, None, None

        # TODO: parametrize these values, currently, random
        if ntracks > 2:
            rnp = np.random.permutation(range(0, ntracks))
            t1 = rnp[0]
            t2 = rnp[1]
        else:
            t1 = 0
            t2 = 1

        cropsvid = negsamples["crops"][vix]
        geomvid = negsamples["geom"][vix]
        pair = np.zeros((winlen, imgsize[0], 2*imgsize[1], 3))  # Allocate memory for the temporal sequence


        lfcrops = negsamples["frames"][vix]
        fcrop = np.zeros((winlen,128,128,3))
        if windowLenMap == 1:
            hmap = np.zeros((64,64,3))
        else:
            hmap = np.zeros((windowLenMap, 64, 64, 3))

        G = 0

        t = 0
        for tix in range(timepos,timepos+winlen):
            ix1 = ltrx[t1][tix-timepos]
            ix2 = ltrx[t2][tix-timepos]

            # Check which one is on the left
            if geomvid[tix][ix1][0] > geomvid[tix][ix2][0]:
                ix1, ix2 = ix2, ix1  # Swap

            crop1 = cropsvid[tix][ix1]
            crop2 = cropsvid[tix][ix2]

            tmpcrop = lfcrops[tix]
            if not meanSampleFM is None:
                step = tmpcrop.shape[0]
                # import pdb; pdb.set_trace()
                tmpcrop = tmpcrop - meanSampleFM['meancrop'][t * step:(t + 1) * step, ]

            fcrop[tix-timepos,] = tmpcrop/255.0

            geo1 = geomvid[tix][ix1]
            geo2 = geomvid[tix][ix2]

            dx = geo2[0]-geo1[0]
            dy = geo2[1]-geo1[1]
            rscale = geo1[2] / geo2[2]

            crop1 = crop1/255.0
            crop2 = crop2/255.0

            if crop1.shape[0] != imgsize[0]:
                crop1 = cv2.resize(crop1, imgsize)
                crop2 = cv2.resize(crop2, imgsize)

            if type(meanSample) == np.ndarray and meanSample.shape[0] == crop1.shape[0]:
                crop1 -= meanSample
                crop2 -= meanSample

            p = np.concatenate((crop1,crop2), axis=1)

            pair[tix - timepos,] = p
            #cv2.imshow("Pair", p/255)
            #cv2.waitKey(-1)

            G = G + np.array([dx, dy, rscale])

            # Create map for the central frame
            if (windowLenMap == 1 and tix-timepos == 5) or windowLenMap > 1:
                # Simulate BBs from G
                lBBs = []
                sz = geo1[2]
                sz2 = sz/2
                lBBs.append((( int(geo1[0]-sz2), int(geo1[1]-sz2)),( int(geo1[0]+sz2), int(geo1[1]+sz2))))
                lBBs.append(((int(geo2[0]-sz2), int(geo2[1]-sz2)),(int(geo2[0]+sz2), int(geo2[1]+sz2))))
                # Draw the map
                hmap_ = mj_drawHeadMapFromBBs(negsamples["imsize"][vix], lBBs, (0,1), (64, 64))

                if windowLenMap == 1:
                    hmap = hmap_
                else:
                    if windowLenMap == 10:
                        hmap[tix - timepos,] = hmap_
                    elif (tix-timepos) < windowLenMap:
                            hmap[tix - timepos,] = hmap_

            t = t+1

        imgsize_ = negsamples["imsize"][vix]
        G = G / winlen
        G[0] = G[0] / imgsize_[1]
        G[1] = G[1] / imgsize_[0]
        #hmap /= winlen

        if winlen == 1:
            pair = np.squeeze(pair)

        if not meanSampleFM is None:
            if windowLenMap == 1:
                step = hmap.shape[0]
            else:
                step = hmap.shape[1]
            t = 5
            meanmap5 = meanSampleFM['meanmap'][t * step:(t + 1) * step, ]
            if windowLenMap == 1:
                hmap = hmap - meanmap5
            else:
                for j in range(0, windowLenMap):
                    hmap[j,] = hmap[j,] - meanmap5

        return pair, G, fcrop, hmap
    else:
        return None, None, None, None

# --------------------- MAIN ------------------------------
if __name__ == '__main__':
    winlen = 10

    framesdir = "/home/mjmarin/databases/avalaeo/frames_per_video"
    detsdir = "/home/mjmarin/databases/avalaeo/dets_per_video"
    #videoname = "negatives02"

    # lCrops, lCenters = mj_getDetsSeq(videoname, detsdir, framesdir)
    #
    # #lTracks = mj_groupDetsIntoTracks(lCenters)
    # lTracks = mj_groupDetsIntoTracksFromEachFrame(lCenters, winlen)
    #
    # track = lTracks[40][1]
    # for frix in range(0,len(track)):
    #     cv2.imshow("Crop", lCrops[frix][track[frix]])
    #     cv2.waitKey(150)
    doWithFrameCrop = True

    homedir = "/home/mjmarin/"

    if doWithFrameCrop:
        negdatafile = "/home/mjmarin/databases/avalaeo/negtracks_fm.h5"  #npz
    else:
        negdatafile = "/home/mjmarin/databases/avalaeo/negtracks.h5"
    import deepdish as dd
    if False:
        if doWithFrameCrop:
            negsamples = mj_getAllLAEOnegativeSamplesWithFC(detsdir, framesdir, winlen)
        else:
            negsamples = mj_getAllLAEOnegativeSamples(detsdir, framesdir, winlen)

        # Save data
        #np.savez(negdatafile, ns=negsamples)
        dd.io.save(negdatafile, negsamples)
    else:
        meanfile = os.path.join(homedir, "experiments", "deepLAEO", "meanMaps.h5")
        meanSampleFM = dd.io.load(meanfile)
        #np.load(negdatafile)
        negsamples = dd.io.load(negdatafile)
        timepos = 5
        if doWithFrameCrop:
            pair, geom, fcrop, hmap = mj_getNegLAEOpairWithFM(negsamples, "negatives01", timepos, winlen,
                                                              meanSample=[0.0], meanSampleFM=meanSampleFM)
        else:
            pair, geom = mj_getNegLAEOpair(negsamples, "negatives01", timepos, winlen)

        viz = LU.mj_genSeqImageForViz(pair)
        viz2 = LU.mj_genSeqImageForViz(fcrop)

        cv2.imshow("Sample", viz)
        cv2.imshow("SampleF", viz2)
        cv2.imshow("Hmap", hmap)
        cv2.waitKey(-1)


    print("End of program!")
