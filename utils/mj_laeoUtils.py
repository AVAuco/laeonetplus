'''
(c) MJMJ/2018
'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import ln_model_utils as mu
import time
import copy
import h5py
import os
import random

from os.path import expanduser
homedir = expanduser("~")


def mj_getGeometryPairFromBB(bbL, bbR, imgsize):

    dx = (int(bbR[0]) - bbL[0]) / imgsize[1]
    dy = (int(bbR[1]) - bbL[1]) / imgsize[0]

    aL = (int(bbL[2]) - bbL[0]+1) * (bbL[3] - bbL[1]+1)
    aR = (int(bbR[2]) - bbR[0]+1) * (bbR[3] - bbR[1]+1)

    rscale = aL / aR

    return np.array([dx,dy,rscale])

def bb_intersection_over_target(boxT, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxT[0], boxB[0])
    yA = max(boxT[1], boxB[1])
    xB = min(boxT[2], boxB[2])
    yB = min(boxT[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of the target rectangle
    boxTArea = (boxT[2] - boxT[0] + 1) * (boxT[3] - boxT[1] + 1)
    #boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over target area
    iot = interArea / float(boxTArea)

    # return the intersection over union value
    return iot


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def mj_generateTracks(bbsfile):
    '''Generate tracks of BBs from annotations'''
    head_ann = mu.parseFileAnnotations(bbsfile)

    nframes = len(head_ann)

    theTracks = []

    for frix in range(0,nframes):
        heads = head_ann[frix]
        nheads = len(heads)

        for hix in range(0,nheads):
            track = [frix, hix]
            boxA = (heads[hix][0][0], heads[hix][0][1], heads[hix][1][0], heads[hix][1][1])
            # For each head in this frame, find it in following ones until broken
            frix2 = frix+1
            broken = False
            while (broken == False) and (frix2 < nframes):
                heads2 = head_ann[frix2]

                # Compare to each head in future frame
                bestIOU = -1
                bestHix = -1
                for hix2 in range(0,len(heads2)):
                    boxB = (heads2[hix2][0][0], heads2[hix2][0][1], heads2[hix2][1][0], heads2[hix2][1][1])
                    iou = bb_intersection_over_union(boxA, boxB)
                    if iou > bestIOU:
                        bestIOU = iou
                        bestHix = hix2

                if bestIOU > 0.5 :
                    #Good matching
                    #pair = (hix, bestHix)
                    track.append(bestHix)
                else:
                    pair = ()
                    broken = True

                # Next frame
                frix2 = frix2+1

            # Save this track
            theTracks.append(track)

    return theTracks

def mj_getLabelsForTracks(pairfile, lTracks):
    """
    Gets corresponding couple for each frame in each track
    :param pairfile: path to pair annotation file
    :param lTracks: list of tracks
    :return: for each track, a list of ids pointing to its corresponding LAEO track; -1 if not LAEO
    """
    ntracks = len(lTracks)
    lLabels = []
    lannots_pairs = mu.parseFileLAEO(pairfile)
    for trix in range(0, ntracks):
        track = lTracks[trix]
        nframes = len(track)-1
        frameId = track[0]

        labels = [0] * nframes

        found = -1
        for frix in range(1,nframes):
            # Check if annotated as LAEO
            pairs = lannots_pairs[frameId]
            for pix in range(0,len(pairs)):
                if pairs[pix][0]-1 == track[frix]:
                    found = pairs[pix][1] - 1
                    break
                if pairs[pix][1] - 1 == track[frix]:
                    found = pairs[pix][0] - 1
                    break

            labels[frix-1] = found

            # Next one
            #frix = frix+1
        lLabels.append(labels)

    return lLabels


def mj_computeTempLAEOpairsFromVideo(videoname, annheaddir, annpairdir, annframedir):

    laeoframefile = annframedir + "/" + videoname + ".txt"
    laeoframe = mu.parseFileLAEOframe(laeoframefile)

    #bbsfile = "/home/mjmarin/databases/avalaeo/annotations_head/got01_heads.txt"
    bbsfile = annheaddir + "/" + videoname + "_heads.txt"

    stime = time.time()
    lTracks = mj_generateTracks(bbsfile)
    #print(time.time()-stime)

    ntracks = len(lTracks)
    #print(ntracks)

    # For each track, find if it is involved in any LAEO
    #pair_file = "/home/mjmarin/databases/avalaeo/annotations_pair/pair_got01.txt"
    pair_file = annpairdir + "/" + "pair_" + videoname + ".txt"

    lLabels = mj_getLabelsForTracks(pair_file, lTracks)
    #print(len(lLabels))

    #print('Done!')

    # Compute length of each track
    lLens = [0] * ntracks
    for trix in range(0,ntracks):
        lLens[trix] = len(lTracks[trix])-1

    # Pointer to tracks starting at each frame
    maxframes = 0
    for trix in range(0,ntracks):
        if lTracks[trix][0] > maxframes:
            maxframes = lTracks[trix][0]

    framesHash = []
    for frix in range(0,maxframes+1):
        thisFrame = []
        for trix in range(0,ntracks):
            if lTracks[trix][0] == frix:
                thisFrame.append(trix)

        framesHash.append(copy.deepcopy(thisFrame))

    #print('Done! [2]')

    # Create set of pairs
    samples = {"pos":[], "neg":[], "amb":[]}
    TEMP_WINDOW_LEN = 10
    middle = int(np.floor(TEMP_WINDOW_LEN/2))
    nframes = len(framesHash)
    for frix in range(0,nframes):
        thisFrame_ = framesHash[frix]
        used = np.zeros((len(thisFrame_), len(thisFrame_)))
        zeroIx = min(thisFrame_)
        for trix_ in range(0,len(thisFrame_)):
           trix = thisFrame_[trix_]
           if lLens[trix] >= TEMP_WINDOW_LEN :
               for j in thisFrame_:
                   if (trix != j) and lLens[j] >= TEMP_WINDOW_LEN:
                       framelab = laeoframe[frix+middle]
                       # There are cases where the whole frame has been annotated as ambiguous, therefore, it shouldn't be used
                       if framelab == 9:
                           samples["amb"].append((trix,j))
                           used[trix - zeroIx, j - zeroIx] = True
                           continue
                       # Choose as positive or negative
                       if (lTracks[trix][middle] == lLabels[j][middle-1]):
                       # This is a positive pair
                          if used[trix-zeroIx,j-zeroIx] == False and used[j-zeroIx,trix-zeroIx] == False:
                             samples["pos"].append((trix,j))
                             used[trix-zeroIx,j-zeroIx] = True
                       else:
                       # This is a negative pair
                            if used[trix-zeroIx, j-zeroIx] == False and used[j-zeroIx, trix-zeroIx] == False:
                                samples["neg"].append((trix, j))
                                used[trix-zeroIx, j-zeroIx] = True

    #print('Done! [3]')

    samples["tracks"] = lTracks

    return samples

def mj_readTempLAEOpairsFromH5(filename):
    # "/home/mjmarin/databases/avalaeo/pairstemp.h5"
    hf = h5py.File(filename, "r")
    allvideos = list(hf.keys())

    laeoAnnots = {allvideos[0]: []}

    for videoname in allvideos:
        samples = {"pos":[], "neg":[], "tracks":[]}

        g1 = hf.get(videoname)
        pos = np.array(g1.get("pos").get("pairs"))
        samples["pos"] = pos
        neg = np.array(g1.get("neg").get("pairs"))
        samples["neg"] = neg

        amb = np.array(g1.get("amb").get("pairs"))
        samples["amb"] = amb

        track_grp = g1.get("tracks")
        ntracks = len(list(track_grp.keys()))
        tracks = []
        for trix in range(0,ntracks):
            tracks.append(np.array(track_grp.get("track{}".format(trix))))

        samples["tracks"]= tracks
        laeoAnnots[videoname] = samples

    return laeoAnnots


def mj_readTempLAEOsample(config, tracks, winlen, meansample=[0.0]):
    """
    Load heads from images and build a temporal sequence
    :param config: tuple
    :param tracks: list
    :param winlen: temporal window length
    :param meansample: to be removed from each half (i.e. head)
    :return:
    """
    videoname = config[0]
    frameId = config[1]
    pair = config[2]
    label = config[3]

    ann_file = homedir+"/databases/avalaeo/annotations_head/"+  videoname + "_heads.txt"
    videonameCap = videoname.lower()
    #pair_file = "/home/mjmarin/databases/avalaeo/annotations_pair/pair_got01.txt"
    head_size = (64,64)

    useMean = type(meansample) != list and meansample.shape[0] == head_size[0]

    sample = np.zeros((winlen, head_size[0], 2*head_size[1], 3))
    geom = 0
    if len(tracks[-1]) > 0:
        last_vid_frame = tracks[-1][0]
    else:
        last_vid_frame = -1

    for tix in range(0,winlen):
        req_frid = frameId + tix
        if req_frid > last_vid_frame and last_vid_frame > 0:  # Check video boundaries
            req_frid = last_vid_frame

        imagepath = homedir+"/databases/avalaeo/frames_per_video/" + videonameCap +\
                    "/%06d.jpg" % (req_frid)
        crop_images, head_ann, imgsize = mu.get_cropped_head(ann_file, imagepath, req_frid, head_size)
        if len(crop_images) < 2:
            continue

        # Patch for shorter tracks: repeat last frame
        if (tix+1) >= len(tracks[pair[0]]):
            tix_ = len(tracks[pair[0]])-1
            ix1 = tracks[pair[0]][tix_]
        else:
            ix1 = tracks[pair[0]][tix+1]  # Remember: track[0] is the frameId --> +1
        
        if (tix+1) >= len(tracks[pair[1]]):
            tix_ = len(tracks[pair[1]])-1
            ix2 = tracks[pair[1]][tix_]
        else:
            ix2 = tracks[pair[1]][tix+1]  # Remember: track[0] is the frameId --> +1

        if len(head_ann) <= max(ix1,ix2):
            #print(imagepath)
            #print(len(head_ann))
            #print((ix1, ix2))
            print("WARNING: something to be checked!!!::mj_readTempLAEOsample " + imagepath)
            ix1=0
            ix2=1

        # Check right order of heads
        if head_ann[ix1][0][0] > head_ann[ix2][0][0]:
            ix1, ix2 = ix2, ix1    # Swap

        # Scale values
        c1 = crop_images[ix1] / 255.0
        c2 = crop_images[ix2] / 255.0
        if useMean:
            c1 -= meansample
            c2 -= meansample

        view = np.concatenate((c1, c2), axis=1)
        sample[tix,] = view

        # cv2.imshow("Heads", view)
        # cv2.waitKey(-1)
        dx = head_ann[ix2][0][0] - head_ann[ix1][0][0]
        dy = head_ann[ix2][0][1] - head_ann[ix1][0][1]
        ah1 = (head_ann[ix1][1][0] - head_ann[ix1][0][0]) * (head_ann[ix1][1][1] - head_ann[ix1][0][1])
        ah2 = (head_ann[ix2][1][0] - head_ann[ix2][0][0]) * (head_ann[ix2][1][1] - head_ann[ix2][0][1])
        rscale = ah1 / ah2
        geom = geom + np.array([dx,dy,rscale])

    geom = geom / winlen
    geom[0] = geom[0] / imgsize[1]
    geom[1] = geom[1] / imgsize[0]

    return sample, label, geom

def mj_prepareTempLAEOsamples(laeoIdxs):
    allSamples = []
    allVideos = list(laeoIdxs.keys())
    allVideos = sorted(allVideos)

    for videoname in allVideos:
        # Positive samples
        pos = laeoIdxs[videoname]["pos"]
        for ix in range(0, len(pos)):
            frameId = laeoIdxs[videoname]["tracks"][pos[ix][0]][0]
            allSamples.append((videoname, frameId, pos[ix], 1))

        # Negative samples
        neg = laeoIdxs[videoname]["neg"]
        for ix in range(0, len(neg)):
            frameId = laeoIdxs[videoname]["tracks"][neg[ix][0]][0]
            allSamples.append((videoname, frameId, neg[ix], 0))

        # Ambiguous samples
        if "amb" in laeoIdxs[videoname]:
            amb = laeoIdxs[videoname]["amb"]
            for ix in range(0, len(amb)):
                frameId = laeoIdxs[videoname]["tracks"][amb[ix][0]][0]
                allSamples.append((videoname, frameId, amb[ix], 9))

    return allSamples


def mj_genSeqImageForViz(lpair):
    mosaic = lpair[0,]
    for tix in range(1,lpair.shape[0]):
        mosaic = np.concatenate((mosaic, lpair[tix,]), axis=0)

    return mosaic

def mj_applyTransformToPair(pair, img_gen, transf):
    nrows = pair.shape[0]
    ncols = pair.shape[1]

    normalize = pair.max() <= 1.0

    ppA = pair[:, 0:int(ncols/2), ]
    ppA = img_gen.apply_transform(ppA, transf)
    ppB = pair[:, int(ncols/2):ncols, ]
    ppB = img_gen.apply_transform(ppB, transf)
    if transf["flip_horizontal"] == 1:
        newpair = np.concatenate((ppB, ppA), axis=1)
    else:
        newpair = np.concatenate((ppA,ppB),axis=1)

    if normalize:
        newpair /= 255

    return newpair


# ------------------------------------------------------------
def mj_genNegLAEOwithGeom(l_img_pairs, geom, neg_mode):

    temp_dim = l_img_pairs.shape[0]
    side = l_img_pairs.shape[1]

    geom_new = copy.deepcopy(geom)  # It will be needed

    # Check perturbation mode

    if neg_mode == 1:
        # Mirror each part of the pair separately, no geometry is changed
        l_img_new = np.zeros(l_img_pairs.shape, dtype=l_img_pairs.dtype)
        for ix in range(0,temp_dim):
            for half in range(0,2):
                l_img_new[ix, :, half*side:(half+1)*side,] = np.fliplr(l_img_pairs[ix, :, half*side:(half+1)*side,])

    elif neg_mode == 2:
        # Swap head locations (no mirroring)
        l_img_new = np.zeros(l_img_pairs.shape, dtype=l_img_pairs.dtype)
        l_img_new[:, :, 0:side, ] = l_img_pairs[:, :, side:2* side, ]
        l_img_new[:, :, side:2 * side, ] = l_img_pairs[:, :, 0:side, ]

        geom_new[2] = 1.0 / geom_new[2]  # Scale is inverted

    elif neg_mode == 3:
        # Change just geometry (i.e. vertical displacement)
        geom_new[1] = 0.65+0.1*np.random.normal()   # Change vertical difference
        geom_new[2] += 0.001 * np.random.normal()   # Very small change in scale

        l_img_new = l_img_pairs

    elif neg_mode == 4:
        # Change just geometry: one on top of the other (dx ~0)
        geom_new[0] = abs(0.05*np.random.normal())   # Change vertical difference
        geom_new[2] += 0.001 * np.random.normal()   # Very small change in scale

        l_img_new = l_img_pairs

    elif neg_mode == 5:
        # Change just geometry: inverted scale + big change
        geom_new[2] = 1.0/geom[2]   # Big change in scale
        if geom_new[2] < 1.25 and geom_new[2] > 0.8:
            if geom_new[2] < 1.0:
                geom_new[2] = geom_new[2] * 0.6
            else:
                geom_new[2] = geom_new[2] * 1.5

        l_img_new = l_img_pairs


    return l_img_new, geom_new


############### MAIN ###############
#def __main__():
if __name__ == '__main__':

    videosdir = "/home/mjmarin/data/databases/avalaeo/videos"
    alldirs = []
    # Find all files in videos dir
    allFiles = os.listdir(videosdir)
    allFiles = sorted(allFiles)  # Make sure files are sorted
    for file in allFiles:
        if file.endswith(".mp4"):
            filename, file_extension = os.path.splitext(file)
            alldirs.append(filename.lower())

    annheaddir = "/home/mjmarin/databases/avalaeo/annotations_head/"
    annpairdir = "/home/mjmarin/databases/avalaeo/annotations_pair/"
    annframedir = "/home/mjmarin/databases/avalaeo/annotations_frame/"

    laeoAnnots = {alldirs[0]:[]}

    for videoname in alldirs:
        # videoname = "got01"
        print(videoname)

        samples = mj_computeTempLAEOpairsFromVideo(videoname, annheaddir, annpairdir, annframedir)
        laeoAnnots[videoname] = samples

    # Save data to disk
    outdir = "/home/mjmarin/databases/avalaeo"
    outfile = os.path.join(outdir, "pairstemp_filtered.h5")
    hf = h5py.File(outfile, 'w')

    for videoname in alldirs:
        hf.create_group(videoname)
        gpos = hf.create_group(videoname + "/pos")
        gpos.create_dataset("pairs", data=laeoAnnots[videoname]["pos"])
        gneg = hf.create_group(videoname + "/neg")
        gneg.create_dataset("pairs", data=laeoAnnots[videoname]["neg"])
        gamb = hf.create_group(videoname + "/amb")
        gamb.create_dataset("pairs", data=laeoAnnots[videoname]["amb"])
        gtracks = hf.create_group(videoname + "/tracks")
        for trix in range(0,len(laeoAnnots[videoname]["tracks"])):
           gtracks.create_dataset("track{}".format(trix), data=laeoAnnots[videoname]["tracks"][trix])

    hf.close()

    # Test reading
    l0 = mj_readTempLAEOpairsFromH5("/home/mjmarin/databases/avalaeo/pairstemp_filtered.h5")
    print(len(l0))

    print("End of program")