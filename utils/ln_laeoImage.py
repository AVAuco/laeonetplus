"""
Utils for image manipulation, applied to LAEO problem

(c) MJMJ/2018
"""

import os

import ln_model_utils as MU
import numpy as np
import cv2

import scipy.ndimage.filters as fi

from mj_genericUtils import mj_smooth
import sys

def mj_gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def mj_drawHeadMapFromBBs(img_shape, lBBs, hpair, target_size=(64, 64)):
    hmap = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)

    tam_limit = min(hmap.shape[0], hmap.shape[1])
    # min_x, min_y, max_x, max_y = li.mj_getBBsExtent(lBBs[frix])

    # Check if hpair[0] is on the left
    if lBBs[hpair[0]][0][0] > lBBs[hpair[1]][0][0]:
        hpair = (hpair[1], hpair[0])       # Swap

    for hix in range(0, len(lBBs)):

        bb = lBBs[hix]
        w = bb[1][0] - bb[0][0] + 1
        h = bb[1][1] - bb[0][1] + 1

        tam = int(max(3,max(w, h)))

        if tam > tam_limit or tam < 3:
            print("ERROR: too big circle for head map! :: {}".format(tam))
            break

        K = mj_gkern(tam, round(tam / 4))
        K = K - K.min()
        K = 255.0 * K / K.max()

        # Place in correct location
        if hpair[0] == hix:
            channel = 0
        elif hpair[1] == hix:
            channel = 1
        else:
            channel = 2

        try:
            # Control limits
            jlim = int(max(0, (bb[0][0] + tam) - hmap.shape[1]))
            jstart = int(max(0, bb[0][0]))
            jlim0 = int(abs(min(0, bb[0][0])))

            ilim = int(max(0, (bb[0][1] + tam) - hmap.shape[0]))
            istart = int(max(0, bb[0][1]))
            ilim0 = int(abs(min(0, bb[0][1])))

            hmap[istart:int(bb[0][1]) + tam - ilim, jstart:int(bb[0][0]) + tam -jlim, channel] = K[ilim0:K.shape[0]-ilim, jlim0:K.shape[1]-jlim]
        except:
            print("Incorrect map")

    # Crop map to match cropped frame
    hmap = mj_cropImageFromBBs(hmap, lBBs)

    # Resize map
    hmap = cv2.resize(hmap, target_size)

    return hmap


def mj_drawHeadMapFromGeom(g, target_size=(64, 64)):
        import random

        isz = (128, 128, 3)
        lBBs = []

        x_init = 24
        y_init = 24
        tam_init = 28
        if g[0] > 0.6:
            x_init = 12
            tam_init = 12

        if g[1] < 0:
            y_init = 104
        if abs(g[1]) > 0.6:
            if g[1] < 0:
                y_init = 110
            else:
                y_init = 12

        hs1 = max(10, int(tam_init + random.normalvariate(0, 4)))
        hs2 = np.sqrt((hs1*hs1) / g[2])  # Note that geometry[2] is encoding area of BB not side

        bb1_corner = (int(x_init + random.normalvariate(0, 3)),
                      int(y_init + random.normalvariate(0, 3)))

        bb1 = (bb1_corner, (bb1_corner[0] + hs1 - 1, bb1_corner[1] + hs1 - 1))
        lBBs.append(bb1)

        bb2 = ((int(round(bb1[0][0] + isz[1] * g[0])),
                int(round(bb1[0][1] + isz[0] * g[1]))),
               (int(round(bb1[0][0] + isz[1] * g[0] + hs2 - 1)),
                int(round(bb1[0][1] + isz[0] * g[1] + hs2 - 1))))

        lBBs.append(bb2)

        if random.random() > 0.6:
            bb3 = ((int((bb1[0][0] + bb2[0][0]) / 2), int((bb1[0][1] + bb2[0][1]) / 2)),
                   (int(0.9 * (bb1[1][0] + bb2[1][0]) / 2), int(0.9 * (bb1[1][1] + bb2[1][1]) / 2)))
            lBBs.append(bb3)

        hmap = mj_drawHeadMapFromBBs(isz, lBBs, (0, 1), target_size=target_size)

        return hmap

def mj_smoothBBannotations(track, window_len=5):
    """
    Applies smoothing to the sequence of 4 coordinates components of BBs [x1,y1,x2,y2]'
    :param track: numpy matrix [4, n]
    :param window_len: extent of smoothing function
    :return: smoothed matrix: If window_len > track.shape[1], nothing is done
    """

    if window_len > track.shape[1]:
        return track

    tracksm = np.zeros(track.shape, np.int16)

    for ix in range(0, track.shape[0]):
        x = track[ix,]

        tracksm[ix,] = mj_smooth(x, window_len, window="flat")

    return tracksm


def mj_getBBsExtent(image_bbs):
    max_x = max(image_bbs[0][0][0], image_bbs[0][1][0])
    max_y = max(image_bbs[0][0][1], image_bbs[0][1][1])
    min_x = min(image_bbs[0][0][0], image_bbs[0][1][0])
    min_y = min(image_bbs[0][0][1], image_bbs[0][1][1])
    for i in range(1, len(image_bbs)):
        max_x = max(max_x, image_bbs[i][0][0], image_bbs[i][1][0])
        max_y = max(max_y, image_bbs[i][0][1], image_bbs[i][1][1])
        min_x = min(min_x, image_bbs[i][0][0], image_bbs[i][1][0])
        min_y = min(min_y, image_bbs[i][0][1], image_bbs[i][1][1])

    return int(min_x), int(min_y), int(max_x), int(max_y)


def mj_cropImageFromBBs(img, image_bbs):
    """
    Given a list of BBs, crops a image window in such a way that the BBs are contained. Black rows are added if needed.
    :param img: input image
    :param image_bbs: list of BBs --> format: image_bbs[ix] == ((x1,y1),(x2,y2))
    :return: cropped image
    """

    # max_x = max(image_bbs[0][0][0], image_bbs[0][1][0])
    # max_y = max(image_bbs[0][0][1], image_bbs[0][1][1])
    # min_x = min(image_bbs[0][0][0], image_bbs[0][1][0])
    # min_y = min(image_bbs[0][0][1], image_bbs[0][1][1])
    # for i in range(1, len(image_bbs)):
    #     max_x = max(max_x, image_bbs[i][0][0], image_bbs[i][1][0])
    #     max_y = max(max_y, image_bbs[i][0][1], image_bbs[i][1][1])
    #     min_x = min(min_x, image_bbs[i][0][0], image_bbs[i][1][0])
    #     min_y = min(min_y, image_bbs[i][0][1], image_bbs[i][1][1])

    min_x, min_y, max_x, max_y = mj_getBBsExtent(image_bbs)

    #img = cv2.imread(images[j])
    # crop_img = img[max(min_y-20,0):min(max_y+20,img.shape[0]), max(min_x-20,0):min(max_x+20,img.shape[1])]
    crop_img = img[0:img.shape[0],
               max(min_x - 20, 0):min(max_x + 20, img.shape[1])]
    crop_height = crop_img.shape[0]
    crop_width = crop_img.shape[1]
    size = max(crop_width, crop_height)

    void_img = np.zeros((size, size, 3), np.uint8)
    height = void_img.shape[0]
    width = void_img.shape[1]

    y_crop_center_dist = round(crop_height / 2)
    y_center_dist = round(height / 2)

    x_crop_center_dist = round(crop_width / 2)
    x_center_dist = round(width / 2)

    y_center = y_center_dist - y_crop_center_dist
    x_center = x_center_dist - x_crop_center_dist

    void_img[y_center:y_center + crop_img.shape[0], x_center:x_center + crop_img.shape[1]] = crop_img

    return void_img


def mj_cropFrameByTracks(framesdir, lBBs, track1, track2, winlen, targetsize):
    """
    Extracts a set of cropped frames given two BB tracks.
    :param framesdir:
    :param lBBs: list of BBs related to the tracks
    :param track1:
    :param track2:
    :param winlen: length of the output sequence
    :param targetsize: tuple for resizing the outputs. E.g. (128, 128)
    :return: a numpy array of size (winlen, targetsize[0], targetsize[1], 3)
    """

    # Smooth tracks
    lBBs1 = np.zeros((4, winlen))
    lBBs2 = np.zeros((4, winlen))

    for t_ in range(0,winlen):
        t = track1[0]+t_
        image_bbs = (lBBs[t][track1[1 + t_]], lBBs[t][track2[1 + t_]])
        lBBs1[0,t_] = image_bbs[0][0][0]
        lBBs1[1, t_] = image_bbs[0][0][1]
        lBBs1[2, t_] = image_bbs[0][1][0]
        lBBs1[3, t_] = image_bbs[0][1][1]

        lBBs2[0,t_] = image_bbs[1][0][0]
        lBBs2[1, t_] = image_bbs[1][0][1]
        lBBs2[2, t_] = image_bbs[1][1][0]
        lBBs2[3, t_] = image_bbs[1][1][1]

    lBBs1s = mj_smoothBBannotations(lBBs1)
    lBBs2s = mj_smoothBBannotations(lBBs2)

    # Get images
    lCrops = np.zeros((winlen, targetsize[0], targetsize[1], 3), np.uint8)
    for t_ in range(0,winlen):
        t = track1[0]+t_
        imgname = os.path.join(framesdir, "{:06d}.jpg".format(t))
        img = cv2.imread(imgname)

        if img is None:
            print("- Error with image {:s}".format(imgname))
            continue

        # Prepare BBs
        bb1 = ((lBBs1s[0,t_], lBBs1s[1,t_]), (lBBs1s[2,t_],lBBs1s[3,t_]))
        bb2 = ((lBBs2s[0,t_], lBBs2s[1,t_]), (lBBs2s[2,t_],lBBs2s[3,t_]))

        # Do cropping
        #image_bbs = (lBBs[t][track1[1+t_]], lBBs[t][track2[1+t_]])
        image_bbs = (bb1, bb2)
        imgcrop = mj_cropImageFromBBs(img, image_bbs)

        imgcrop = cv2.resize(imgcrop, targetsize)
        lCrops[t_,] = imgcrop

    return lCrops


def mj_cropFrameMapsByTracks(framesdir, lBBs, track1, track2, winlen, targetsize, targetsize_map):
    """
    Extracts a set of cropped frames given two BB tracks.
    :param framesdir:
    :param lBBs: list of BBs related to the tracks
    :param track1:
    :param track2:
    :param winlen: length of the output sequence
    :param targetsize: tuple for resizing the outputs. E.g. (128, 128)
    :return: a numpy array of size (winlen, targetsize[0], targetsize[1], 3)
    """

    init_frix = track1[0]

    nframes = winlen
    lCrops = np.zeros((winlen, targetsize[0], targetsize[1], 3), np.uint8)
    for frix_ in range(0, nframes):
        frix = init_frix + frix_
        framename = os.path.join(framesdir, "{:06d}.jpg".format(frix))
        if not os.path.isfile(framename):
            continue
        img = cv2.imread(framename)

        img_crop = mj_cropImageFromBBs(img, lBBs[frix])

        # Resize crop
        img_crop = cv2.resize(img_crop, targetsize)

        lCrops[frix_,] = img_crop

    # # Smooth tracks
    # lBBs1 = np.zeros((4, winlen))
    # lBBs2 = np.zeros((4, winlen))
    #
    # for t_ in range(0,winlen):
    #     t = track1[0]+t_
    #     image_bbs = (lBBs[t][track1[1 + t_]], lBBs[t][track2[1 + t_]])
    #     lBBs1[0,t_] = image_bbs[0][0][0]
    #     lBBs1[1, t_] = image_bbs[0][0][1]
    #     lBBs1[2, t_] = image_bbs[0][1][0]
    #     lBBs1[3, t_] = image_bbs[0][1][1]
    #
    #     lBBs2[0,t_] = image_bbs[1][0][0]
    #     lBBs2[1, t_] = image_bbs[1][0][1]
    #     lBBs2[2, t_] = image_bbs[1][1][0]
    #     lBBs2[3, t_] = image_bbs[1][1][1]
    #
    # lBBs1s = mj_smoothBBannotations(lBBs1)
    # lBBs2s = mj_smoothBBannotations(lBBs2)

    # Get images

    lMaps = np.zeros((winlen, targetsize_map[0], targetsize_map[1], 3), np.uint8)
    for t_ in range(0,winlen):
        t = track1[0]+t_
        imgname = os.path.join(framesdir, "{:06d}.jpg".format(t))
        img = cv2.imread(imgname)

        if img is None:
            print("- Error with image {:s}".format(imgname))
            continue

        # # Prepare BBs
        # bb1 = ((lBBs1s[0,t_], lBBs1s[1,t_]), (lBBs1s[2,t_],lBBs1s[3,t_]))
        # bb2 = ((lBBs2s[0,t_], lBBs2s[1,t_]), (lBBs2s[2,t_],lBBs2s[3,t_]))
        #
        # # Do cropping
        # #image_bbs = (lBBs[t][track1[1+t_]], lBBs[t][track2[1+t_]])
        # image_bbs = (bb1, bb2)
        # imgcrop = mj_cropImageFromBBs(img, image_bbs)
        #
        # imgcrop = cv2.resize(imgcrop, targetsize)
        # lCrops[t_,] = imgcrop

        hpair = (track1[1 + t_], track2[1 + t_])
        hmap = mj_drawHeadMapFromBBs(img.shape, lBBs[t], hpair, targetsize_map)
        lMaps[t_,] = hmap

        #print(hmap.shape)


    return lMaps, lCrops


def mj_padImageTrack(itrack, winlen, mode="shared"):
    """

    :param itrack: array [templen, nrows, ncols, nchannels]
    :param winlen: target temporal length (usually greater than templen)
    :return: padded image track
    """

    clen = itrack.shape[0]
    dlen = winlen - clen

    if mode == "shared":
        nbefore = int(np.floor(dlen / 2))
    else:
        nbefore = 0

    nafter = dlen - nbefore

    new_track = np.zeros((winlen, itrack.shape[1], itrack.shape[2], itrack.shape[3]), dtype=itrack.dtype)

    # Starting padding
    pos = 0
    for i in range(0, nbefore):
        new_track[i,] = itrack[0,]
        pos += 1

    # Copy central part
    for i in range(0, clen):
        new_track[pos,] = itrack[i,]
        pos += 1

    # Final padding
    for i in range(0, nafter):
        new_track[pos,] = itrack[clen-1,]
        pos += 1

    return new_track


def mj_readImageFromTar(tar, tarmember):
    """
    Very useful to speed up reading lots of images
    :param tar: object already open in read mode
    :param tarmember:
    :return:
    """

    try:	
        c = tar.extractfile(tarmember).read()
    except:
        c = tar.extractfile("./"+tarmember).read()

    if sys.getsizeof(c) > 266:
        #print(sys.getsizeof(c))
        na = np.frombuffer(c, dtype=np.uint8)
        im = cv2.imdecode(na, cv2.IMREAD_COLOR)

        return im
    else:
        return None

# ------------- MAIN -------------
if __name__ == '__main__':

    import mj_laeoUtils as LU

    winlen = 10
    videoname = 'got02'

    # Define image dir
    framesdirbase = "/home/mjmarin/databases/avalaeo/frames_per_video"
    framesdir = os.path.join(framesdirbase, videoname)

    # Define tracks data
    print("* Loading tracks...")
    laeoIdxs = LU.mj_readTempLAEOpairsFromH5("/home/mjmarin/databases/avalaeo/pairstemp.h5")

    bbdirbase = "/home/mjmarin/databases/avalaeo/annotations_head/"

    if False:
        # Load BBs
        bbfile = os.path.join(bbdirbase, videoname+"_heads.txt")
        lBBs = MU.parseFileAnnotations(bbfile)

        vtracks = laeoIdxs[videoname]
        pair = vtracks["pos"][10]

        track1 = vtracks["tracks"][pair[0]]
        track2 = vtracks["tracks"][pair[1]]

        # Define time interval
        lcrops = mj_cropFrameByTracks(framesdir, lBBs, track1, track2, winlen, (128,128))

        cropstack = LU.mj_genSeqImageForViz(lcrops)

        outdir = "/home/mjmarin/databases/avalaeo/frame_crops"
        outimagename = os.path.join(outdir, "delme.jpg")

        cv2.imwrite(outimagename, cropstack, [int(cv2.IMWRITE_JPEG_QUALITY), 97])

    else:
        print('* Organising track data...')
        allSamples = LU.mj_prepareTempLAEOsamples(laeoIdxs)

        nsamples = len(allSamples)

        for six in range(0, nsamples):
            if six % 100 == 0:
                print(" Image {:d} of {:}".format(six, nsamples))

            videoname = allSamples[six][0]
            initframe = allSamples[six][1]
            pair = allSamples[six][2]
            label = allSamples[six][3]

            # Load BBs
            bbfile = os.path.join(bbdirbase, videoname + "_heads.txt")
            lBBs = MU.parseFileAnnotations(bbfile)

            vtracks = laeoIdxs[videoname]

            track1 = vtracks["tracks"][pair[0]]
            track2 = vtracks["tracks"][pair[1]]

            framesdir = os.path.join(framesdirbase, videoname)

            # Define time interval
            lcrops = mj_cropFrameByTracks(framesdir, lBBs, track1, track2, winlen, (128, 128))

            cropstack = LU.mj_genSeqImageForViz(lcrops)

            outdir = "/home/mjmarin/databases/avalaeo/frame_crops"
            # outimagename = os.path.join(outdir, "{:s}_{:04d}_{:02d}_{:02d}_{:d}.jpg".format(videoname,
            #                                                                            initframe,
            #                                                                            pair[0], pair[1],
            #                                                                                 label))

            outimagename = os.path.join(outdir, "{:s}_{:04d}_{:02d}_{:02d}.jpg".format(videoname,
                                                                                       initframe,
                                                                                       pair[0], pair[1]))


            cv2.imwrite(outimagename, cropstack, [int(cv2.IMWRITE_JPEG_QUALITY), 97]) # TODO: enable me!

    print("End of program")
