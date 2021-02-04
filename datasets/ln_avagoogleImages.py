"""
Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2021
"""

__author__ = "Manuel J Marin-Jimenez"

import ln_laeoImage as LI
import mj_laeoUtils as LU
from mj_miningSamples import mj_detBB2squared
from ln_laeoImage import mj_drawHeadMapFromBBs
from mj_genericUtils import mj_bbarray2bblist

import cv2
import numpy as np

import os
import tarfile
from ln_laeoImage import mj_readImageFromTar, mj_padImageTrack

def mj_trackMat2BBMat(track_mat):
    """

    :param track_mat: numpy array [nframes, 6] --> [frix, 4xcoords, score]
    :return:
    """

    track_len = track_mat.shape[0]
    bb_mat = np.zeros((track_len,4))

    for ix in range(0, track_len):
        bb1 = track_mat[ix, 1:5]
        bb_mat[ix,] = bb1

    return bb_mat

def mj_getFrameBBsPairFromTracks(tracks, trix_pair, init_frame, winlen, framesdir, strict_mode=True,
                                 framepatt="%06d.jpg"):

    lbbs1, lbbs2 = mj_getListBBsmoothFromTracks(tracks, trix_pair, init_frame, winlen, strict_mode=strict_mode)

    nbbs = min(lbbs1.shape[0], lbbs2.shape[0])

    central_frix = int(nbbs/2)
    frame_t = int(init_frame + central_frix)

    imgname = os.path.join(framesdir, framepatt % frame_t)
    img = cv2.imread(imgname)

    imagesize = (img.shape[0], img.shape[1])

    # bb1 = track1[idx1+tix, 1:5]
    bb1 = lbbs1[central_frix,]
    bb1s = mj_detBB2squared(bb1, imagesize)

    bb2 = lbbs2[central_frix,]
    bb2s = mj_detBB2squared(bb2, imagesize)

    return bb1s, bb2s, frame_t, img

def mj_getImagePairSeqFromTracks(tracks, trix_pair, init_frame, winlen, framesdir, targetsize,
                                 mean_sample=[0.0], with_maps=False, mean_map=[0.0],
                                 strict_mode=True, with_other_maps=False, winlenMap=1, framepatt="%06d.jpg"):
    """

    :param tracks:
    :param trix_pair:
    :param init_frame:
    :param winlen:
    :param framesdir:
    :param targetsize:
    :param mean_sample:
    :param with_maps:
    :param mean_map: [64,64,3] (for position #5 in winlen)

    :return: M is [64,64,3] (for position #5 in winlen)
    """
    lbbs1, lbbs2 = mj_getListBBsmoothFromTracks(tracks, trix_pair, init_frame, winlen, strict_mode=strict_mode)

    not_in_pair = []
    lbbs_others = []

    if with_other_maps:
        for trix in range(0, tracks.ntracks):
            if not trix in trix_pair:

                if not tracks.is_frame_in_track(init_frame, trix) or not tracks.is_frame_in_track(init_frame+winlen-1, trix):
                    continue

                not_in_pair.append(trix)
                lbbs_not = tracks.getTrackIntervalBBs(trix, init_frame, init_frame + winlen - 1, strict_mode=strict_mode)
                lbbs_others.append(lbbs_not)

    if with_maps:
        sampleL, sampleR, G, M = mj_getImagePairSeqFromBBseq(lbbs1, lbbs2, init_frame, framesdir, targetsize, winlen,
                                             mean_sample, with_maps, mean_map=mean_map, lbbs_others=lbbs_others,
                                                             winlenMap=winlenMap, framepatt=framepatt)

        return sampleL, sampleR, G, M
    else:
        sampleL, sampleR, G = mj_getImagePairSeqFromBBseq(lbbs1, lbbs2, init_frame, framesdir, targetsize, winlen,
                                             mean_sample, with_maps)

        return sampleL, sampleR, G


def mj_getListBBsmoothFromTracks(tracks, trix_pair, init_t, winlen, strict_mode=True):
    trix1 = trix_pair[0]
    trix2 = trix_pair[1]

    lbbs1_ = tracks.getTrackIntervalBBs(trix1, init_t, init_t+winlen-1, strict_mode=strict_mode)
    lbbs1 = LI.mj_smoothBBannotations(np.transpose(lbbs1_), 5); lbbs1 = np.transpose(lbbs1)

    lbbs2_ = tracks.getTrackIntervalBBs(trix2, init_t, init_t+winlen-1, strict_mode=strict_mode)
    lbbs2 = LI.mj_smoothBBannotations(np.transpose(lbbs2_), 5); lbbs2 = np.transpose(lbbs2)

    return lbbs1, lbbs2


def mj_getImagePairSeqFromBBseqTar(tar, lbbs1, lbbs2, init_t, videoname, pairkey, winlen,
                                mean_sample, with_maps=False, mean_map=[0.0], winlenMap=1):


    initframe = 0  # DEVELOP!!!

    mname = "{:s}_{:04d}_{:s}.jpg".format(videoname, initframe, pairkey)
    mname = mname.replace("/", "_")
    try:
        tarinfo = tar.getmember(mname)
    except:
        tarinfo = tar.getmember("./"+mname)

    img = mj_readImageFromTar(tar, tarinfo)

    ncols = img.shape[1]
    ncols_2 = int(ncols/2)

    sampleL = np.zeros((winlen, ncols_2, ncols_2 ,3))
    sampleR = np.zeros((winlen, ncols_2, ncols_2, 3))
    G = np.zeros((3))
    G[2] = 1.0

    # import pdb; pdb.set_trace()

    # Separate into two head tracks
    for t in range(0, winlen):
        sampleL[t,] = (img[t*ncols_2:(t+1)*ncols_2, 0:ncols_2, ]/255.0) - mean_sample
        sampleR[t,] = (img[t*ncols_2:(t+1)*ncols_2, ncols_2:ncols, ]/255.0) - mean_sample

    # Map
    if with_maps:
        mname = "{:s}_{:04d}_{:s}_map.jpg".format(videoname, initframe, pairkey)
        mname = mname.replace("/", "_")

        try:
            tarinfo = tar.getmember(mname)
        except:
            tarinfo = tar.getmember("./"+mname)

        M = mj_readImageFromTar(tar, tarinfo)
        
        # import pdb; pdb.set_trace()
        
        if winlenMap > 1:
            M3d = np.zeros((winlenMap, M.shape[1], M.shape[1], M.shape[2]))
            for mix in range(0, winlenMap):
                M3d[mix,] = (M[mix*M.shape[1]:(mix+1)*M.shape[1],] -mean_map)/255.0

            M = M3d
        else:
            M = (M - mean_map) / 255.0


    # TODO geometry will not be computed
    if with_maps:
        return sampleL, sampleR, G, M
    else:
        return sampleL, sampleR, G



def mj_getImagePairSeqFromBBseq(lbbs1, lbbs2, init_t, framesdir, targetsize, winlen,
                                mean_sample, with_maps=False, mean_map=[0.0], lbbs_others=[],
                                winlenMap=1, framepatt="%06d.jpg",
                                middle_frame=-1):
    """

    :param lbbs1: numpy array [nframes, 4]
    :param lbbs2: numpy array [nframes, 4]
    :param init_t:
    :param framesdir:
    :param targetsize:
    :param winlen:
    :param mean_sample:
    :param with_maps:
    :param mean_map: [64,64,3] (for position #5 in winlen)

    :return: M is [64,64,3] (for position #5 in winlen)
    """

    nbbs = min(lbbs1.shape[0], lbbs2.shape[0])   # Using this variable for size is more robust than winlen, in cases where shorter tracks are valid

    sampleL = np.zeros((nbbs, targetsize[0], targetsize[1], 3), dtype=np.float64)
    l1 = lbbs1.shape[0]
    l2 = lbbs2.shape[0]
    sampleR = np.zeros((nbbs, targetsize[0], targetsize[1], 3), dtype=np.float64)

    G = 0.0
    mapdims = (64, 64)
    if winlenMap > 1:
        M = np.zeros((nbbs, mapdims[0], mapdims[1], 3), dtype=np.float64)
    else:
        M = None

    m_tix = int(np.floor(nbbs/2))

    # Read image
    for tix in range(0, nbbs):
        frame_t = init_t + tix

        imgname = os.path.join(framesdir, framepatt%frame_t)
        img = cv2.imread(imgname)
        if img is None:
            print("ERROR: cannot read: {:s}".format(imgname))
            exit(-1)

        #bb1 = track1[idx1+tix, 1:5]
        bb1 = lbbs1[tix,]
        imagesize = (img.shape[0], img.shape[1])

        bb1s = mj_detBB2squared(bb1, imagesize)

        #print(bb1s)

        imcrop = img[int(bb1s[1]):int(bb1s[3]), int(bb1s[0]):int(bb1s[2]),]
        imcrop = cv2.resize(imcrop, targetsize)
        imcrop = (imcrop / 255.0) - mean_sample

        # Second head
        bb2 = lbbs2[tix,]
        bb2s = mj_detBB2squared(bb2, imagesize)

        imcrop2 = img[int(bb2s[1]):int(bb2s[3]), int(bb2s[0]):int(bb2s[2]), ]
        imcrop2 = cv2.resize(imcrop2, targetsize)
        imcrop2 = (imcrop2 / 255.0) - mean_sample

        # Join heads
        if bb1s[0] < bb2s[0]:
            #pair = np.concatenate((imcrop, imcrop2), axis=1)
            sampleL[tix,] = imcrop
            sampleR[tix,] = imcrop2

            if with_maps and (tix == m_tix or winlenMap > 1):
                if lbbs_others == []:
                    input_bbs = [mj_bbarray2bblist(bb1s), mj_bbarray2bblist(bb2s)]
                else:
                    input_bbs = [mj_bbarray2bblist(bb1s), mj_bbarray2bblist(bb2s)]
                    for i in range(0, len(lbbs_others)):
                        input_bbs.append(mj_bbarray2bblist(lbbs_others[i][tix,]))

                M_ = mj_drawHeadMapFromBBs(img.shape, input_bbs, (0, 1), target_size=(64, 64))
                M_ = M_-mean_map
                M_ /= 255.0

                if winlenMap > 1:
                    M[tix,] = M_
                else:
                    M = M_


            g = LU.mj_getGeometryPairFromBB(bb1s, bb2s, imagesize)
        else:
            #pair = np.concatenate((imcrop2, imcrop), axis=1)
            sampleL[tix,] = imcrop2
            sampleR[tix,] = imcrop

            if with_maps and (tix == m_tix or winlenMap > 1):
                if lbbs_others == []:
                    input_bbs = [mj_bbarray2bblist(bb2s), mj_bbarray2bblist(bb1s)]
                else:
                    input_bbs = [mj_bbarray2bblist(bb2s), mj_bbarray2bblist(bb1s)]
                    for i in range(0, len(lbbs_others)):
                        input_bbs.append(mj_bbarray2bblist(lbbs_others[i][tix,]))
                M_ = mj_drawHeadMapFromBBs(img.shape, input_bbs, (0, 1), target_size=(64, 64))
                M_ = M_ - mean_map
                M_ /= 255.0

                if winlenMap > 1:
                    M[tix,] = M_
                else:
                    M = M_

            g = LU.mj_getGeometryPairFromBB(bb2s, bb1s, imagesize)

        G = G + g

        # cv2.imshow("Pair", pair)
        # cv2.waitKey(-1)

    # Averaged geometry
    if nbbs > 0:
        G /= nbbs

    if with_maps:
        return sampleL, sampleR, G, M
    else:
        return sampleL, sampleR, G


def mj_getImagePairSeqFromPklTrackTar(tar, tracks, framesdirbase, vidname, pairkey, targetsize=(64,64), winlen=10,
                                   mean_head=[0.0], with_maps=False, mean_map=[0.0], lbbs_others=[], winlenMap=1):


    track_len = tracks[0].shape[0]
    #print("Track length is: {}".format(track_len))
    frix_init = tracks[0][0, 0]

    # Convert into a list of BBs
    lbbs1 = mj_trackMat2BBMat(tracks[0])
    lbbs2 = mj_trackMat2BBMat(tracks[1])

#    if winlen != track_len:
#        print("wl {} vs tl {}".format(winlen, track_len))
    framesdir = os.path.join(framesdirbase, vidname)

    if with_maps:
        sampleL, sampleR, G, M = mj_getImagePairSeqFromBBseqTar(tar, lbbs1, lbbs2, frix_init, vidname, pairkey, winlen,
                                    mean_head, with_maps=with_maps, mean_map=mean_map, winlenMap=winlenMap)

        return sampleL, sampleR, G, M
    else:
        sampleL, sampleR, G = mj_getImagePairSeqFromBBseqTar(tar, lbbs1, lbbs2, frix_init, vidname, pairkey, winlen,
                                                                mean_head, with_maps=with_maps, mean_map=mean_map)

        return sampleL, sampleR, G


def mj_getImagePairSeqFromPklTrack(tracks, framesdirbase, vidname, targetsize=(64,64), winlen=10,
                                   mean_head=[0.0], with_maps=False, mean_map=[0.0], lbbs_others=[],
                                   middle_frame=-1, winlenMap=1):
    """
    To be used with the all-in-one pkl file
    :param tracks: list of (at least) two elements (i.e. tracks)
    :param framesdirbase:
    :param vidname: xxxxx/clipnumber
    :param targetsize:
    :param winlen:
    :param mean_sample:
    :param with_maps:
    :param mean_map:
    :param lbbs_others:
    :return:
    """

    track_len = tracks[0].shape[0]
    #print("Track length is: {}".format(track_len))
    frix_init = tracks[0][0, 0]


    # Check that both tracks start at the same frame
    if tracks[0][0, 0] != tracks[1][0, 0]:
        frix_init = max(tracks[0][0, 0], tracks[1][0, 0])
        l1 = tracks[0].shape[0]
        l2 = tracks[1].shape[0]
        frix_end = min(tracks[0][l1-1, 0], tracks[1][l2-1, 0])

        tlens = int(frix_end-frix_init+1)

        # t1 = np.zeros((tlens, tracks[0].shape[1]), dtype=tracks[0].dtype)
        # t2 = np.zeros((tlens, tracks[0].shape[1]), dtype=tracks[0].dtype)

        start_copying1 = int(frix_init - tracks[0][0, 0])
        start_copying2 = int(frix_init - tracks[1][0, 0])
        t1 = tracks[0][start_copying1:start_copying1+tlens,]
        t2 = tracks[1][start_copying2:start_copying2 + tlens, ]
        tracks = (t1, t2)

    # Convert into a list of BBs
    lbbs1 = mj_trackMat2BBMat(tracks[0])
    lbbs2 = mj_trackMat2BBMat(tracks[1])

    # if winlen != track_len:
    #     print("wl {} vs tl {}".format(winlen, track_len))

    framesdir = os.path.join(framesdirbase, vidname)
    sampleL, sampleR, G, M = mj_getImagePairSeqFromBBseq(lbbs1, lbbs2, frix_init, framesdir, targetsize, winlen,
                                                         mean_head, with_maps, mean_map, lbbs_others,
                                                         middle_frame=middle_frame, winlenMap=winlenMap)

    # Is padding needed?
    if sampleL.shape[0] < winlen and sampleL.shape[0] > 0:
        sampleL = mj_padImageTrack(sampleL, winlen=winlen, mode="shared")
        sampleR = mj_padImageTrack(sampleR, winlen=winlen, mode="shared")

    if with_maps and winlenMap > 1 and M.shape[0] < winlenMap and M.shape[0] > 0:
        M = mj_padImageTrack(M, winlen=winlenMap, mode="shared")

    if with_maps:
        return sampleL, sampleR, G, M
    else:
        return sampleL, sampleR, G


# --------------------- MAIN ------------------------------
if __name__ == '__main__':

    from mj_avagoogleConfig import AvaGoogledb
    from mj_shotsManager import ShotsManager
    from mj_tracksManager import TracksManager
    import os

    avadb = AvaGoogledb(case_wanted="train", basedir="/home/mjmarin/experiments/ava",
                        framesdirbase="/home/mjmarin/databases/ava_google/Frames")

    videoname = "AYebXQ8eUkM"

    clipnumber = "00917"

    shotsfile = os.path.join(avadb.shots_path, videoname, clipnumber+"_shots.txt")

    sm = ShotsManager(shotsfile)

    nshots = sm.nshots

    lTracksInShots = []

    for six in range(0,nshots):
        shot = sm.getShot(six)
        tracksfile =  os.path.join(avadb.head_tracks, videoname,
                                   clipnumber+"_shot{:s}_{:s}.mat".format(shot[0], shot[1]) )
        tm = TracksManager(tracksfile)

        lTracksInShots.append(tm)

        ntracks = tm.ntracks

    # Load head crops from tracks
    from mj_laeoImage import mj_cropFrameByTracks

    framesdir = os.path.join(avadb.frames, videoname, clipnumber)
    winlen = 10
    targetsize = (64,64)

    # Get tracks from shot
    shix = 1
    tracks = lTracksInShots[shix]

    starting_frame = []
    end_frame = []

    for trix in range(0, tracks.ntracks):
        track = tracks.getTrack(trix)
        starting_frame.append(tracks.start(trix))
        end_frame.append(tracks.end(trix))
        print(tracks.len(trix))

    # Selecting two sample tracks
    trix1 = 0
    trix2 = 1

    init_t = max(starting_frame[trix1], starting_frame[trix2])

    end_t = min(end_frame[trix1], end_frame[trix2])

    print(starting_frame)


    #track1 = tracks.getTrack(trix1)

    #idx1 = tracks.index_of_frame(trix1, init_t)

    # Get sample
    lsamplesL = []
    lsamplesR = []

    for init_frame in range(init_t, end_t-winlen+1):
        sampleL, sampleR, G = mj_getImagePairSeqFromTracks(tracks, (trix1, trix2), init_frame, winlen, framesdir, targetsize)
        lsamplesL.append(sampleL)
        lsamplesR.append(sampleR)
        print(G)

        print(sampleL.shape)

    print(len(lsamplesL))
