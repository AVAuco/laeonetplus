__author__ = "Manuel J Marin-Jimenez"


import os
from scipy.io import loadmat

import numpy as np

class TracksManager(object):
    def __init__(self, filepath, data=None):
        self.filepath = filepath
        self.tracks = []
        self.ntracks = 0
        self.error_loading = False
        self.with_names = False

        if filepath == "" or filepath==None:
            self._loadFromData(data)
        else:
            self._loadFromMat()


    def _loadFromData(self, data):
        if type(data) is list:
            self.tracks = np.asarray(data)
        else:
            self.tracks = data
        self.ntracks = len(data)

    def _loadFromMat(self):
        if os.path.isfile(self.filepath):
            self.tracks = loadmat(self.filepath)["tubes"]
            self.ntracks = self.tracks.shape[0]

            if self.ntracks > 0:
                self.with_names = self.tracks.shape[1] > 2
        else:
            print("WARN: tracks file does not exist :: {}".format(self.filepath))
            self.tracks = None
            self.ntracks = 0
            self.error_loading = True

    def is_valid(self):
        return not self.error_loading

    def getTrack(self, trix):
        assert trix < self.ntracks, "Invalid index"
        return self.tracks[trix,0]

    def getTrackIntervalBBs(self, trix, frame_start, frame_end, strict_mode=True):
        """

        :param trix: track index
        :param frame_start:
        :param frame_end: last frame id (included!)
        :return:
        """

        track = self.getTrack(trix)
        len_ = int(frame_end-frame_start+1)
        idx0 = self.index_of_frame(trix, frame_start)

        len_track = track.shape[0]

        last_pos = idx0+len_-1
        if last_pos >= len_track:
            if strict_mode:
                print("ERROR: requested interval cannot be obtained, too long. Index out of range.")
                exit(-1)
            else:
                last_pos = len_track-1
                len_ = len_track-idx0


        if idx0 < 0:
            print("ERROR: invalid frame number [{:d}] (i.e. not included in target track #{:d})".format(frame_start, trix))
            exit(-1)

        bbs = np.zeros((len_,4), dtype=np.float32)
        for idx in range(idx0, last_pos+1):
            bbs[idx-idx0,] = track[idx, 1:5]

        return bbs

    def getTrackScore(self, trix):
        assert trix < self.ntracks, "Invalid index"
        if type(self.tracks[trix,1]) == np.float32:
            return self.tracks[trix, 1]
        else:
            return self.tracks[trix, 1][0][0]

    def getTrackName(self, trix):
        assert trix < self.ntracks, "Invalid index"
        if self.tracks.shape[1] > 2:
            return self.tracks[trix, 2][0]
        else:
            return ""

    def start(self, trix):
        return int(self.getTrack(trix)[0,0])

    def end(self, trix):
        return int(self.getTrack(trix)[self.len(trix)-1,0])

    def len(self, trix=-1):
        if trix == -1:
            return self.ntracks
        else:
            return self.getTrack(trix).shape[0]

    def __len__(self):
        return self.ntracks

    def index_of_frame(self, trix, frix):
        track = self.getTrack(trix)

        idx = -1

        for ix in range(0,self.len(trix)):
            if frix == track[ix,0]:
                idx = ix
                break

        return idx

    def is_frame_in_track(self, frix, trix):

        if self.index_of_frame(trix, frix) == -1:
            return False
        else:
            return True





