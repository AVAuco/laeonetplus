"""
Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2021
"""

__author__      = 'Manuel J Marin-Jimenez'
__copyright__   = 'November 2018'

import pickle
import os, sys

from os.path import expanduser
homedir = expanduser("~")

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join( dir_path, '..', '..', 'ava-oxford', 'annotations'))

from ln_avagoogleConfig import AvaGoogledb
#from mj_parse_via_json import mj_load_pairs_laeo_annots

class AvaGoogleLAEOAnnotations(object):

    def __init__(self, annots_file, case_wanted="val",
                 basedir="/scratch/shared/nfs1/mjmarin/experiments/",
                 framesdirbase=""):   #   # **kwargs
        '''

        :param annots_file: pkl file
        :param case_wanted:
        :param basedir:
        :param framesdirbase:
        '''
        self.basedir = basedir

        self.avagoogledb = AvaGoogledb(case_wanted=case_wanted, basedir=basedir, framesdirbase=framesdirbase)

        self.annots_file = annots_file
        self.load_annotations(annots_file)

        self.list_tuples = None

    def load_annotations(self, annots_file):
        with open(annots_file, 'rb') as fid:
            D = pickle.load(fid, encoding='latin1')

        self.D = D

    def is_valid(self, vidkey):
        return not 'omit' in self.D[vidkey].keys()

    def is_valid_pair(self, vidkey, pairkey, windowlen=0, minscore=0):

        is_valid = self.is_valid(vidkey)

        tracks_ = self.get_tracks(vidkey, pairkey)

        if tracks_[0] is None or tracks_[1] is None:  # We need both tracks
            is_valid = False

        elif (tracks_[0].shape[0] < windowlen or tracks_[1].shape[0] < windowlen):
            is_valid = False
        elif minscore > 0:
            scores = self.get_pair_track_scores(vidkey, pairkey)
            
            if scores[0] < minscore or scores[1] < minscore:
               is_valid = False

        return is_valid

    def get_names_valid(self):

        lnames = self.get_names()
        lvalids = []
        for vix in lnames:
            if self.is_valid(vix):
                lvalids.append(vix)

        return lvalids

    def get_names(self):

        self.all_names = list(self.D.keys())
        self.all_names.sort()

        return self.all_names

    def get_annotation(self, key):

        return self.D[key]

    def get_bbs(self, vidkey, pairkey):
        return self.D[vidkey][pairkey]['manuel_dets'][0]

    def get_tracks(self, vidkey, pairkey):
        return self.D[vidkey][pairkey]['manuel_tracks'][0]

    def get_middle_frame(self, vidkey):
        return self.D[vidkey]['middle_frame']

    def get_pair_track_scores(self, vidkey, pairkey):
        return (self.D[vidkey][pairkey]['track_score1'],self.D[vidkey][pairkey]['track_score2'])

    def is_laeo(self, keyvideo):
        laeo = -1

        if self.is_valid(keyvideo):
            laeo = 0
            lpairs = self.get_pairs_names(keyvideo)
            for pix in range(0, len(lpairs)):
                if self.is_laeo_this_pair(keyvideo, lpairs[pix]) == 1:
                    laeo = 1
                    break

        return laeo

    def is_laeo_obsolete(self, key):

        if 'LAEO' in self.D[key].keys():   # TODO : tell this to Vicky
            return self.D[key]['LAEO']
        elif 'laeo' in self.D[key].keys():
            return self.D[key]['laeo']
        else:
            laeos = []
            all_pairs = self.D[key].keys()
            for i in range(0,len(all_pairs)):
                laeos.append( self.is_laeo_this_pair(key, i) )
            return laeos

    def __len__(self):

        return len(self.D)

    def get_pairs_names(self, vidkey):
        l_keys = list(self.D[vidkey].keys())
        l_keys.remove('all_names_idx')
        l_keys.remove('middle_frame')
        pairsnames = l_keys
        return pairsnames

    def get_list_of_tuples(self, minlen=-1, minscore=0, with_ambig=False,
                           reset=False, include_invalid_pairs=False):
        """
        Convenient for training/testing per pair. Only valid pairs will be returned.
        :return: list with tuples (vidkey, pairkey, laeo_label)
        """

        if reset:
            self.list_tuples = None

        if self.list_tuples is None:  # Do it just the first time it is needed
            allSamples = []
            # Prepare a list of tuples
            vidnames = self.get_names()
            nclips = len(vidnames)
            for vix in range(0, nclips):
                vidkey = vidnames[vix]
                if not self.is_valid(vidkey):
                    continue

                pair_names = self.get_pairs_names(vidkey)
                npairs = len(pair_names)
                for pix in range(0, npairs):
#                    tracks = self.get_tracks(vidkey, pair_names[pix])
#                    if tracks[0] is None or tracks[1] is None: # We need both tracks
#                        continue

#                    if minlen > 0 and (tracks[0].shape[0] < minlen or tracks[1].shape[0] < minlen):
#                        continue
                    if not include_invalid_pairs and not self.is_valid_pair(vidkey, pair_names[pix], windowlen=minlen, minscore=minscore):
                        continue
						
                    tup = (vidkey, pair_names[pix], self.is_laeo_this_pair(vidkey, pair_names[pix]))
                    if tup[2] == 9 and not with_ambig:
                        continue

                    allSamples.append(tup)

            self.list_tuples = allSamples

        return self.list_tuples

    def is_laeo_this_pair_old(self, key, pair_idx):

        pairname = "pair_{:02d}".format(pair_idx)
        return self.D[key][pairname]['laeo']

    def is_laeo_this_pair(self, vidkey, pairkey):

        return self.D[vidkey][pairkey]['laeo']

    def __split_filename(self, filename):
        lsp = filename.split('_')
        pairstr = lsp[len(lsp)-1]
        p0 = int(pairstr[1:3])

        return filename[0:len(filename)-8],  p0

    def __group_per_image(self):

        n = len(self.annotations_raw)

        self.annotations = {}

        for i in range(0, n):
            sample = self.annotations_raw[i]
            imname, pair = self.__split_filename(sample[0])

            if imname in self.annotations.keys():
                self.annotations[imname].append((pair, sample[1]))
            else:
                self.annotations[imname] = [(pair, int(sample[1]))]


# ========== MAIN ===========
if __name__ == '__main__':

    if False:
        pklfile = os.path.join(homedir, "databases/ava_google/Annotations/LAEO/", "AVA_LAEO_all_rounds_val.pkl")
    else:
        pklfile = os.path.join(homedir, "databases/ava_google/Annotations/LAEO/round3", "AVA_LAEO_all_rounds1-2-3_val__v2.2_tracks.pkl")
        # pklfile = os.path.join(homedir, "databases/ava_google/Annotations/LAEO/round3", "AVA_LAEO_all_rounds1-2-3_train__v2.2_tracks.pkl")

    # with open(pklfile, 'rb') as fid:
    #     D = pickle.load(fid, encoding='latin1')  # This is SUPER-important for the files generated by Vicky with Python2.x: encoding='latin1'

    avalaeodb = AvaGoogleLAEOAnnotations(pklfile)

    print("{} samples ready!".format(len(avalaeodb)))

    lvids_valids = avalaeodb.get_names_valid()
    print("* {} VALID samples ready!".format(len(lvids_valids)))

    all_vids = avalaeodb.get_names()
    laeopos = 0
    for vix in all_vids:
        labs = avalaeodb.is_laeo(vix)

        if labs == 1:
            laeopos += 1   # This shouldn't happen, but just in case

    print("+ Found {} LAEO pos (frames)".format(laeopos))

    # In terms of pairs
    npairs_tot = 0
    npos_pairs = 0

    # Prepare a list of tuples
    vidnames = avalaeodb.get_names()
    nclips = len(vidnames)
    for vix in range(0, nclips):
        vidkey = vidnames[vix]
        if not avalaeodb.is_valid(vidkey):
            continue

        pair_names = avalaeodb.get_pairs_names(vidkey)
        npairs = len(pair_names)
        for pix in range(0, npairs):
            npairs_tot += 1
            if avalaeodb.is_laeo_this_pair(vidkey, pair_names[pix]) == 1:
                npos_pairs += 1

    print("{} pairs!".format(npairs_tot))

    print("+ Found {} LAEO pos (pairs)".format(npos_pairs))

    # Find a list of ambig cases
    namb_pairs = 0
    l_amb = []
    for vix in range(0, nclips):
        vidkey = vidnames[vix]
        if not avalaeodb.is_valid(vidkey):
            continue

        pair_names = avalaeodb.get_pairs_names(vidkey)
        npairs = len(pair_names)
        for pix in range(0, npairs):
            npairs_tot += 1
            if avalaeodb.is_laeo_this_pair(vidkey, pair_names[pix]) == 9:
                namb_pairs += 1
                tup = (vidkey, pair_names[pix])
                l_amb.append(tup)

    print("Done!")



