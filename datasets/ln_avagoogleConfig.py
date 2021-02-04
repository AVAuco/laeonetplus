"""
Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2021
"""

__author__      = 'Manuel J Marin-Jimenez'
__copyright__   = 'August 2018'

import os

class AvaGoogledb(object):

    def __init__(self, case_wanted="val",
                 basedir="/scratch/shared/nfs1/mjmarin/experiments/",
                 framesdirbase=""):   #   # **kwargs
        self.basedir = basedir

        self.detections_path = self.basedir + "/ssd-head-detections/" + case_wanted + "/dets/"
        self.shots_path = self.basedir + "/ssd-head-detections/" + case_wanted + "/shots/"
        self.detforlinking = self.basedir + "/ssd-head-detections/" + case_wanted + "/dets_tracking/"
        self.head_tracks = self.basedir + "/ssd-tracking-heads/" + case_wanted + "/tracks_proc/"

        self.framesdirbase = framesdirbase
        self.frames = os.path.join(self.framesdirbase, case_wanted)

        #
        # for k in kwargs:
        #     assert hasattr(self,k), "Unkwown parameter %s"%(k)
        #     self.__setattr__(k, kwargs[k])


# ========== MAIN ===========
if __name__ == '__main__':


    avadb = AvaGoogledb(case_wanted="train", basedir="/home/mjmarin/experiments/ava")

    print(avadb.head_tracks)


