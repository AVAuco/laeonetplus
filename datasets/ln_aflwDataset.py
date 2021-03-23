"""
Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2021
"""

import socket
import numpy as np
import scipy.io  # For Mat-files
from mj_genericUtils import  mj_isDebugging

hostname = socket.gethostname()

from os.path import expanduser

homedir = expanduser("~")

def mj_getAFLWdata(basedir="", nmax = -1, isTestMode=False, toBGR=True, withHBacks=True):

    if (not isTestMode) and (hostname == "sylar" or mj_isDebugging()):
        if nmax == -1:
            if hostname == "sylar":
                nmax = 250
            else:
                nmax = 2500
        matlab_file_path = homedir+"/databases/aflw/mj_headposes_set3_V6.mat"
        f = scipy.io.loadmat(matlab_file_path)
        fMsamples = f["Msamples"][:,:,:,0:nmax]
        fMposes = f["Mposes"][0:nmax,]

        if withHBacks:
            matlab_file_path = homedir+"/databases/aflw/mj_headposes_hollybacks_V6.mat"
            f = scipy.io.loadmat(matlab_file_path)
            fMsamples = np.concatenate((fMsamples, f["Msamples"]), axis=3)
            fMposes = np.concatenate((fMposes, f["Mposes"]), axis=0)

    else:
        matlab_file_path = homedir+"/databases/aflw/mj_headposes_set0_V6.mat"
        f = scipy.io.loadmat(matlab_file_path)
        fMsamples = f["Msamples"]
        fMposes = f["Mposes"]

        matlab_file_path = homedir+"/databases/aflw/mj_headposes_set2_V6.mat"
        f = scipy.io.loadmat(matlab_file_path)
        fMsamples = np.concatenate((fMsamples, f["Msamples"]), axis=3)
        fMposes = np.concatenate((fMposes, f["Mposes"]), axis=0)

        matlab_file_path = homedir+"/databases/aflw/mj_headposes_set3_V6.mat"
        f = scipy.io.loadmat(matlab_file_path)
        fMsamples = np.concatenate((fMsamples, f["Msamples"]), axis=3)
        fMposes = np.concatenate((fMposes, f["Mposes"]), axis=0)

        if withHBacks:
            matlab_file_path = homedir+"/databases/aflw/mj_headposes_hollybacks_V6.mat"
            f = scipy.io.loadmat(matlab_file_path)
            fMsamples = np.concatenate((fMsamples, f["Msamples"]), axis=3)
            fMposes = np.concatenate((fMposes, f["Mposes"]), axis=0)

    if toBGR:
        import cv2
        nsamples = fMposes.shape[0]
        for i in range(0, nsamples):
            fMsamples[:,:,:,i] = cv2.cvtColor(fMsamples[:,:,:,i], cv2.COLOR_RGB2BGR)


    return fMsamples, fMposes

