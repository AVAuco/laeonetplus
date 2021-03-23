import argparse
import os
import csv
import pickle as pkl
import os.path as osp
from platform import python_version

def str2bool(v):
    """Parses string parameters as boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_csv(in_csv_file, delimiter=' '):
    out_csv = []
    with open(in_csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            out_csv.append(row)

    return out_csv


def load_pkl(in_pkl):
    with open(in_pkl, 'rb') as fid: 
        out_pkl = pkl.load(fid, encoding='latin1')

    return out_pkl


def save_pkl(in_pkl, mydict):
    output = open(in_pkl, 'wb')
    pkl.dump(mydict, output)
    output.close()


def make_if_not_exist(path):
    if not osp.exists(path):
        os.makedirs(path)


def make_if_not_exist_file(file):
    if not osp.isdir(osp.dirname(file)):
        os.system('mkdir -p ' + osp.dirname(file))


def get_detector_version():
    py_version = ''.join(python_version().split('.'))
    model_version = '3.6'
    if int(py_version) < 360:
        model_version = '3.5'
    return model_version


class Paths(object):

    def __init__(self, **kwargs):
        cwd = os.path.join(os.path.dirname(__file__), "..", "data")
        # path of the script that downloads the detection model
        self.down_script = os.path.join(cwd, 'models', 'detector',
                                        'download_model_py{}.sh'.format(get_detector_version()))
        # path of the detection model weights file
        self.det_model = os.path.join(cwd, 'models', 'detector',
                                      'ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py{}.h5'
                                      .format(get_detector_version()))
        # folder with the detections files
        self.out_detections = os.path.join(cwd, 'results', 'dets')
        # where you want to save the tracks
        self.out_tracks = os.path.join(cwd, 'results', 'tracks_outfolder')

        for k in kwargs:
            assert hasattr(self,k), "Unkwown parameter %s" % (k)
            self.__setattr__(k, kwargs[k])

    def detections_matfile(self, videofile):
        return self.out_detections + '/' + videofile + '_processed_th0.2.mat' # name of the file

    def detections_picklefile(self, videofile):
        return self.out_detections + '/' + videofile + '_processed_th0.2.pkl' # name of the file


class DetectionParameters(object):

    def __init__(self, **kwargs):
        self.InputWidth = 512       # input image width
        self.InputHeight = 512      # input image height
        self.ClipBoxes = False      # whether to clip the anchor box coordinates within img boundaries
        # output params
        self.NormalizeCoords = True # whether to output coordinates relative to image size
        self.MinScore = 0.2         # remove detections with scores lower than this threshold
        self.IoUThr = 0.45          # minimum overlap between consecutive frames
        self.NKeep = 200            # keep top-N per frame
        self.NmsKeep = 400          # max number of predictions left after NMS
        for k in kwargs:
            assert hasattr(self, k), "Unknown parameter %s" % k
            self.__setattr__(k, kwargs[k])


class TrackingParameters(object): # class for parameters when linking tubelets transformed into boxes with online linking

    def __init__(self, **kwargs):
        self.KeepDetScore = True # keep detection score 
        self.NKeep = 30          # keep top-N per frame 
        # tracking params
        self.MaxSkip = 5         # maximum number of skip before terminating a tube 
        self.IoUThr = 0.5        # minimum overlap between consecutive frames 
        # output params
        self.MinLength = 9       # remove tracks shorter than this 
        self.MinScore = 0.01     # remove tracks with scores lower than this threshold 
        self.nms3DT = True       # nms3dt on tracks
        self.nms3DT_Thr = 0.3    # threshold for nms 3DT 
        for k in kwargs:
            assert hasattr(self,k), "Unknown parameter %s" % k
            self.__setattr__(k, kwargs[k])
