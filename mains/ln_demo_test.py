"""
Demo code for testing a trained model

Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2021
"""

__author__ = "Manuel J Marin-Jimenez"

import os, sys, getopt
import numpy as np
import cv2
import os.path as osp

mainsdir = os.path.dirname(os.path.abspath(__file__))

# Add custom directories with source code
sys.path.insert(0, os.path.join(mainsdir, "../utils"))  # CHANGE ME, if needed

homedir = "/home/mjmarin/research/laeonetplus/"

# Add custom directories with source code
sys.path.insert(0, homedir + "datasets")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # CHANGE ME, if needed
gpu_rate = 0.45  # CHANGE ME!!! Set the percentage of GPU you want to use for this process

theSEED = 1330

import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json

# Tensorflow config
# tf.set_random_seed(theSEED)
tf.compat.v1.set_random_seed(theSEED)

# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_rate
config.gpu_options.visible_device_list = ""  # "0"
#set_session(tf.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
# for reproducibility
np.random.seed(theSEED)


# Get command line parameters: could be used to select the input model and/or test file
argv = sys.argv[1:]

winlen = 10            # Fixed: temporal length

# Define path to file containing the model
dataset = "AVA"  # Choose: "AVA", "UCO"
modeldir = homedir + "/models/best{}/".format(dataset.upper())
 # Models trained on UCO or AVA
modelpath = osp.join(modeldir, "model-hmaps-tr{}.hdf5".format(dataset.lower()))
json_path = osp.join(modeldir, "model-hmaps-tr{}_config.json".format(dataset.lower()))
wei_path = osp.join(modeldir, "model-hmaps-tr{}_weights.h5".format(dataset.lower()))

# Load model into memory
 # Option 1) Requires Python 3.5
if False:
    model = load_model(modelpath)
else:
 # Option 2) This should more compatible

    with open(json_path) as json_file:
        json_config = json_file.read()
    model = model_from_json(json_config)
    model.load_weights(wei_path)

model.summary()

if False:
    # This is just for exporting the model
    #cfg = model.get_config()

    json_config = model.to_json()

    with open(json_path, 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk

    model.save_weights(wei_path)

# Just for exporting maps to npy
if False:
    import deepdish as dd
    means_ = dd.io.load("/home/mjmarin/research/laeonetplus/models/meanmaps10.h5")
    mmap = means_["meanmap"]
    np.save("/home/mjmarin/research/laeonetplus/models/meanmaps10.npy", mmap)


# Load test data (already cropped for speed)
imagesdir = homedir +"/data/ava_val_crop/"

# Select the example
# ===========================
# The followings are LAEO
#basename = "om_83F5VwTQ_01187_0000_pair_51_49"
#basename = "covMYDBa5dk_01024_0000_pair_37_35"
#basename = "7T5G0CmwTPo_00936_0000_pair_20_19"
#basename = "914yZXz-iRs_01549_0000_pair_192_194"

# The followings are not LAEO
basename = "914yZXz-iRs_01569_0000_pair_196_195"
#basename = "SCh-ZImnyyk_00902_0000_pair_1_0"

pairspath = os.path.join(imagesdir, basename + ".jpg")
mapspath = os.path.join(imagesdir, basename + "_map.jpg")

imgpairs = cv2.imread(pairspath)
imgmaps = cv2.imread(mapspath)
from mj_inflateMeans import mj_inflateMeanMat
imgmaps = mj_inflateMeanMat(imgmaps, 10)

# cv2.imshow("Pairs", imgpairs)
# cv2.waitKey()

# Load mean head and mean map
# meanpath = homedir+"/models/meanhead.npy"
# meansample = np.load(meanpath)

meanfile = os.path.join(homedir, "models", "meanmaps10.npy")
mean_map5 = np.load(meanfile)
print(mean_map5.max())

# Prepare inputs
ncols = imgpairs.shape[1]
ncols_2 = int(ncols / 2)

sampleL = np.zeros((winlen, ncols_2, ncols_2, 3))
sampleR = np.zeros((winlen, ncols_2, ncols_2, 3))

# Separate into two head tracks
for t in range(0, winlen):
    sampleL[t,] = (imgpairs[t * ncols_2:(t + 1) * ncols_2, 0:ncols_2, ] / 255.0) #- meansample
    sampleR[t,] = (imgpairs[t * ncols_2:(t + 1) * ncols_2, ncols_2:ncols, ] / 255.0) #- meansample

headmapnorm = (imgmaps - mean_map5) / 255.0
# To map track
headmaptrack = np.zeros((winlen, 64, 64, 3))
for t in range(0, winlen):
    headmaptrack[t,] = headmapnorm[t * ncols_2:(t + 1) * ncols_2, 0:ncols_2, ]

# Run inference
X0 = np.expand_dims(sampleL, axis=0)
X1 = np.expand_dims(sampleR, axis=0)

M = np.expand_dims(headmaptrack, axis=0)

X = [X0, X1, M]

prediction = model.predict(X)

print("Probability of LAEO is: {:.2f}%".format(prediction[0][1]*100))

print("End of test.")