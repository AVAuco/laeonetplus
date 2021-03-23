"""

Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2021
"""

import tensorflow.keras as K
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import numpy as np

import matplotlib
matplotlib.use('Agg') # We need to set matplotlib not to use the Xwindows backend in order to make it work via ssh. We must do it before importing pyplot
import matplotlib.pyplot as plt # For graphs and images

def mj_lr_scheduler(epoch, lr, nepochs=100, reduction=0.1):
    if epoch%nepochs==0 and epoch!=0:
        lr = lr * reduction

    print("Current learning rate is {:1.8f}".format(lr))
    return lr


def mj_plotTrainingHistory(history, outfile):

    # history["ap"]
    # history["m0r"]
    # history["m1r"]
    # history["m0"]
    # history["m1"]

    nepochs = len(history["ap"])

    plt.figure(figsize=(16, 9))
    #plt.plot(range(1,nepochs+1), history["m1"], 'b-', label='training_acc')
    plt.plot(range(1, nepochs + 1), history["ap"], 'b-^', label='AP')
    plt.plot(range(1, nepochs + 1), history["auc"], 'go-', label='AUC')
    plt.plot(range(1, nepochs + 1), history["m1"], 'r-', label='Acc-syn')
    plt.plot(range(1, nepochs + 1), history["m1r"], 'c-s', label='Acc-real')
    plt.xlabel('Epoch')
    plt.ylabel(' % ')
    plt.legend()
    plt.axis(b=True, visible=True, linestyle='dashed', which='minor')
    plt.grid(True, which="major")
    plt.grid(True, which="minor", linestyle='--', color='b')
    plt.yticks(np.arange(0.5, 1, step=0.05))
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def mj_plotTrainingHistoryWithLoss(history, outfile):

    nepochs = len(history["ap"])

    plt.figure(figsize=(20, 16))

    s1 = plt.subplot(1,2,1)
    #plt.plot(range(1,nepochs+1), history["m1"], 'b-', label='training_acc')
    s1.plot(range(1, nepochs + 1), history["ap"], 'b-^', label='AP')
    s1.plot(range(1, nepochs + 1), history["auc"], 'go-', label='AUC')
    s1.plot(range(1, nepochs + 1), history["m1"], 'r-', label='Acc-syn')
    s1.plot(range(1, nepochs + 1), history["m1r"], 'c-s', label='Acc-real')
    plt.xlabel('Epoch')
    plt.ylabel(' % ')
    s1.legend()
    s1.axis(b=True, visible=True, linestyle='dashed', which='minor')
    s1.grid(True, which="major")
    s1.grid(True, which="minor", linestyle='--', color='b')
    plt.yticks(np.arange(0.5, 1, step=0.05))

    s2 = plt.subplot(1, 2, 2)
    s2.plot(range(1, nepochs + 1), history["m0r"], 'b-o', label='Loss-real')
    plt.xlabel('Epoch')
    plt.ylabel(' Loss ')
    s2.legend()
    s2.axis(b=True, visible=True, linestyle='dashed', which='minor')
    s2.grid(True, which="major")

    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


# --------------------- MAIN ------------------------------
if __name__ == '__main__':
    import numpy as np
    import deepdish as dd

    resdir = "/home/mjmarin/experiments/deepLAEO/results3DconvGeomMBranchPre/exper_pt1_bs16_lr04_dr0.3_hn03_whf_cd100_hl2_nd1_ds32_ax4"
    resfile = os.path.join(resdir, "model-mix-history.h5")

    #history = np.load(resfile)
    history = dd.io.load(resfile)

    outfile = os.path.join(resdir, "history_test.pdf")

    #history = {"m1": [0.5, 0.6, 0.65, 0.69, 0.74, 0.8, 0.83]}
    #mj_plotTrainingHistory(history, outfile)
    mj_plotTrainingHistoryWithLoss(history, outfile)