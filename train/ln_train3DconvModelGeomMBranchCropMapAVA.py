"""
Demo code for training a LAEO-Net++ model on AVA-LAEO dataset

DISCLAIMER: note that this is not production code, just research code.
Many experiments were done during the review period, therefore, there
might be pieces of code that are not useful anymore. Hence, the code
is not fully cleaned.

Command-line example:
python train/ln_train3DconvModelGeomMBranchCropMapAVA.py -g 0.43 -e 60 -l 0.0001 \
-d 0.2 -n 1 -s 32 -a 1 -z 0 -b 8 -f 1 -F 0 -k 0.3 -c 0 -R 2 -L 1 -m 0 -u 0 \
--useframecrop=0 --usegeometry=0 --initfilefm="" --usemap=1 --mapwindowlen=10 -S 1 \
--DEBUG=0 --trainuco=0 --testuco=1 --infix=_ss64jitter \
-w ./model-init-ssheadbranch.hdf5 --useself64=1

# exper_pt1_bs8_lr04p10f02_dr0.2_mwl10_ihn03_wh_nfc_nge_er02_cd_hl2_teuco_hs064ss_ss64jitter_nd1_ds32_ax1

Reference:
 MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
 LAEO-Net++: revisiting people Looking At Each Other in videos
 IEEE TPAMI, 2021

(c) MJMJ/2019-2021
"""

__author__ = "Manuel J Marin-Jimenez"

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import sys, getopt
import os
import os.path as osp
from os.path import expanduser
homedir = expanduser("~")
traindir = os.path.dirname(os.path.abspath(__file__))

DEBUG__=False

# Add custom directories with source code
sys.path.insert(0, traindir)
sys.path.insert(0, osp.join(traindir, "../datasets"))
sys.path.insert(0, osp.join(traindir, "../utils"))

import copy

from ln_dataGeneratorAVALAEO import DataGeneratorAVALAEO
# from ln_avagoogleConfig import AvaGoogledb
from ln_avagoogleLAEOAnnotations import AvaGoogleLAEOAnnotations

import ln_netUtils as NU
from ln_aflwDataset import mj_getAFLWdata
from sklearn.metrics import roc_auc_score, roc_curve
import random

# from joblib import Parallel, delayed
# import multiprocessing
import cv2

from sklearn.metrics import average_precision_score, precision_recall_curve

def ln_fnTrain3DconvModelGeomMBranchCropMapAVA(outdirbase, gpuRate=0.75, initialLR = 0.0001, epcs=50, dropoutval = 0.5,
                              partitionLAEO=1, ndense_top=2, densesize_top=64, augmentation_x=1,
                              batchnorm=False, batchsize=16, submean=False,
                              optimizer="Adam",
                              miningHardNegatives=False, miningHardNegativesInternal=False,
                              freezehead=False,
                              freezecrop=True, epoch4real=5,
                              kappaH=0.8, combineDataSrc=True, realWfactor=1.0, usel2=False,
                              initfileFM="", initfileHG="", useVal4training=False,
                              headweights="", withMap=True,
                              withFCrop=True, withGeom=True, custominfix="",
                              lr_update=False, lr_factor=0.2, lr_patience=5, lr_check_from_epoch=10, lr_min=1e-07,
                              epoch_forced=-1, finetuning=False, windowLen = 10,
                              withSyntData = True, useself64=False, windowLenMap=1,
                                               avalaeodir = "data",
                                              initmodel="", trainOnUCO=False, testOnUCO=False):
    '''

    :param outdirbase:
    :param gpuRate:
    :param initialLR:
    :param epcs:
    :param dropoutval:
    :param partitionLAEO:
    :param ndense_top:
    :param densesize_top:
    :param augmentation_x:
    :param batchnorm:
    :param batchsize:
    :param headweights:
    :param submean:
    :param miningHardNegatives:
    :param freezehead:
    :param kappaH: Percentile for taking the hard negatives
    :return:
    '''

    import numpy as np
    # import math
    from time import time
    import os
    import socket
    hostname = socket.gethostname()
    if hostname == "sylar-msi":
       os.environ["CUDA_VISIBLE_DEVICES"]= ""    #""-1"

    import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

    from tensorflow.keras import optimizers
    from tensorflow.keras.models import load_model
    import tensorflow.keras.backend as K

    import deepdish as dd

    from ln_laeoNets import mj_genNetHeadsGeoCropMap
    from ln_dataGeneratorLAEOhgfm import DataGeneratorLAEOhgfm
    from ln_dataGeneratorLAEOsyn import DataGeneratorLAEOsyn
    from ln_trainingUtils import mj_plotTrainingHistoryWithLoss

    import mj_laeoUtils as LU

    if hostname == "sylar-msi":
       gpu_rate = 0.45 # CHANGE ME!!!
    else:
       gpu_rate = gpuRate #0.75  # CHANGE ME!!!
    theSEED = 1330

    # Tensorflow config
    tf.compat.v1.set_random_seed(theSEED)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_rate
    config.gpu_options.visible_device_list = "" #"0"
    #set_session(tf.Session(config=config))
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # for reproducibility
    np.random.seed(theSEED)

    # Self-supervised branch of Sophia and Olivia' style
    if useself64:
        headsize = 64
        submean = False

    # Some parameters
    max_kappaH = 0.8

    if miningHardNegatives and withFCrop:
        print("ERROR: hard negatives mining cannot be used if frame-crop branch is enabled!")
        exit(-1)


    if initfileFM != "" or initfileHG != "" or headweights !="":
        print("* Model branches will be initialized from pretrained ones...")
    else:
        print("* Whole model will be trained from scratch...")

    if initmodel == "":
        model = mj_genNetHeadsGeoCropMap(windowLen, densesize_top=densesize_top, ndense_top=ndense_top,
                                        dropoutval=dropoutval, batchnorm=batchnorm, usel2=usel2, initfileFM=initfileFM,
                                        initfileHG=initfileHG, initfileHead=headweights, useMap=withMap,
                                        freezehead=freezehead, freezecrop=freezecrop, useFCrop=withFCrop,
                                        useGeom=withGeom, useself64=useself64, windowLenMap=windowLenMap)
                                        #initmodel="")
    else:
        model = load_model(initmodel)
        print("=============================================================")
        print("* Whole model initialized from: {}".format(initmodel))
        print("=============================================================")

        if freezehead:
            print("- Freezing head branch...")
            br01 = model.get_layer('sequential_1')
            for lix in br01.layers:
                lix.trainable = False

        if withMap and freezecrop:
            print("- Freezing map branch...")
            brmap = model.get_layer('sequential_2')
            for lix in brmap.layers:
                lix.trainable = False

    infix = ""
    optimfun = optimizers.Adam(lr=initialLR)
    if optimizer != "Adam":
        infix = "_op" + optimizer
        if optimizer == "SGD":
            optimfun = optimizers.SGD(lr=initialLR, momentum=0.9, decay=1e-05)
        elif optimizer == "AMSGrad":
            optimfun = optimizers.Adam(lr=initialLR, amsgrad=True)
        else:
            optimfun = eval("optimizers."+optimizer+"(lr=initialLR)")

    # Compile model
    model.compile(optimizer=optimfun, loss='binary_crossentropy',
                  metrics=['accuracy'])

    case_wanted = 'train'

    # Train the model
    # --------------------------------------
    # Prepare data
    useUCO = trainOnUCO or testOnUCO
    if useUCO:
        print('* Loading UCO data from file...')
        laeoIdxs = LU.mj_readTempLAEOpairsFromH5(homedir+"/databases/avalaeo/pairstemp.h5")

        print('* Organising data...')
        allSamples_uco = LU.mj_prepareTempLAEOsamples(laeoIdxs)

    # Data from AVA-LAEO
    print('* Info: AVA-LAEO dir is '+avalaeodir)
    print('* Loading AVA data from file...')
    # avasubdir = "databases/ava_google/Annotations/LAEO/round3/"
    # pklfile = os.path.join(homedir, avasubdir, 'AVA_LAEO_all_rounds1-2-3_' + case_wanted + '__v2.2_tracks.pkl')
    pklfile = os.path.join(avalaeodir, 'AVA_LAEO_all_rounds1-2-3_train__v2.2_tracks.pkl')
    #
    avalaeo_annots = AvaGoogleLAEOAnnotations(pklfile)
    # allSamples = avalaeo_annots.get_list_of_tuples(minlen=7)
    allSamples = dd.io.load("./data/avalaeo_samplist_train.h5")

    npairs = len(allSamples)
    print(npairs)

    # Same for the validation partition of AVA
    print('* Loading AVA (validation) data from file...')
    pklfile_val = os.path.join(avalaeodir, 'AVA_LAEO_all_rounds1-2-3_val__v2.2_tracks.pkl')
    #
    avalaeo_annots_val = AvaGoogleLAEOAnnotations(pklfile_val)
    # allSamples_val = avalaeo_annots_val.get_list_of_tuples(minlen=7)
    allSamples_val = dd.io.load("./data/avalaeo_samplist_val.h5")

    npairs_val = len(allSamples_val)
    print(npairs_val)

    # Load head data from Matlab file
    if withSyntData:
        fMsamples, fMposes = mj_getAFLWdata(toBGR=True)
    else:
        fMsamples = []
        fMposes = []

    if hostname == "sylar-msi":
        nvalidationSamples = 50
    else:
        nvalidationSamples = 1000

    if useUCO:
        from ln_ucolaeoConfig import UcoLAEOdb
        #partitionLAEO = 1
        al = UcoLAEOdb(partitionLAEO)

        validationVids, testVids = al.getPartitionConfig()

        npairs_uco = len(allSamples_uco)

        sused = [False] * npairs_uco

        valIdx = []
        for vix in validationVids:
            for pix in range(0,npairs_uco):
                if sused[pix] == False and allSamples_uco[pix][0] == vix:
                    valIdx.append(pix)
                    sused[pix] = True

        testIdx = []
        for vix in testVids:
            for pix in range(0,npairs_uco):
                if sused[pix] == False and allSamples_uco[pix][0] == vix:
                    testIdx.append(pix)
                    sused[pix] = True

        trainIdx = []
        for i in range(0,npairs_uco):
            if sused[i] is False:
                trainIdx.append(i)
                sused[i] = True


        total_samples = npairs_uco
        nsamples_train = len(trainIdx) #math.floor(npairs*0.8)

        # Datasets
        partition = {}
        partition['train'] = trainIdx #range(0,nsamples_train)       # IDs
        partition['validation'] = valIdx #range(nsamples_train, total_samples)
        partition['test'] = testIdx

        if useVal4training:
            print("* Validation partition will be added to training samples")
            partition['train'] = partition['train'] + partition['validation']
            partition['validation'] = partition['test']

        if finetuning:
            print("* Test partition will be added to training samples")
            K.set_value(model.optimizer.lr, lr_min)
            print("- Learning rate has been set to {}".format(lr_min))

            partition['train'] = partition['train'] + partition['test']
            if not useVal4training:    # Also added for fine-tuning
                partition['train'] = partition['train'] + partition['validation']
                partition['validation'] = partition['test']

    if submean:
        meanPath = homedir+"/experiments/deepLAEO/results3DHeadPose/model001_setX_mean.npy"
        meanSample = np.load(meanPath)
        meanSampleH = cv2.cvtColor(meanSample.astype(np.float32), cv2.COLOR_RGB2BGR) # Compatilibity with OpenCV: BGR
    else:
        meanSampleH = [0.0]

    meanfile = os.path.join(homedir, "experiments", "deepLAEO", "meanMaps.h5")
    meanSampleFM = dd.io.load(meanfile)

    if windowLenMap > 10:
        from mj_inflateMeans import mj_inflateMeanMat
        meanmap_new = mj_inflateMeanMat(meanSampleFM["meanmap"], windowLenMap)
        meancrop_new = mj_inflateMeanMat(meanSampleFM["meancrop"], windowLenMap)

        meanSampleFM = {'meanmap': meanmap_new, 'meancrop': meancrop_new}

    # Tar file with samples
    suffix = ""
    if windowLenMap > 1:
        suffix = "_mw{:02d}".format(windowLenMap)
    if windowLen != 10: 
        subdirtar = "w{:02d}".format(windowLen)
    else:
        subdirtar = "w10"
    subdirtar = subdirtar + suffix
    tardir = os.path.join(homedir, "experiments/ava/preprocdata/", subdirtar, case_wanted)
    tarname = os.path.join(tardir, "allsamples"+suffix+".tar")
    print("* Info: tar file: "+tarname)

    # Parameters
    params = {'dim': (windowLen,64,2*64),
              'batch_size': batchsize,
              'n_classes': 2,
              'n_channels': 3,
              'shuffle': True,
              'augmentation': True,
              'withFCrop': withFCrop,
              'withGeom': withGeom,
              'withHMap': withMap,
              'splitHeads': True,
              'augmentation_x': augmentation_x,
              'meanSampleH': meanSampleH,
              'meanSampleFM': meanSampleFM,
              'winlenMap': windowLenMap
              }

    params_syn = {'dim': (windowLen, 64,64),
              'batch_size': batchsize,
              'shuffle': True,
              'augmentation': True,
              'augmentation_x': augmentation_x,
              'meanSample': meanSampleH,
              'withGeom': withGeom,
              'withHMap': withMap,
              'meanSampleFM': meanSampleFM,
              'winlenMap': windowLenMap
              }

    params_avalaeo = {'dim': (windowLen,64,2*64),
              'batch_size': batchsize,
              'n_classes': 2,
              'n_channels': 3,
              'shuffle': True,
              'augmentation': True,
              'withFCrop': withFCrop,
              'withGeom': withGeom,
              'withHMap': withMap,
              'splitHeads': True,
              'augmentation_x': augmentation_x,
              'meanSampleH': meanSampleH,
              'meanSampleFM': meanSampleFM,
              'case_wanted': case_wanted,
              'tarpath': tarname,
              'winlenMap': windowLenMap
    }

    # Data generators were here before
    if batchnorm:
        infix = "_bn"

    if windowLen != 10:
        infix = infix + "_wl{:02d}".format(windowLen)
        print("* Info: temporal window is {:d}.".format(windowLen))

    if withMap and windowLenMap > 1:
        infix = infix + "_mwl{:02d}".format(windowLenMap)
        print("* Info: temporal map window is {:d}.".format(windowLenMap))

    if miningHardNegatives:
        infix = infix + "_hn"
        if kappaH != 1:
            infix = infix + "{:02d}".format(int(kappaH*10))

    if miningHardNegativesInternal:
        infix = infix + "_ihn"
        if kappaH != 1:
            infix = infix + "{:02d}".format(int(kappaH*10))

    if useVal4training:
        infix = infix + "_uv"

    if initfileHG != "" or headweights != "":
        infix = infix + "_wh"
        print("* Info: head branch will be initialized from file.")

        if freezehead:
            infix = infix + "f"
            print("* Info: head brach weights will be frozen [not trained].")

    if initfileFM != "":
        infix = infix + "_wc"
        print("* Info: crop branch will be initialized from file.")

        if freezecrop:
            infix = infix + "f"
            print("* Info: crop brach weights will be frozen [not trained].")

    if not withFCrop:
        infix = infix + "_nfc"
        print("* Info: frame crop is disabled.")

    if not withGeom:
        infix = infix + "_nge"
        print("* Info: geometry branch is disabled.")

    if not withSyntData:
        infix = infix + "_nsy"
        print("* Info: synthetic data will NOT be used.")

    if not withMap:
        infix = infix + "_nhm"
        print("* Info: head map branch is disabled.")

    if epoch4real != 5:
        infix = infix + "_er{:02d}".format(epoch4real)

    if combineDataSrc:
        infix = infix + "_cd"
        if realWfactor != 1.0:
            infix = infix + "{:02d}".format(int(realWfactor*10))

    if usel2:
        infix = infix + "_hl2"

    lr_fix = ""
    if lr_update:
        lr_fix = "p{:d}f{:02d}".format(lr_patience, int(lr_factor*10))

    if trainOnUCO:
        infix = infix + "_truco"
        print("* Info: UCO-LAEO training data will be used.")

    if testOnUCO:
        infix = infix + "_teuco"
        print("* Info: UCO-LAEO test data will be used for validation.")

    if useself64:
        infix = infix + "_hs064ss"
        print("* Info: self-supervised head branch is enabled.")

    infix = infix + custominfix

    # Learning rate string
    ddlr = np.ceil(abs(np.log10(initialLR)))
    ddlr_val = int(initialLR * 10 ** ddlr)

    if ddlr_val == 1:
        lr_val_str = "_lr0{:d}".format(int(ddlr))
    else:
        lr_val_str = "_lr{:d}0{:d}".format(ddlr_val, int(ddlr))

    expersubdir = "exper_pt{:d}_bs{:d}{:s}{:s}_dr{:1.1f}{}_nd{:d}_ds{:d}_ax{:d}".format(partitionLAEO,
                    params['batch_size'],
                    lr_val_str, lr_fix, dropoutval, infix, ndense_top, densesize_top,
                    params['augmentation_x'])
    outdir = os.path.join(outdirbase, expersubdir)

    print("* The results will be saved to: "+outdir)
    sys.stdout.flush()

    # Load existing model
    pattern_file = "model-mix-{:04d}.hdf5"
    if finetuning:
        pattern_file = "model-mix-ft-{:04d}.hdf5"

    previous_model = NU.mj_findLatestFileModel(outdir, pattern_file, epoch_max=epoch_forced)
    print(previous_model)
    initepoch = 0
    if previous_model != "":
        pms = previous_model.split("-")
        initepoch = int(pms[len(pms)-1].split(".")[0])
        print("* Info: a previous model was found. Warming up from it...[{:d}]".format(initepoch))
        model = load_model(previous_model)
    else:
        # This is just for the case of finetuning
        pattern_file_basic = "model-mix-{:04d}.hdf5"
        previous_model = NU.mj_findLatestFileModel(outdir, pattern_file_basic, epoch_max=epoch_forced)
        if previous_model != "":
            pms = previous_model.split("-")
            initepoch = int(pms[len(pms) - 1].split(".")[0])
            print("* Info: a previous model was found. Warming up from it...[{:d}]".format(initepoch))
            model = load_model(previous_model)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Generators
    if withSyntData:
        nsamples = fMsamples.shape[3]
    else:
        nsamples = 0

    if trainOnUCO:
        training_generator_uco = DataGeneratorLAEOhgfm(partition['train'], allSamples_uco, laeoIdxs, **params)

    print("Choosing valid training samples...")
    import tarfile
    valid_idx = []
    file_valid_idx = "./data/avalaeo_train_valid.h5"
    if os.path.exists(file_valid_idx):
        valid_idx = dd.io.load(file_valid_idx)
    else:
        # Check what files are included in the TAR
        if not os.path.exists(tarname):
            print("ERROR: cannot find tar file: {}".format(tarname))
            exit(-1)
        tar = tarfile.open(tarname, 'r')

        all_names = tar.getnames()

        for i in range(0, len(allSamples)):
            tup = allSamples[i]
            memname = tup[0].replace("/", "_") + "_0000_" + tup[1] + ".jpg"

            if memname in all_names:
                valid_idx.append(i)
        tar.close()
        
        dd.io.save(file_valid_idx, valid_idx)
    nps = len(valid_idx)
    print("Valid tuples: {}".format(nps))

    tardirval = os.path.join(homedir, "experiments/ava/preprocdata/", subdirtar, "val")
    tarnameval = os.path.join(tardirval, "allsamples"+suffix+".tar")

    # Extract some validation samples from the training ones
    valid_idx_val = copy.deepcopy(valid_idx)

    if useVal4training:  # This is just for a possible ultimate model
        npairs_val = int(nps * 0.001)
    else:
        npairs_val = int(nps * 0.15)
    npairs_train = nps - npairs_val
    partitionAVA = {'train': valid_idx[0: npairs_train],
                    'validation': valid_idx[npairs_train: nps]}

    if finetuning:
        partitionAVA['train'] = partitionAVA['train'] + partitionAVA['validation']

    training_generator = DataGeneratorAVALAEO(avalaeo_annots, partitionAVA['train'], allSamples, **params_avalaeo)

    if withSyntData:
        print("- Preparing synthetic training data...")
        training_generator_syn = DataGeneratorLAEOsyn(fMsamples[:, :, :, nvalidationSamples:nsamples],
                                                  fMposes[nvalidationSamples:nsamples, ], **params_syn)

    params_synVal = params_syn
    params_synVal['shuffle'] = False
    params_synVal['augmentation'] = False

    useRealData = True        # WARNING This value is crucial, it should be TRUE!!! Val-synth not tested!!!

    if testOnUCO:
        paramsVal = params
        paramsVal['shuffle'] = False
        paramsVal['augmentation'] = False

        validation_generator_uco = DataGeneratorLAEOhgfm(partition['validation'], allSamples_uco, laeoIdxs, **paramsVal)

    params_avalaeoVal = copy.deepcopy(params_avalaeo)
    params_avalaeoVal['shuffle'] = False
    params_avalaeoVal['augmentation'] = False
    params_avalaeoVal['tarpath'] = tarname  # tarnameval
    params_avalaeoVal['case_wanted'] = 'train'  # 'val'

    # validation_generator = DataGeneratorAVALAEO(avalaeo_annots, partitionAVA['validation'], allSamples_val, **params_avalaeoVal)
    validation_generator = DataGeneratorAVALAEO(avalaeo_annots, partitionAVA['validation'], allSamples, **params_avalaeoVal)

    print("- Data generators are ready!")
    sys.stdout.flush()

    # from keras.utils import plot_model
    # plot_model(model, to_file=os.path.join(outdir,'model_hgfm.png'))

    tbdir = os.path.join(outdir, "logstb")
    tensorboard = TensorBoard(log_dir=tbdir + "/{}".format(time()), write_images=True)

    if not os.path.exists(os.path.dirname(tbdir)):
        os.makedirs(tbdir)

    checkptr = ModelCheckpoint(filepath=outdir+'/model-mix-{epoch:04d}.hdf5', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    from ln_trainingUtils import mj_lr_scheduler
    from tensorflow.keras.callbacks import LearningRateScheduler

    lr_decay = LearningRateScheduler(mj_lr_scheduler)

    callbacks = [tensorboard, checkptr, lr_decay]

    lr_last_best_val = -1     # To store the latest best validation value
    lr_check_from_epoch = max(lr_check_from_epoch, epoch4real)
    lr_patience_counter = 0

    # Learn parameters
    nepochs = epcs
    print("*** Starting training ***")

    import mj_miningSamples as MS
    import deepdish as dd

    if miningHardNegatives:
        negdatafile = homedir+"/databases/avalaeo/negtracks_fm.h5"

        # Preloading negative samples for mining
        print("* Preloading negative samples for mining...")
        negsamples = dd.io.load(negdatafile)

    history = {'ap': [],
               "m0": [],
               "m1": [],
               "m0r": [],
               "m1r": [],
               'auc':[],
               'lr': []      # Learning rate
    }
    historyfile = os.path.join(outdir, "model-mix-history.h5")
    if os.path.isfile(historyfile):
        history = dd.io.load(historyfile)

    if not withSyntData:
        useRealTrainData = True
        combineDataSrc = False
    else:
        useRealTrainData = False

    hardNegatives = None

    for epoch in range(initepoch+1, nepochs+1):
        print("Epoch {:d}/{:d}".format(epoch, nepochs))

        if epoch >= epoch4real:
            useRealTrainData = True

        if hostname == "sylar":
            nbatches = 4
        else:
            if withSyntData:
                nbatches = min(training_generator_syn.__len__(), training_generator.__len__())
            else:
                nbatches = training_generator.__len__()

        # Metrics
        cumm0 = 0
        cumm1 = 0
        cumm0r = 0
        cumm1r = 0

        total_time_epoch = 0
        nhardnegperbatch = int(np.ceil(batchsize*0.1))   # TODO: this is to be selected

        # Loop on batches        
        for batch_idx in range(0, nbatches):
            #print(batch_idx)

            b_init_time = time()

            if withSyntData:
                # Synthetic data
                Xs, Ys = training_generator_syn.__getitem__(batch_idx)

                # Combining or alternating?
                if not useRealTrainData or not combineDataSrc:
                    etrain = model.train_on_batch(x=Xs, y=Ys)
                    cumm0 += etrain[0]
                    cumm1 += etrain[1]
                else:  # Fake constants just for statistical info
                    cumm0 += 0.1
                    cumm1 += 0.1

            else:  # Fake constants just for statistical info
                cumm0 += 0.1
                cumm1 += 0.1

            # Real data
            if useRealTrainData:
                X, Y = training_generator.__getitem__(batch_idx)

                if not hardNegatives is None:
                    for ii in range(0, len(X)):
                        X[ii] = np.concatenate((X[ii],
                                                hardNegatives[ii] ))

                    Yhn = np.zeros((nhardnegperbatch,2))
                    Yhn[:,0] = 1
                    Y = np.concatenate((Y, Yhn))
                
                etrainR = model.train_on_batch(x=X, y=Y)

                del X
                del Y

                cumm0r += etrainR[0]
                cumm1r += etrainR[1]
            else:  # Copy from synthetic
                cumm0r += etrain[0]
                cumm1r += etrain[1]

            sys.stdout.write(
                "#{:04d}/{:04d} batch: {:.6f} {:.6f} {:.6f} {:.6f}".format(batch_idx+1, nbatches,
                                                                             cumm0/(batch_idx+1), cumm1/(batch_idx+1),
                                                                             cumm0r/(batch_idx+1), cumm1r/(batch_idx+1)))
            sys.stdout.flush()

            # Mine hard negatives per batch, to be used in the next one
            if miningHardNegatives:
                maxhneg = nhardnegperbatch*4   # TODO constant factor

                Xn = []

                # Clear memory of previous iteration
                if 'pairs_1' in locals():
                    del pairs_1

                if 'pairs_2' in locals():
                    del pairs_2

                if 'G' in locals():
                    del G

                if 'fcs' in locals():
                    del fcs

                if 'hmaps' in locals():
                    del hmaps

                pairs_1 = np.zeros((maxhneg, windowLen, 64, 64, 3))
                pairs_2 = np.zeros((maxhneg, windowLen, 64, 64, 3))
                if withGeom:
                    G = np.zeros((maxhneg,3))
                if withFCrop:
                    fcs = np.zeros((maxhneg, windowLen, 128, 128, 3))

                if withMap:
                    if windowLenMap == 1:
                        hmaps = np.zeros((maxhneg, 64, 64, 3))
                    else:
                        hmaps = np.zeros((maxhneg, windowLenMap, 64, 64, 3))

                k = 0
                att = 0
                while k < maxhneg and att < 1000:
                    vix = random.randint(1,29)
                    videoname = "negatives{:02d}".format(vix)
                    if windowLen <= 10:
                        timepos = random.randint(0, 45)
                    else:
                        timepos = random.randint(0, 35)
                    att += 1
                    try:
                        pair, geom, fc, hmap = MS.mj_getNegLAEOpairWithFM(negsamples, videoname, timepos, windowLen,
                                                                          meanSample=meanSampleH, meanSampleFM=meanSampleFM,
                                                                          windowLenMap=windowLenMap)
                    except:
                        continue

                    if pair is None:
                        continue

                    # Pack in format suitable for evaluation
                    if withGeom:
                        G[k,] = geom
                    pairs_1[k,] = pair[:,:,0:64,:]
                    pairs_2[k,] = pair[:,:,64:128,:]
                    if withFCrop:
                        fcs[k,] = fc

                    if withMap:
                        hmaps[k,] = hmap

                    k += 1

                Xn.append(pairs_1)
                Xn.append(pairs_2)
                if withGeom:
                    Xn.append(G)
                if withFCrop:
                    Xn.append(fcs)
                if withMap:
                    Xn.append(hmaps)

                predneg = model.predict(Xn)

                # Sort by positive class score: descending
                prix = np.argsort(-predneg[:, 1])

                keepK = nhardnegperbatch                       # TODO: parametrize this value

                initHN = int(np.ceil((1-kappaH) * predneg.shape[0] ))  # Difficulty of samples is controled by Kappa
                endHN = initHN+keepK
                if endHN > predneg.shape[0]:
                    initHN = predneg.shape[0]-keepK-1
                    endHN = initHN+keepK

                hardNegatives = []
                hardNegatives.append(Xn[0][prix[initHN:endHN],])
                hardNegatives.append(Xn[1][prix[initHN:endHN],])
                if withGeom:
                    hardNegatives.append(Xn[2][prix[initHN:endHN],])
                if withFCrop:
                    if len(Xn) > 3:
                        hardNegatives.append(Xn[3][prix[initHN:endHN],])
                    else:
                        hardNegatives.append(Xn[2][prix[initHN:endHN],])

                hardNegatives.append(Xn[len(Xn)-1][prix[initHN:endHN],])

                maxhardnegscore = predneg[prix[0], 1]
                inithardscore = predneg[prix[initHN], 1]

                # Verbose
                if (nbatches - batch_idx) < 2:
                    sys.stdout.write(" Max hard neg: {:.4f} ({:.3f})".format(maxhardnegscore, inithardscore))

            # Mine hard negatives from this batch
            if miningHardNegativesInternal and useRealTrainData:
                X, Y = training_generator.__getitem__(batch_idx)

                the_negs = np.where(Y[:,1] == 0)[0]

                Xneg_ = []
                for ii in range(0,len(X)):
                    xx = np.squeeze(X[ii][the_negs,])
                    Xneg_.append(xx)

                predneg = model.predict(Xneg_)

                # Sort by positive class score: descending
                prix = np.argsort(-predneg[:, 1])

                keepK = nhardnegperbatch  # TODO: parametrize this value

                initHN = int(np.ceil((1 - kappaH) * predneg.shape[0]))  # Difficulty of samples is controled by Kappa
                endHN = initHN + keepK
                if endHN > predneg.shape[0]:
                    initHN = predneg.shape[0] - keepK - 1
                    endHN = initHN + keepK

                hardNegatives = []
                hardNegatives.append(Xneg_[0][prix[initHN:endHN],])
                hardNegatives.append(Xneg_[1][prix[initHN:endHN],])
                if withGeom:
                    hardNegatives.append(Xneg_[2][prix[initHN:endHN],])
                if withFCrop:
                    if len(Xneg_) > 3:
                        hardNegatives.append(Xneg_[3][prix[initHN:endHN],])
                    else:
                        hardNegatives.append(Xneg_[2][prix[initHN:endHN],])

                hardNegatives.append(Xneg_[len(Xneg_) - 1][prix[initHN:endHN],])

                maxhardnegscore = predneg[prix[0], 1]
                inithardscore = predneg[prix[initHN], 1]

                # Verbose
                if (nbatches - batch_idx) < 2:
                    sys.stdout.write(" Max hard neg: {:.4f} ({:.3f})".format(maxhardnegscore, inithardscore))

            b_end_time = time()
            b_time = b_end_time - b_init_time
            total_time_epoch += b_time
            if batch_idx < (nbatches-1):
                sys.stdout.write(" ({:.1f}s / ETA {:.1f}s)     ".format(b_time,
                                                             ((nbatches-batch_idx+1)*total_time_epoch/(batch_idx+1))))
            else:
                sys.stdout.write(" (Total time: {:.1f}s)     ".format(total_time_epoch))
            sys.stdout.write("\r")
            sys.stdout.flush()

        sys.stdout.write("\nEnd of epoch {} - ".format(epoch))
        sys.stdout.flush()

        # Get validation data and predict
        # -------------------------------
        cumvalid = 0
        pred = []
        gtlabs = []
        max_val_samples = 1024   # TODO: look at the constant value!
        max_val_batches = int(np.ceil(max_val_samples / batchsize))

        if testOnUCO:
            val_generator = validation_generator_uco
        else:
            val_generator = validation_generator

        nbatchesval = min(val_generator.__len__(), max_val_batches)
        if hostname == "sylar":
            nbatchesval = 6

        for batch_idx in range(0, nbatchesval):
            X, Y = val_generator.__getitem__(batch_idx)

            pred_ = model.predict(X)

            if batch_idx == 0:
                pred = pred_
                gtlabs = Y
            else:
                pred = np.concatenate((pred, pred_))
                gtlabs = np.concatenate((gtlabs, Y))

        estim_labs_test = [pred[:,1] >= 0.5]
        hits_test = np.array([estim_labs_test == gtlabs[:,1]]).sum()
        acc_test = hits_test / gtlabs.shape[0]
        if testOnUCO:
            sys.stdout.write("[UCO] ")
        else:
            sys.stdout.write("[AVA] ")

        sys.stdout.write("ACC={:.3f} ".format(acc_test))
        ap_test = average_precision_score(y_true=np.argmax(gtlabs, axis=1), y_score=pred[:, 1])
        sys.stdout.write("AP={:.3f} ".format(ap_test))
        auc_test = roc_auc_score(y_true=np.argmax(gtlabs, axis=1), y_score=pred[:, 1])
        sys.stdout.write("AUC={:.3f} ".format(auc_test))

        # Compute statistics about validation samples
        nnegv = gtlabs[:,0].sum() / gtlabs.shape[0]
        nposv = gtlabs[:,1].sum() / gtlabs.shape[0]
        print("* INFO: number of samples used: neg={} vs pos={}".format(nnegv, nposv))

        if lr_update and epoch >= lr_check_from_epoch and len(history["ap"]) > lr_patience:
            if ap_test > lr_last_best_val:
                lr_last_best_val = ap_test
                lr_patience_counter = 0
            else:
                lr_patience_counter += 1

            if lr_patience_counter >= lr_patience:
                old_lr = K.get_value(model.optimizer.lr)
                new_lr = max(lr_min, old_lr * lr_factor)
                K.set_value(model.optimizer.lr, new_lr)
                print("* INFO: learning rate updated to {}".format(new_lr))
                lr_patience_counter = 0

        # Update history
        history["ap"].append(ap_test)
        history["auc"].append(auc_test)
        history["m0r"].append(cumm0r/nbatches)
        history["m1r"].append(cumm1r/nbatches)
        history["m0"].append(cumm0/nbatches)
        history["m1"].append(cumm1/nbatches)
        if "lr" in history.keys():
            history["lr"].append(K.get_value(model.optimizer.lr))
        else:
            history["lr"] = [K.get_value(model.optimizer.lr)]

        # Go for a new epoch
        training_generator.on_epoch_end()
        validation_generator.on_epoch_end()

        # Save model after each epoch
        saveEach = 1
        if epoch % saveEach == 0:
            if finetuning:
                model.save(os.path.join(outdir, "model-mix-ft-{:04d}.hdf5".format(epoch)))
            else:
                model.save(os.path.join(outdir, "model-mix-{:04d}.hdf5".format(epoch)))

            dd.io.save(historyfile, history)
            if finetuning:
                figfile = os.path.join(outdir, "history-ft.pdf")
            else:
                figfile = os.path.join(outdir, "history.pdf")
            mj_plotTrainingHistoryWithLoss(history, figfile)

        # Update Kappa?
        if useRealTrainData and kappaH < max_kappaH and (epoch%2 == 0):
            kappaH = min(kappaH*1.1, max_kappaH)

    print("*** End of training ***")

    # Save final model
    model.save(os.path.join(outdir, "model-final.hdf5"))


# =================================================================================

if __name__ == '__main__':

    # Get command line parameters
    argv = sys.argv[1:]

    epochs = 55
    gpuRate = 0.2
    lrate = 0.0001
    sizedense_top=32
    ndense_top=1
    augmentation_x=1
    batchnorm = False
    batchsize = 8
    dropoutval = 0.5
    initfilew = ""
    submean = True
    miningHardNegatives = False
    miningHardNegativesInternal = False
    freezehead = False
    freezecrop = False
    kappaH = 0.3
    combinedatasrc = False
    realWfactor = 1.0
    epoch4real = 5
    usel2 = True
    useVal4training = False
    useTest4finetuning = False
    windowLen = 10
    withFCrop = False
    withGeom = False
    withSyntData = True
    withMap = True
    windowLenMap = 10
    lr_update = True
    lr_patience = 5
    lr_factor = 0.2
    optimizer = "Adam"
    custominfix = ""
    epoch_forced = -1
    initmodel=""    

    useself64 = True
    testOnUCO = False
    trainOnUCO = False

    initfileFM = ""
    initfileHG = ""

    outdirbase = homedir+"/experiments/deepLAEO/results3DconvGeomMBranchCropMapAVA"
    avalaeodir = homedir+"/databases/AVA2.2/Annotations/LAEO/round3"
    
    opts_short = "hg:e:l:n:s:a:z:b:w:d:m:M:f:F:k:c:C:G:r:R:L:u:o:E:t:S:I:O:"
    opts_long = ["gpurate=","epochs=","lrate=","sizedense=","ndense=",
                                    "augx=","batchnorm=","batchsize=","headweights=",
                                    "dropout=","hardnegatives=","internalnegatives",
                                    "freezehead=", "freezecrop=", "kappah=", "combinedatasrc=",
                                    "useframecrop=","usegeometry=",
                                    "realwfactor=", "epoch4real=",
                                    "usel2=","useval=","infix=",
                                    "initfilefm=", "initfilehg=",
                                    "lrupd=", "lrpat=", "lrfactor=",
                                    "optimizer=", "Epoch=", "testdataft=",
                                    "usesyndata=","windowlen=","usemap=",
                                    "useself64=", "trainuco=", "testuco=",
                                    "mapwindowlen=", "initmodel=", "outdirbase=",
                 "avalaeodir=", "DEBUG="]

    try:
        opts, args = getopt.getopt(argv, opts_short,
                                   opts_long)
    except getopt.GetoptError:
        print("Invalid option: {}".format(getopt.GetoptError.msg))
        print('Usage: {} {}'.format(sys.argv[0], opts_long))
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('{} {}'.format(sys.argv[0], opts_short))
            sys.exit()
        elif opt in ("-g", "--gpurate"):
            gpuRate = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-l", "--lrate"):
            lrate = float(arg)
        elif opt in ("-d", "--dropout"):
            dropoutval = float(arg)
        elif opt in ("-n", "--ndense"):
            ndense_top = int(arg)
        elif opt in ("-s", "--sizedense"):
            sizedense_top = int(arg)
        elif opt in ("-a", "--augx"):
            augmentation_x = int(arg)
        elif opt in ("-z", "--batchnorm"):
            batchnorm = int(arg) > 0
        elif opt in ("-b", "--batchsize"):
            batchsize = int(arg)
        elif opt in ("-w", "--headweights"):
            initfilew = arg
        elif opt in ("-m", "--hardnegatives"):
            miningHardNegatives = int(arg)
        elif opt in ("-M", "--internalnegatives"):
            miningHardNegativesInternal = int(arg)
        elif opt in ("-f", "--freezehead"):
            freezehead = int(arg) > 0
        elif opt in ("-F", "--freezecrop"):
            freezecrop = int(arg) > 0
        elif opt in ("-k", "--kappah"):
            kappaH = float(arg)
        elif opt in ("-c", "--combinedatasrc"):
            combinedatasrc = int(arg) > 0
        elif opt in ("-C", "--useframecrop"):
            withFCrop = int(arg) > 0
        elif opt in ("-G", "--usegeometry"):
            withGeom = int(arg) > 0
        elif opt in ("-S", "--usesyndata"):
            withSyntData = int(arg) > 0
        elif opt in ("-r", "--realwfactor"):
            realWfactor = float(arg)
        elif opt in ("-R", "--epoch4real"):
            epoch4real = int(arg)
        elif opt in ("-L", "--usel2"):
            usel2 = int(arg) > 0
        elif opt in ("-u", "--useval"):
            useVal4training = int(arg) > 0
        elif opt in ("-t", "--testdataft"):
            useTest4finetuning = int(arg) > 0
        elif opt in ("--initfilefm"):
            initfileFM = arg
        elif opt in ("--initfilehg"):
            initfileHG = arg
        elif opt in ("--infix"):
            custominfix = arg
        elif opt in ("--lrupd"):
            lr_update = int(arg) > 0
        elif opt in ("--lrpat"):
            lr_patience = int(arg)
        elif opt in ("--lrfactor"):
            lr_factor = float(arg)
        elif opt in ("-o","--optimizer"):
            optimizer = arg
        elif opt in ("-E","--Epoch"):
            epoch_forced = int(arg)
        elif opt in ("--windowlen"):
            windowLen = int(arg)
        elif opt in ("--mapwindowlen"):
            windowLenMap = int(arg)
        elif opt in ("--usemap"):
            withMap = int(arg) > 0
        elif opt in ("--useself64"):
            useself64 = int(arg) > 0
        elif opt in ("--trainuco"):
            trainOnUCO = int(arg) > 0
        elif opt in ("--testuco"):
            testOnUCO = int(arg) > 0
        elif opt in ("--avalaeodir"):
            avalaeodir = arg
        elif opt in ("--DEBUG"):
            DEBUG__ = int(arg) > 0
        elif opt in ("-I", "--initmodel"):
            initmodel = arg         
        elif opt in ("-O", "--outdirbase"):
            outdirbase = arg            

    # # DEVELOP: export head branch to multiple Python versions
    # print("Loading head model...")
    # json_path = osp.join(traindir, "../models/model-init-ssheadbranch.json")
    # wei_path = osp.join(traindir, "../models/model-init-ssheadbranch_w.hdf5")
    # with open(json_path) as json_file:
    #     json_config = json_file.read()
    # from tensorflow.keras.models import model_from_json
    # modelh = model_from_json(json_config)
    # modelh.load_weights(wei_path)
    # modelh.save('model-init-ssheadbranch_py36.hdf5')

    # Call the main function
    ln_fnTrain3DconvModelGeomMBranchCropMapAVA(outdirbase, gpuRate=gpuRate, epcs=epochs, initialLR=lrate,
                                     ndense_top=ndense_top, densesize_top=sizedense_top,
                                     augmentation_x=augmentation_x,batchnorm=batchnorm,
                                     batchsize=batchsize,
                                     dropoutval=dropoutval, submean=submean,
                                     miningHardNegatives=miningHardNegatives,
                                               miningHardNegativesInternal=miningHardNegativesInternal,
                                     freezehead=freezehead, epoch4real=epoch4real, kappaH=kappaH,
                                     freezecrop= freezecrop,
                                     combineDataSrc=combinedatasrc,realWfactor=realWfactor,
                                     usel2=usel2, initfileFM=initfileFM, initfileHG=initfileHG,
                                     headweights=initfilew,
                                     useVal4training=useVal4training, withFCrop=withFCrop,
                                            withMap=withMap,
                                     withGeom=withGeom, custominfix=custominfix,
                                     lr_update=lr_update, lr_patience=lr_patience, lr_factor=lr_factor,
                                            optimizer=optimizer, epoch_forced=epoch_forced, finetuning=useTest4finetuning,
                                            windowLen=windowLen, withSyntData=withSyntData, windowLenMap=windowLenMap,
                                            useself64=useself64, avalaeodir=avalaeodir,
                                               trainOnUCO=trainOnUCO, testOnUCO=testOnUCO,
                                            initmodel=initmodel)
