# Demo for training a LAEO-Net++ model on AVA-LAEO

We describe here how to train a model with already preprocessed data.

## Data preparation

Download AVA-LAEO preprocessed samples: 
   [train](https://ucordoba-my.sharepoint.com/:u:/g/personal/in1majim_uco_es/EVkU-xdW2aBFttdOS7mh_P0B4hocxDIu2emydcjIzgXi7Q?e=qgLfg7), 
   [val](https://ucordoba-my.sharepoint.com/:u:/g/personal/in1majim_uco_es/ETVdi2M2H2hNrMPres9saQgB-NLCQjVdMGcsRaZz_fBJfw?e=YE4J5D)

The training code assumes that the previous tar files are not unpacked. 
   So, just place them in a couple of directories and update the code to point to those directories.   
For example, 
in file `ln_train3DconvModelGeomMBranchCropMapAVA.py`, there is a variable named `tardir` that can be customized. 
Currently, it points to `/experiments/ava/preprocdata/w10_mw10/train` for training data. Similarly, use `val` for the 
validation data.

### AFLW head crops
We are not sure whether we can distribute the preprocessed set of [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) 
heads used in our experiments. Therefore, such data
is not currently available. 


## Run the training script

From the root directory of the project, run the following command:
```bash
python train/ln_train3DconvModelGeomMBranchCropMapAVA.py -g 0.50 -e 60 -l 0.0001 
-d 0.2 -n 1 -s 32 -a 1 -z 0 -b 8 -f 1 -F 0 -k 0.3 -c 0 -R 2 -L 1 -m 0 -u 0 
--useframecrop=0 --usegeometry=0 --initfilefm="" --usemap=1 --mapwindowlen=10 -S 0 
--DEBUG=0 --trainuco=0 --testuco=1 --infix=_ss64jitter 
-w ./model-init-ssheadbranch.hdf5 --useself64=1
```

Parameters:
* `-S`: use synthetic data? Since AFLW preprocessed data is not released, we set it to `0` in the example. 

