import numpy as np


def trackscore(tt): 
    return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))]))

""" 
2D boxes are N*4 numpy arrays (float32) with each row at format <xmin> <ymin> <xmax> <ymax>

3D tracks are Nframes*5 arrays (float32) with each row at format <frame> <xmin>  <ymin> <xmax> <ymax>
"""


def area2d(b):
    """ compute the areas for a set of 2D boxes"""
    return (b[:,2]-b[:,0]+1)*(b[:,3]-b[:,1]+1)


def overlap2d(b1, b2):
    """ compute the overlaps between a set of boxes b1 and 1 box b2 """
    xmin = np.maximum( b1[:,0], b2[:,0] )
    xmax = np.minimum( b1[:,2]+1, b2[:,2]+1)
    width = np.maximum(0, xmax-xmin)
    ymin = np.maximum( b1[:,1], b2[:,1] )
    ymax = np.minimum( b1[:,3]+1, b2[:,3]+1)
    height = np.maximum(0, ymax-ymin)   
    return width*height          


def iou2d(b1, b2):
    """ compute the IoU between a set of boxes b1 and 1 box b2"""
    if b1.ndim == 1: b1 = b1[None,:]
    if b2.ndim == 1: b2 = b2[None,:]
    assert b2.shape[0]==1
    o = overlap2d(b1, b2)
    return o / ( area2d(b1) + area2d(b2) - o ) 


def iou3d(b1, b2):
    """ compute the IoU between two tracks with same temporal extent"""
    assert b1.shape[0]==b2.shape[0], pdb.set_trace()
    assert np.all(b1[:,0]==b2[:,0]), pdb.set_trace()
    o = overlap2d(b1[:,1:5],b2[:,1:5])
    return np.mean( o/(area2d(b1[:,1:5])+area2d(b2[:,1:5])-o) )  
    #return np.mean(np.array([iou2d(b1[i,1:5],b2[i,1:5]) for i in range(b1.shape[0])]))


def iou3dt(b1, b2, spatialonly=False):
    """ compute the spatio-temporal IoU between two tracks"""
    tmin = max(b1[0,0], b2[0,0])
    tmax = min(b1[-1,0], b2[-1,0])
    if tmax<tmin: return 0.0    
    temporal_inter = tmax-tmin+1
    temporal_union = max(b1[-1,0], b2[-1,0]) - min(b1[0,0], b2[0,0]) + 1 
    return iou3d( b1[int(np.where(b1[:,0]==tmin)[0]):int(np.where(b1[:,0]==tmax)[0])+1,:] , b2[int(np.where(b2[:,0]==tmin)[0]):int(np.where(b2[:,0]==tmax)[0])+1,:]  ) * ( 1.0 if spatialonly else (temporal_inter / temporal_union) )


def nms3dt(detections, overlap=0.5): # list of (tube,score)
    if len(detections)==0: return np.array([],dtype=np.int32)
    I = np.argsort([d[1] for d in detections ])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0
    while I.size>0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([ iou3dt(detections[ii][0],detections[i][0]) for ii in I[:-1] ])
        I  = I[np.where(ious<=overlap)[0]]
    return indices[:counter]
