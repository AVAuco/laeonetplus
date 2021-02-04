"""
Util methods for image and annotation files treatment

Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net++: revisiting people Looking At Each Other in videos
IEEE TPAMI, 2021

(c) MJMJ/2021
"""

import os
import numpy as np
import cv2

def parseFileLAEOframe(filename):
    """

    :param filename: annotations at frame level
    :return: the annotations
    """
    labels = -1
    if os.path.exists(filename):
        labels = np.loadtxt(filename, delimiter=" ", usecols=1)
    else:
        print('Error: cannot find file')

    return labels

def parseFileAnnotations(filename):
    """
    Aux method used to parse  bounding box annotations from file
    :param filename: File where bb annotations are stored
    :return Annotations list

    """
    lAnnotations = []
    annotation = []

    file = open(filename, "r")
    for line in file:
        ldata = line.split()
        if len(ldata) == 1:

            if len(annotation) > 0:
                lAnnotations.append(annotation)
            annotation = []
        else:
            if len(ldata) >= 4:
                annotation.append([[int(ldata[0]), int(ldata[1])], [int(ldata[2]), int(ldata[3])]])

    # Add last annotation to list
    if len(annotation) > 0:
        lAnnotations.append(annotation)
    file.close()
    return lAnnotations


def parseFileLAEO(filename):
    """
    Aux method used to parse LAEO annotations from file
    :param filename: File where LAEO annotations are stored
    :return Annotations list

    """
    lAnnotations = []
    annotation = []
    file = open(filename, "r")
    for line in file:
        ldata = line.split()
        if len(ldata) == 1:
            if len(annotation) > 0:
                lAnnotations.append(annotation)
            annotation = []
        else:
            if len(ldata) >= 2:
                annotation.append((int(ldata[0]), int(ldata[1])))

    # Add last annotation to list
    if len(annotation) > 0:
        lAnnotations.append(annotation)
    file.close()

    return lAnnotations


def get_cropped_head(ann_file, image, frame, head_size):
    """
    Aux method used to get cropped resized images from source image
    :param ann_file: File where bounding box annotations are stored
    :param image: Source image to be cropped
    :param frame: Frame or row in ann_file for selecting correct annotation
    :param head_size: Cropped image desired output size
    :return Cropped images list

    """
    crop_images = []
    file = ann_file
    if not os.path.exists(image):
        print("Image file not found: "+image)
        head_ann = []
        return crop_images, head_ann, [1,1]

    image = cv2.imread(image)
    imgsize = image.shape
    head_ann = parseFileAnnotations(file)
    if frame < len(head_ann):
        head_ann = head_ann[frame]
    else:
        print("ERROR: invalid index ({:d}) for head_ann (len={:d}), using file:\n{:s}".format(frame, len(head_ann), ann_file))
        head_ann = []
        return crop_images, head_ann, [1,1]

    for j in range(0, len(head_ann)):
        x = int(head_ann[j][0][0])
        xh = int(head_ann[j][1][0])
        y = int(head_ann[j][0][1])
        yh = int(head_ann[j][1][1])

        dist = max(xh-x, yh-y)

        crop_img0 = image[y:y+dist, x:x+dist]

        if crop_img0.size != 0:
            crop_img0 = cv2.resize(crop_img0, head_size)
        else:
            print("WARN: empty image crop [{:s}]: {}".format(__name__, dist))

        crop_images.append(crop_img0)

    return crop_images, head_ann, imgsize


def create_single_map(head_ann=[], map_size=25):
    """
    Aux method to create head_map from bounding box annotations
    :param head_ann: List of bb annotations
    :param map_size: Size of output head map
    :return Grayscale head_map image

    """
    def gauss2d(x, y, amp, x0, y0, a, b, c):
        inner = a * (x - x0) ** 2
        inner += 2 * b * (x - x0) ** 2 * (y - y0) ** 2
        inner += c * (y - y0) ** 2
        return amp * np.exp(-inner)

    if not head_ann[0][0][1] == -1 or not head_ann[0][1][1] == -1:
        ref = abs(head_ann[0][0][1] - head_ann[0][1][1])
        h_map = np.zeros((map_size, map_size))
        centx = []
        centy = []
        for j in range(len(head_ann)):

            x0 = round(head_ann[j][0][0]/1280 * map_size + (head_ann[j][1][0]/1280 * map_size - head_ann[j][0][0]/1280 * map_size)/2)
            y0 = round(head_ann[j][0][1]/720 * map_size + (head_ann[j][1][1]/720 * map_size - head_ann[j][0][1]/720 * map_size)/2)
            centx.append(x0)
            centy.append(y0)

            heigth = abs(head_ann[j][0][1] - head_ann[j][1][1])
            rel_size = (heigth/ref)
            gauss_value = gauss2d(x0 - 1, y0 - 1, 1, x0, y0, 0.5 / rel_size, 0, 0.5 / rel_size)

            h_map[min(y0 + 1, map_size-1)][x0] = max(h_map[min(y0 + 1, map_size-1)][x0], round(gauss_value, 3))
            h_map[max(y0 - 1, 0)][x0] = max(h_map[max(y0 - 1, 0)][x0], round(gauss_value, 3))
            h_map[y0][min(x0 + 1, map_size-1)] = max(h_map[y0][min(x0 + 1, map_size-1)], round(gauss_value, 3))
            h_map[y0][max(x0 - 1, 0)] = max(h_map[y0][max(x0 - 1, 0)], round(gauss_value, 3))
            h_map[y0][x0] = 1.00

        if 1.00 in h_map[0,:]:
            h_map = np.roll(h_map, 2, axis=0)
        if 1.00 in h_map[1,:]:
            h_map = np.roll(h_map,1,axis=0)

        while not 1.00 in h_map[2,:]:
            h_map = np.roll(h_map,-1,axis=0)

        top_h = np.argmin(centy)
        if centx[int(top_h)] <= map_size/2:
            if 1.00 in h_map[:, 0] and not 1.00 in h_map[:,-1]:
                h_map = np.roll(h_map, 1, axis=1)
            if 1.00 in h_map[:, 1] and not 1.00 in h_map[:,-1]:
                h_map = np.roll(h_map, 1, axis=1)
            while not 1.00 in h_map[:, 2] and not np.any(h_map[:,0] > 0.0):
                h_map = np.roll(h_map, -1, axis=1)
        else:
            if 1.00 in h_map[:, 0]and not 1.00 in h_map[:,0]:
                h_map = np.roll(h_map, -2, axis=1)
            if 1.00 in h_map[:, 1]and not 1.00 in h_map[:,0]:
                h_map = np.roll(h_map, -1, axis=1)
            while not 1.00 in h_map[:, 2] and not np.any(h_map[:,-1] > 0.0) :
                h_map = np.roll(h_map, 1, axis=1)

        h_map = h_map*255
        h_map = h_map[...,np.newaxis]
        return h_map


def laeo_head(frame, head, laeo_ann_file=""):
    """
    Aux method used to return the class label from two images
    :param laeo_ann_file: File where LAEO pair annotations are stored
    :param head: Two sized tuple which represent the two head indexes
    :param frame: Frame or row in ann_file for selecting correct annotation
    :return 1 if the two images are LAEO, otherwise returns 0

    """
    if laeo_ann_file[-14:-6] == 'negative':
        return 0

    if not os.path.exists(laeo_ann_file):
        print("LAEO file not found")
        exit(-1)

    lHead = parseFileLAEO(laeo_ann_file)
    if lHead[frame] == head or lHead[frame] == [(head[0][1], head[0][0])]:
        return 1

    return 0
