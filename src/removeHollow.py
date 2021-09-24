import numpy as np
from osgeo import gdal
import os, UNIT
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy import ndimage


def remove(img, thres):
    img = img.astype('uint8')
    num, labels, status, centroids = cv2.connectedComponentsWithStats(img)
    # remove samll parcels
    for i in tqdm(range(1, num)):
        x, y, xlen, ylen, mountain_size = status[i]
        if mountain_size > thres:
            continue
        # removing
        area_subset = labels[y:y+ylen, x:x+xlen]
        area_subset = np.where(area_subset == i, 0, area_subset)
        labels[y:y+ylen, x:x+xlen] = area_subset
    labels = np.where(labels >= 1, 1, 0).astype('uint8')
    return labels