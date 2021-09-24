import numpy as np
from skimage.transform import (hough_line, hough_line_peaks, hough_circle,
hough_circle_peaks)
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.data import astronaut
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage import img_as_float
from skimage.morphology import skeletonize
from skimage import data, img_as_float
import matplotlib.pyplot as pylab
from matplotlib import cm
from skimage.filters import sobel, threshold_otsu
from skimage.feature import canny
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries, find_boundaries
import os, UNIT
from osgeo import gdal
from skimage import morphology


os.chdir("D:\licong\GrassIsolines\data\MODIS13Q1_Result")
path_feature = "periodicity.tif"
path_roi_buffer = "D:\licong\GrassIsolines\data\DEM\DangXiong_mountain_buffer.shp"

raster_ds = gdal.Open(path_feature, gdal.GA_ReadOnly)
data_feature = raster_ds.ReadAsArray()
xSize = raster_ds.RasterXSize
ySize = raster_ds.RasterYSize
geotransform = raster_ds.GetGeoTransform()
projection = raster_ds.GetProjection()

# create the target layer (1 band)
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create("", xSize, ySize, bands=1, eType=gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
target_ds.SetGeoTransform(geotransform)
target_ds.SetProjection(projection)

# gdal.Rasterize(target_ds, path_mountain_shp, burnValues=[1])
# mount_roi = target_ds.ReadAsArray()

gdal.Rasterize(target_ds, path_roi_buffer, burnValues=[1])
mask_roi = target_ds.ReadAsArray()
data_feature[np.where(mask_roi == 0)] = np.nan
data_feature[np.where(data_feature == 0)] = np.nan

data_feature_255, func = UNIT.reflectance2rgb(data_feature, bgr=False, func=True)

elevation_map = sobel(data_feature)
fig, axes = pylab.subplots(figsize=(10, 6))
axes.imshow(elevation_map, cmap=pylab.cm.gray, interpolation='nearest')
axes.set_title('elevation map'), axes.axis('off'), pylab.show()

markers = np.zeros_like(data_feature_255)
markers[data_feature_255 < func(0.3)] = 1
markers[data_feature_255 > func(0.7)] = 2
print(np.max(markers), np.min(markers))
fig, axes = pylab.subplots(figsize=(10, 6))
a = axes.imshow(markers, cmap=pylab.cm.hot, interpolation='nearest')
pylab.colorbar(a)
axes.set_title('markers'), axes.axis('off'), pylab.show()

segmentation = morphology.watershed(elevation_map, markers)
UNIT.numpy2img("watershed.tif", segmentation)
# UNIT.numpy2img("thres03.tif", markers)

# pylab.subplots(121)
# pylab.imshow(segmentation, cmap=pylab.cm.gray, interpolation='nearest')
# pylab.title('segmentation'), axes.axis('off'), pylab.show()
#
# pylab.subplots(122)
# pylab.imshow(data_feature_255, cmap=pylab.cm.gray, interpolation='nearest')
# pylab.title('data_feature'), axes.axis('off'), pylab.show()
