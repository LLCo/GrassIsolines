import numpy as np
from osgeo import gdal
import os, UNIT
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy import ndimage
import removeHollow


def slice(img_grey):
    nothing = lambda x: 0
    width = 1200
    height = 800
    cv2.namedWindow('res')
    cv2.resizeWindow('res', int(width * (height - 80) / height), height - 80)
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.moveWindow("res", 1000, 100)

    cv2.createTrackbar('min', 'res', 0, 100, nothing)
    cv2.createTrackbar('max', 'res', 0, 100, nothing)
    cv2.createTrackbar('ats', 'res', 0, 2, nothing)

    while (1):
        if cv2.waitKey(1) & 0xFF == 27:
            break
        maxVal = cv2.getTrackbarPos('max', 'res')
        minVal = cv2.getTrackbarPos('min', 'res')
        atsVal = cv2.getTrackbarPos('ats', 'res')
        atsVal = int(atsVal*2) + 3
        print(atsVal)
        canny = cv2.Canny(img_grey, minVal, maxVal, apertureSize=atsVal)
        cv2.imshow('res', canny)

    cv2.destroyAllWindows()


def __landsat_band_format(prefix, band):
    filename = prefix + "_SR_B{}.TIF".format(band)
    return filename


def __landsat_qa_format(prefix):
    filename = prefix + "_ST_QA.TIF"
    return filename


def __landsat_vi(prefix):
    img_qa, proj, geot = UNIT.img2numpy(__landsat_qa_format(prefix), geoinfo=True)
    path_red = __landsat_band_format(prefix, 4)
    path_nir = __landsat_band_format(prefix, 5)
    path_swir1 = __landsat_band_format(prefix, 6)
    img_red, img_nir, img_swir1 = [UNIT.img2numpy(path) for path in (path_red, path_nir, path_swir1)]
    img_red, img_nir, img_swir1 = img_red * 0.0000275, img_nir * 0.0000275, img_swir1 * 0.0000275
    img_red, img_nir, img_swir1 = [np.where(img > 0, img, 0) for img in (img_red, img_nir, img_swir1)]
    ndvi = (img_nir - img_red) / (img_nir + img_red + 0.01)
    # ndvi = np.where(np.logical_and(ndvi > 0.3, img_red < 0.05), 0, ndvi)
    # ndpi = (img_nir - img_red) / (img_nir + img_red)
    return ndvi, proj, geot


def edge_detection(path_potential_shp, path_landsat_feature, ):

    def get_boundary(label, kernel_size=(3, 3)):
        tlabel = label.astype(np.uint8)
        temp = cv2.Canny(tlabel, 0, 1)
        tlabel = cv2.dilate(
            temp,
            cv2.getStructuringElement(
                cv2.MORPH_CROSS,
                kernel_size),
            iterations=1
        )
        tlabel = tlabel.astype(np.float32)
        tlabel /= 255.
        return tlabel

    raster_ds = gdal.Open(path_landsat_feature, gdal.GA_ReadOnly)
    img_feature = raster_ds.ReadAsArray()
    xSize = raster_ds.RasterXSize
    ySize = raster_ds.RasterYSize
    geot = raster_ds.GetGeoTransform()
    proj = raster_ds.GetProjection()

    # create the target layer (1 band)
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create("", xSize, ySize, bands=1, eType=gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
    target_ds.SetGeoTransform(geot)
    target_ds.SetProjection(proj)
    gdal.Rasterize(target_ds, path_potential_shp, burnValues=[1])
    img_potential = target_ds.ReadAsArray()

    thres = 0.2568
    grass_line = np.zeros((ySize, xSize), dtype='uint8')
    grass_line[np.where(np.logical_and(img_feature < thres, 1))] = 1
    grass_line = removeHollow.remove(grass_line, 200)
    grass_line = ndimage.binary_closing(grass_line).astype('uint8')
    grass_line_canny = cv2.Canny(grass_line, 0, 1, apertureSize=3)
    grass_line_canny = removeHollow.remove(grass_line_canny, 10)

    UNIT.numpy2img("grass_line.tif", grass_line, proj=proj, geot=geot)
    UNIT.numpy2img("grass_line_canny.tif", grass_line_canny, proj=proj, geot=geot)
    grass_line_canny = np.where(img_potential == 1, grass_line_canny, 0)
    UNIT.numpy2img("grass_line_modis_canny.tif", grass_line_canny, proj=proj, geot=geot)

    # slice(labels)

    # grass_line = get_boundary(grass_line)

    # UNIT.numpy2img("img_potential.tif", img_potential, proj=proj, geot=geot)
    # data_feature = img_feature[np.where(img_potential)]
    # plt.figure()
    # plt.hist(data_feature.reshape(-1), bins=100, range=(0.1, 0.5), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("bins")
    # plt.ylabel("frequency")
    # plt.show()


if __name__ == "__main__":
    os.chdir("D:\licong\GrassIsolines\data\Landsat8")
    path_potential_shp = "D:\licong\GrassIsolines\data\MODIS13Q1_Result\potential_area.shp"
    path_landsat_feature = "NDPI_summer_max_2013_2020.tif"
    edge_detection(path_potential_shp, path_landsat_feature)
    # prefixs = os.listdir()
    # for i in range(len(prefixs)):
    #     i = 1
    #     path = prefixs[i] + "/" + prefixs[i]
    #     ndvi, proj, geot = __landsat_vi(path)
    #     UNIT.numpy2img("ndvi_{}.tif".format(prefixs[i]), ndvi, proj=proj, geot=geot)
    #     break


