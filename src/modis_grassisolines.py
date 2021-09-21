import numpy as np
from osgeo import gdal
import os, UNIT
from scipy import ndimage
from scipy import stats
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


def greenness_index(img):
    years, band, xlen, ylen = img.shape
    greenness_index = np.mean(img[:, 7:16, :, :], axis=(0, 1)) - np.mean(img[:, :7, :, :], axis=(0, 1))
    return greenness_index


def evi_composite():
    floder_modis_stacking = "D:\licong\GrassIsolines\data\MODIS13Q1_Stacking"
    floder_modis_result = "D:\licong\GrassIsolines\data\MODIS13Q1_Result"
    evi_multi_year = None
    for i, year in enumerate(range(2010, 2020)):
        evi = UNIT.img2numpy(os.path.join(floder_modis_stacking, "EVI_{}.tif".format(year)))
        if evi_multi_year is None:
            evi_multi_year = np.zeros((10, 23, evi.shape[1], evi.shape[2]), dtype=evi.dtype)
        evi_multi_year[i, :, :, :] = evi
    return evi_multi_year


def bare_extraction():
    pass


def smooth_index(img):
    years, band, xlen, ylen = img.shape
    img = img.reshape((-1, xlen, ylen))
    img = (img - np.mean(img, axis=0)) / np.std(img, axis=0)
    img_median = np.copy(img)
    for i in tqdm(range(xlen)):
        for j in range(ylen):
            img_median[:, i, j] = ndimage.median_filter(img[:, i, j], size=3)
    return - np.sqrt(np.mean(((img - img_median) ** 2), axis=0))


def periodicity_index(img):
    evi_composite = np.mean(img, axis=0)
    xlen, ylen = evi_composite.shape[1], evi_composite.shape[2]
    r_img = np.zeros((xlen, ylen))
    for i in tqdm(range(xlen)):
        for j in range(ylen):
            if len(np.unique(img[0, :, i, j])) <= 1:
                continue
            for year in range(10):
                r2 = stats.pearsonr(evi_composite[:, i, j], img[year, :, i, j])[0] ** 2
                r_img[i, j] += r2
            r_img[i, j] = r_img[i, j] / 10
    return r_img


def coverage_index():
    floder_modis_stacking = "D:\licong\GrassIsolines\data\MODIS13Q1_Stacking"
    floder_modis_result = "D:\licong\GrassIsolines\data\MODIS13Q1_Result"
    evi_multi_year = None
    proj, geot = None, None
    for i, year in enumerate(range(2010, 2020)):
        evi, proj, geot = UNIT.img2numpy(os.path.join(floder_modis_stacking, "EVI_{}.tif".format(year)), geoinfo=True)
        if evi_multi_year is None:
            evi_multi_year = np.zeros((10, 23, evi.shape[1], evi.shape[2]), dtype=evi.dtype)
        evi_multi_year[i, :, :, :] = evi
    evi_multi_year = evi_multi_year / 10000

    # greenness = greenness_index(evi_multi_year)
    # UNIT.numpy2img(os.path.join(floder_modis_result, "greenness.tif"), greenness)
    smooth = smooth_index(evi_multi_year)
    UNIT.numpy2img(os.path.join(floder_modis_result, "smooth.tif"), smooth)
    #periodicity = periodicity_index(evi_multi_year)
    #UNIT.numpy2img(os.path.join(floder_modis_result, "periodicity.tif"), periodicity)


def slice(img_grey):
    nothing = lambda x: 0
    cv2.namedWindow('res')
    cv2.createTrackbar('min', 'res', 0, 100, nothing)
    cv2.createTrackbar('max', 'res', 0, 100, nothing)
    while (1):
        if cv2.waitKey(1) & 0xFF == 27:
            break
        maxVal = cv2.getTrackbarPos('max', 'res')
        minVal = cv2.getTrackbarPos('min', 'res')
        canny = cv2.Canny(img_grey, 10 * minVal, 10 * maxVal)
        cv2.imshow('res', canny)
    cv2.destroyAllWindows()


def edge_detection(smooth, greenness, periodicity):
    smooth = UNIT.reflectance2rgb(smooth, bgr=False)
    greenness = UNIT.reflectance2rgb(greenness, bgr=False)
    periodicity = UNIT.reflectance2rgb(periodicity, bgr=False)
    # img_grey = ((smooth + greenness + periodicity) / 3).astype("uint8")
    img_grey = periodicity

    # Canny
    slice(img_grey)

    # bilateral
    # bilateral_periodicity = cv2.bilateralFilter(periodicity, 9, 75, 75)

    # OTSU
    re2, th_img2 = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # print(re2)
    # edges = cv2.Canny(img_grey, 100, 200)
    plt.subplot(121), plt.imshow(periodicity, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(th_img2, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    UNIT.numpy2img("ostu_img.tif", th_img2)


if __name__ == "__main__":
    os.chdir("D:\licong\GrassIsolines\data\MODIS13Q1_Result")
    # img = evi_composite()
    # img = smooth_index(img)
    # UNIT.numpy2img("smooth.tif", img)

    s, g, p = UNIT.img2numpy("smooth.tif"), UNIT.img2numpy("greenness.tif"), UNIT.img2numpy("periodicity.tif")
    edge_detection(s, g, p)