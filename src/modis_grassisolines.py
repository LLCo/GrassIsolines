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


def greenness_index_years(img):
    years, band, xlen, ylen = img.shape
    greenness_index = np.mean(img[:, 7:16, :, :], axis=(1)) - np.mean(img[:, :7, :, :], axis=(1))
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


def periodicity_index_years(img):
    evi_composite = np.mean(img, axis=0)
    xlen, ylen = evi_composite.shape[1], evi_composite.shape[2]
    r_img = np.zeros((10, xlen, ylen))
    for i in tqdm(range(xlen)):
        for j in range(ylen):
            if len(np.unique(img[0, :, i, j])) <= 1:
                continue
            for year in range(10):
                r2 = stats.pearsonr(evi_composite[:, i, j], img[year, :, i, j])[0] ** 2
                r_img[year, i, j] += r2
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

    # slice(img_grey)
    minVal, maxVal = 15, 32
    img_grey_canny = cv2.Canny(img_grey, 10 * minVal, 10 * maxVal)

    # bilateral
    # bilateral_periodicity = cv2.bilateralFilter(periodicity, 9, 75, 75)

    # OTSU
    # re2, th_img2 = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # print(re2)
    # edges = cv2.Canny(img_grey, 100, 200)
    # plt.subplot(121), plt.imshow(periodicity, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(th_img2, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    UNIT.numpy2img("img_grey_canny.tif", img_grey_canny)


def regionGrowNumpy(gray, img_seeds, thresh, p=8):
    xs, ys = np.where(img_seeds)
    xs, ys = xs.tolist(), ys.tolist()
    seedMark = np.zeros(gray.shape)
    seedMark[xs, ys] = 1

    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    #seeds内无元素时候生长停止
    while len(xs) != 0:
        px, py = xs.pop(0), ys.pop(0)
        for i in range(p):
            tmpX = px + connection[i][0]
            tmpY = py + connection[i][1]
            #检测边界点
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue
            if abs(gray[tmpX, tmpY] - gray[px, py]) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 1
                xs.append(tmpX)
                ys.append(tmpY)
    return seedMark


def regionGrow(gray, seeds, thresh, p=4):
    seedMark = np.zeros(gray.shape)
    #八邻域
    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    #seeds内无元素时候生长停止
    while len(seeds) != 0:
        #栈顶元素出栈
        pt = seeds.pop(0)
        for i in range(p):
            tmpX = pt[0] + connection[i][0]
            tmpY = pt[1] + connection[i][1]
            #检测边界点
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue
            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
    return seedMark


def originalSeed(gray, th):
    ret, thresh = cv2.cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)#二值图，种子区域(不同划分可获得不同种子)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))#3×3结构元
    thresh_copy = thresh.copy() #复制thresh_A到thresh_copy
    thresh_B = np.zeros(gray.shape, np.uint8) #thresh_B大小与A相同，像素值为0
    seeds = [ ] #为了记录种子坐标

    #循环，直到thresh_copy中的像素值全部为0
    while thresh_copy.any():
        Xa_copy, Ya_copy = np.where(thresh_copy > 0) #thresh_A_copy中值为255的像素的坐标
        thresh_B[Xa_copy[0], Ya_copy[0]] = 255 #选取第一个点，并将thresh_B中对应像素值改为255
        #连通分量算法，先对thresh_B进行膨胀，再和thresh执行and操作（取交集）
        for i in range(200):
            dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
            thresh_B = cv2.bitwise_and(thresh, dilation_B)
        #取thresh_B值为255的像素坐标，并将thresh_copy中对应坐标像素值变为0
        Xb, Yb = np.where(thresh_B > 0)
        thresh_copy[Xb, Yb] = 0
        #循环，在thresh_B中只有一个像素点时停止
        while str(thresh_B.tolist()).count("255") > 1:
            thresh_B = cv2.erode(thresh_B,  kernel, iterations=1) #腐蚀操作
        X_seed, Y_seed = np.where(thresh_B > 0) #取处种子坐标
        if X_seed.size > 0 and Y_seed.size > 0:
            seeds.append((X_seed[0], Y_seed[0]))#将种子坐标写入seeds
        thresh_B[Xb, Yb] = 0 #将thresh_B像素值置零
    return seeds


def edge_detection_single(path_feature, path_roi_shp, path_mountain_shp):
    # open the raster layer and get its relevant properties
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

    gdal.Rasterize(target_ds, path_mountain_shp, burnValues=[1])
    mount_roi = target_ds.ReadAsArray()

    gdal.Rasterize(target_ds, path_roi_shp, burnValues=[1])
    mask_roi = target_ds.ReadAsArray()

    data_feature[np.where(mask_roi == 0)] = np.nan
    data_feature[np.where(data_feature == 0)] = np.nan

    data_feature_255, func = UNIT.reflectance2rgb(data_feature, bgr=False, func=True)
    re2, th_img2 = cv2.threshold(data_feature_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    minVal, maxVal = 15, 32
    img_canny = cv2.Canny(data_feature_255, 10 * minVal, 10 * maxVal)

    # mask_edge_canny = cv2.Canny(mask_roi * 255, 10 * minVal, 10 * maxVal)
    # img_canny -= mask_edge_canny

    # seeds = originalSeed(data_feature_255, func(0.3))
    # seeds = np.where(data_feature < 0.2, 1, 0)
    img_region_grow = regionGrowNumpy(data_feature, mount_roi, 0.02)

    UNIT.numpy2img("mount_roi.tif", mount_roi)
    UNIT.numpy2img("img_region_grow.tif", img_region_grow)
    UNIT.numpy2img("img_canny.tif", img_canny)

    plt.subplot(131), plt.imshow(data_feature_255, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_region_grow, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_canny, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def edge_detection_simple(path_feature, path_roi_shp, path_mountain_shp):
    # open the raster layer and get its relevant properties
    raster_ds = gdal.Open(path_feature, gdal.GA_ReadOnly)
    data_feature = raster_ds.ReadAsArray()
    xSize = raster_ds.RasterXSize
    ySize = raster_ds.RasterYSize
    geot = raster_ds.GetGeoTransform()
    proj = raster_ds.GetProjection()

    # create the target layer (1 band)
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create("", xSize, ySize, bands=1, eType=gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
    target_ds.SetGeoTransform(geot)
    target_ds.SetProjection(proj)

    gdal.Rasterize(target_ds, path_roi_shp, burnValues=[1])
    mask_roi = target_ds.ReadAsArray()

    data_feature[np.where(mask_roi == 0)] = np.nan
    data_feature[np.where(data_feature == 0)] = np.nan

    # plt.figure()
    # plt.hist(data_feature.reshape(-1), bins=40, range=(0.1, 0.9), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("bins")
    # plt.ylabel("frequency")
    # plt.show()

    mask = np.where(np.isnan(data_feature), 0, data_feature)
    mask[np.where(data_feature >= 0.7)] = 3
    mask[np.where(data_feature < 0.7)] = 2
    mask[np.where(data_feature < 0.3)] = 1
    UNIT.numpy2img("img_mask.tif", mask)

    potential_area = np.where(mask == 2, 1, 0)
    potential_area = ndimage.median_filter(potential_area, 3)
    potential_area = ndimage.binary_dilation(potential_area, iterations=1).astype('uint8')
    UNIT.numpy2img("potential_area.tif", potential_area, proj=proj, geot=geot)
    UNIT.raster2Polygon("potential_area.tif", "potential_area.shp")

    # plt.subplot(131), plt.imshow(data_feature_255, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(img_region_grow, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(img_canny, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()


if __name__ == "__main__":
    os.chdir("D:\licong\GrassIsolines\data\MODIS13Q1_Result")
    # img = evi_composite()
    # index_years = greenness_index_years(img)
    # for i, year in enumerate(range(2010, 2020)):
    #     UNIT.numpy2img("greenness_{}.tif".format(year), index_years[i, :, :])

    # img = evi_composite()
    # img = smooth_index(img)
    # UNIT.numpy2img("smooth.tif", img)

    # s, g, p = UNIT.img2numpy("smooth.tif"), UNIT.img2numpy("greenness.tif"), UNIT.img2numpy("periodicity.tif")
    # edge_detection(s, g, p)
    edge_detection_simple("periodicity.tif",
                          "D:\licong\GrassIsolines\data\DEM\DangXiong_mountain_buffer.shp",
                          "D:\licong\GrassIsolines\data\DEM\DangXiong_mountain.shp")