import numpy as np
from osgeo import gdal
import os, UNIT


def __modis_vi_format(vi_type, year, doy):
    doy = str(doy).zfill(3)
    filename = "MOD13Q1.006__250m_16_days_{}_doy{}{}_aid0001.tif".format(vi_type, year, doy)
    return filename


def __modis_qa_format(year, doy):
    doy = str(doy).zfill(3)
    filename = "MOD13Q1.006__250m_16_days_pixel_reliability_doy{}{}_aid0001.tif".format(year, doy)
    return filename


def __modis_vi_query(vi_type, year):
    doys = np.arange(1, 366, 16)
    filenames = [__modis_vi_format(vi_type, year, doy) for doy in doys]
    return np.array(filenames)


def __filename_check(filenames):
    exist = np.array([os.path.exists(filename) for filename in filenames])
    locations = np.where(~exist)
    if len(locations[0]):
        filenames_lacking = filenames[locations]
        print(filenames_lacking)
        assert False, "Some files aren't involved."


def modis_stacking(vi_type, year, floder_raw, floder_stacking):
    img_stacking = None
    proj, geot = None, None
    filenames = __modis_vi_query(vi_type, year)
    filenames = [os.path.join(floder_raw, filename) for filename in filenames]
    __filename_check(filenames)
    for i, filename in enumerate(filenames):
        img, proj, geot = UNIT.img2numpy(filename, geoinfo=True)
        if img_stacking is None:
            x, y = img.shape
            img_stacking = np.zeros((len(filenames), x, y), dtype=img.dtype)
        img_stacking[i, :, :] = img[:, :]
    UNIT.numpy2img(os.path.join(floder_stacking, "{}_{}.tif".format(vi_type, year)),
                   img_stacking, proj=proj, geot=geot)
    pass


if __name__ == "__main__":
    floder_modis_raw = "D:\licong\GrassIsolines\data\MODIS13Q1"
    floder_modis_stacking = "D:\licong\GrassIsolines\data\MODIS13Q1_Stacking"
    for year in range(2010, 2020):
        modis_stacking("EVI", year, floder_modis_raw, floder_modis_stacking)
    pass