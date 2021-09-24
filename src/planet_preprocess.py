import os, UNIT
import numpy as np

def func():
    rootdir = "D:\\licong\\GrassIsolines\\data\\Planet\\Xizang2018_psscene4band_analytic_sr"
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename[-6:] != "SR.tif":
                continue
            path_filename = os.path.join(parent, filename)
            img, proj, geot = UNIT.img2numpy(path_filename, geoinfo=True)
            img = img / 1e4
            img_nir, img_red, img_blue = img[3, :, :], img[2, :, :], img[0, :, :]
            ndvi = (img_nir - img_red) / (img_nir + img_red + 0.001)
            ndvi = np.where(img_red < 0.1, 0, ndvi)
            ndvi = np.where(img_blue > 0.3, 0, ndvi)
            UNIT.numpy2img(os.path.join(rootdir, "NDVI" + filename), ndvi, proj=proj, geot=geot)
            UNIT.numpy2img(os.path.join(rootdir, filename), img, proj=proj, geot=geot)


if __name__ == "__main__":
    func()
