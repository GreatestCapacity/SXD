import cv2 as cv
import numpy as np

data = 'map/'

ds = cv.imread(data + 'disp.pgm', 0).astype('float32') / 8
nocc = cv.imread(data + 'nocc.png', 0).astype('float32')

sad_ds = cv.imread(data + 'sad_disp.pgm', 0).astype('float32') / 8
ssd_ds = cv.imread(data + 'ssd_disp.pgm', 0).astype('float32') / 8
sxd_ds = cv.imread(data + 'sxd_disp.pgm', 0).astype('float32') / 8

sad_err = np.absolute(ds - sad_ds) * nocc
ssd_err = np.absolute(ds - ssd_ds) * nocc
sxd_err = np.absolute(ds - sxd_ds) * nocc
 
print('SAD nocc', np.round(100 * sad_err[sad_err > 255].shape[0] / (sad_err.shape[0] * sad_err.shape[1]), 2))
print('SSD nocc', np.round(100 * ssd_err[ssd_err > 255].shape[0] / (ssd_err.shape[0] * ssd_err.shape[1]), 2))
print('SXD nocc', np.round(100 * sxd_err[sxd_err > 255].shape[0] / (sxd_err.shape[0] * sxd_err.shape[1]), 2))

sad_err = 255 * sad_err / (np.max(sad_err) - np.min(sad_err))
ssd_err = 255 * ssd_err / (np.max(ssd_err) - np.min(ssd_err))
sxd_err = 255 * sxd_err / (np.max(sxd_err) - np.min(sxd_err))

cv.imwrite(data + 'sad_err.pgm', sad_err)
cv.imwrite(data + 'ssd_err.pgm', ssd_err)
cv.imwrite(data + 'sxd_err.pgm', sxd_err)

