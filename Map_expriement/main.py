import cv2 as cv
import numpy as np
import functools as ft
import stereo_alg as sa
import scipy.signal as sg

data = 'map/'
scale = 8

im0 = cv.imread(data + 'im0.ppm', 0).astype('float32')
im1 = cv.imread(data + 'im1.ppm', 0).astype('float32')
patch_size = 21
ndisp = 29

# SAD
cost_maps = sa.sad(im0, im1, patch_size, ndisp)
sad_disp = sa.cost_to_disp(cost_maps)
cv.imwrite(data + 'sad_disp.pgm', sad_disp * scale)

# SSD
cost_maps = sa.ssd(im0, im1, patch_size, ndisp)
ssd_disp = sa.cost_to_disp(cost_maps)
cv.imwrite(data + 'ssd_disp.pgm', ssd_disp * scale)

# SXD
kernel = np.ones((patch_size, patch_size))
s = ft.partial(sg.convolve2d, in2=kernel, mode='same')
t = 12.5
x = lambda a: 255 / (1 + np.exp(-(abs(a)-t)/(0.14 * t)))
d = lambda a, b: a - b
cost_maps = sa.sxd(im0, im1, s, x, d, ndisp)
sxd_disp = sa.cost_to_disp(cost_maps)
cv.imwrite(data + 'sxd_disp.pgm', sxd_disp * scale)

