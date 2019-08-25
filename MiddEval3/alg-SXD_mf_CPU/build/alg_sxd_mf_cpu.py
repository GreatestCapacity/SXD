import cv2 as cv
import numpy as np
import functools as ft
import stereo_alg as sa
import scipy.signal as sg


def alg_sxd_mf_cpu(img1, img2, maxdisp):
    mat1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype('float32')
    mat2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype('float32')

    patch_size = 15
    t = 12.5
    scale = 255

    kernel = np.ones((patch_size, patch_size))
    s = ft.partial(sg.convolve2d, in2=kernel, mode='same')
    x = lambda a: scale / (1 + np.exp(-(abs(a)-t)/(0.14*t)))
    d = lambda a, b: a - b
    cost_maps = sa.sxd(mat1, mat2, s, x, d, maxdisp)
    disp_mat = sa.cost_to_disp(cost_maps)
    disp_mat = cv.medianBlur(disp_mat.astype('float32'), 5)

    return disp_mat
