import cv2 as cv
import stereo_alg_gpu as sag
from numba import cuda


def alg_sad_interp_gpu(img1, img2, maxdisp):
    mat1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype('float32')
    mat2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype('float32')

    mat1_gpu = cuda.to_device(mat1)
    mat2_gpu = cuda.to_device(mat2)

    patch_size = 15

    cost_maps_gpu = sag.sad(mat1_gpu, mat2_gpu, patch_size, maxdisp)
    disp_mat_gpu = sag.cost_to_disp(cost_maps_gpu)
    right_cost_maps_gpu = sag.get_right_cost_maps(cost_maps_gpu)
    right_disp_mat_gpu = sag.cost_to_disp(right_cost_maps_gpu)
    check_mat_gpu = sag.left_right_check(disp_mat_gpu, right_disp_mat_gpu)
    disp_mat_gpu = sag.left_interpolation(disp_mat_gpu, check_mat_gpu)
    disp_mat = disp_mat_gpu.copy_to_host()

    return disp_mat
