import cv2 as cv
import stereo_alg as sa


def alg_sad_mf_cpu(img1, img2, maxdisp):
    mat1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype('float32')
    mat2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype('float32')

    patch_size = 15

    cost_maps = sa.sad(mat1, mat2, patch_size, maxdisp)
    disp_mat = sa.cost_to_disp(cost_maps)
    disp_mat = cv.medianBlur(disp_mat.astype('float32'), 5)

    return disp_mat
