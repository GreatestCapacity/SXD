import cv2 as cv
import numpy as np
import stereo_alg as sa
import stereo_alg_gpu as sag
from numba import cuda


def alg_wba_gpu(img1, img2, maxdisp):
    mat1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY).astype('float32')
    mat2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).astype('float32')

    patch_size = 9

    mat1_gpu = cuda.to_device(mat1)
    mat2_gpu = cuda.to_device(mat2)
    img1_gpu = cuda.to_device(img1.astype('float32'))

    rad = patch_size // 2

    # left large disparity map
    cost_maps_gpu = sag.sad(mat1_gpu, mat2_gpu, patch_size, maxdisp)
    disp_mat_gpu = sag.cost_to_disp_with_penalty(mat1_gpu, cost_maps_gpu)

    # left small disparity map
    small_window_matching[(50, 50), (20, 20)](disp_mat_gpu, mat1_gpu, mat2_gpu, rad, True)

    # right large disparity map
    right_cost_maps_gpu = sag.get_right_cost_maps(cost_maps_gpu)
    right_disp_mat_gpu = sag.cost_to_disp_with_penalty(mat2_gpu, right_cost_maps_gpu)

    # right small disparity map
    small_window_matching[(50, 50), (20, 20)](right_disp_mat_gpu, mat1_gpu, mat2_gpu, rad, False)

    # left-right consistency check
    check_mat_gpu = sag.left_right_check(disp_mat_gpu, right_disp_mat_gpu)
    disp_mat_gpu = sag.rgb_interpolation(img1_gpu, disp_mat_gpu, check_mat_gpu)

    # Refinement
    img1_lab = cv.cvtColor(img1, cv.COLOR_BGR2Lab).astype('float32')
    img1_lab_gpu = cuda.to_device(img1_lab)
    refinement[(30, 30), (30, 30)](img1_lab_gpu, disp_mat_gpu, 1)

    disp_mat = disp_mat_gpu.copy_to_host()

    return disp_mat


@cuda.jit
def small_window_matching(disp_mat, mat1, mat2, rad, is_left):
    r, c = cuda.grid(2)
    rows, cols = disp_mat.shape
    if r < rows and c < cols:
        is_border = False
        for z in range(-rad, rad):
            if 0 <= c + z - 1 and c + z + 1 < cols:
                is_border = is_border or abs(disp_mat[r, c + z + 1] - disp_mat[r, c + z - 1]) > 2
        if is_border:
            min_cost = np.inf
            min_d = 0
            for i in range(-rad, rad):
                if r + i < 0 or r + i >= rows:
                    continue
                for j in range(-rad, rad):
                    if c + j < 0 or c + j >= cols:
                        continue

                    d = int(disp_mat[r+i, c+j])

                    cost = 0
                    for k in range(-1, 2):
                        if r + k < 0 or r + k >= rows:
                            continue
                        for h in range(-1, 2):
                            if c + h - d < 0 or c + h >= cols:
                                continue
                            if is_left:
                                cost += abs(mat1[r + k, c + h] - mat2[r + k, c + h - d])
                            else:
                                cost += abs(mat2[r + k, c + h] - mat1[r + k, c + h + d])

                    if cost < min_cost:
                        min_cost = cost
                        min_d = d
            disp_mat[r, c] = min_d


@cuda.jit
def refinement(img, disp_mat, w):
    r, c = cuda.grid(2)
    rows, cols = disp_mat.shape
    if r < rows and c < cols:
        min_d = 0  # index <- 0
        min_dif = np.inf  # mdif <- inf
        for j in range(-w, w+1):  # for q from (p-w) to (p+w) do
            if j != 0 and 0 <= c + j < cols:  # if p != q then
                dif = sag.dist_l2(img[r, c], img[r, c+j])  # dif <- d(p, q)
                if dif < min_dif:  # if mdif > dif then
                    min_d = disp_mat[r, c+j]  # index <- q
                    min_dif = dif  # mdif <- dif
                # end if
            # end if
        # end for
        disp_mat[r, c] = min(min_d, disp_mat[r, c])  # disp(p) <- min(disp(p), disp(q))

