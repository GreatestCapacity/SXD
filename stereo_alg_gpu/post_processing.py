import math
import numpy as np
from numba import cuda
from basic_alg import argmin, dist_l2, mat_min


@cuda.jit('void(float32[:, :], float32[:, :, :])')
def subpx(disp_mat, cost_maps):
    r, c = cuda.grid(2)
    ndisp, rows, cols = cost_maps.shape
    if r < rows and c < cols:
        d = disp_mat[r, c]
        if d - 1 < 0 or d + 1 >= min(ndisp, c+1):
            return
        cost = cost_maps[int(d), r, c]
        cost_f = cost_maps[int(d)+1, r, c]
        cost_b = cost_maps[int(d)-1, r, c]
        disp_mat[r, c] = d - (cost_f - cost_b) / (2 * (cost_f - 2*cost + cost_b))


def subpixel_enhance(disp_mat_gpu, cost_maps_gpu):
    subpx[(30, 30), (30, 30)](disp_mat_gpu, cost_maps_gpu)
    return disp_mat_gpu


@cuda.jit('void(float32[:, :, :], float32[:, :])')
def gen_disp(cost_maps_gpu, disp_mat_gpu):
    r, c = cuda.grid(2)
    ndisp, rows, cols = cost_maps_gpu.shape
    if r < rows and c < cols:
        disp_mat_gpu[r, c] = argmin(cost_maps_gpu[:, r, c])


def cost_to_disp(cost_maps_gpu):
    _, rows, cols = cost_maps_gpu.shape
    disp_mat = np.zeros([rows, cols], dtype='float32')
    disp_mat_gpu = cuda.to_device(disp_mat)

    gen_disp[(30, 30), (30, 30)](cost_maps_gpu, disp_mat_gpu)

    return disp_mat_gpu


@cuda.jit
def gen_disp_with_penalty(mat_gpu, cost_maps_gpu, disp_mat_gpu, T, is_left):
    r = cuda.grid(1)
    ndisp, rows, cols = cost_maps_gpu.shape

    if r < rows:
        if is_left:
            c_range = range(1, cols)
            d0 = argmin(cost_maps_gpu[:, r, 0])
            disp_mat_gpu[r, 0] = d0
        else:
            c_range = range(cols-2, -1, -1)
            d0 = argmin(cost_maps_gpu[:, r, cols-1])
            disp_mat_gpu[r, cols-1] = d0

        for c in c_range:
            min_cost = np.inf
            min_d = 0
            for d in range(ndisp):
                if cost_maps_gpu[d, r, c] == np.inf:
                    break
                if is_left:
                    _c = c - 1
                else:
                    _c = c + 1
                penalty = T * abs(d - d0) * (1 - abs(mat_gpu[r, c] - mat_gpu[r, _c]) / 255)
                cost = cost_maps_gpu[d, r, c] + penalty
                if cost < min_cost:
                    min_cost = cost
                    min_d = d

            d0 = min_d
            disp_mat_gpu[r, c] = d0


def cost_to_disp_with_penalty(mat_gpu, cost_maps_gpu, T=8):
    disp_mat = np.zeros_like(mat_gpu, dtype='float32')
    d_l = cuda.to_device(disp_mat)
    d_r = cuda.to_device(disp_mat)
    disp_mat_gpu = cuda.to_device(disp_mat)
    gen_disp_with_penalty[30, 30](mat_gpu, cost_maps_gpu, d_l, T, True)
    gen_disp_with_penalty[30, 30](mat_gpu, cost_maps_gpu, d_r, T, False)
    mat_min[(30, 30), (30, 30)](d_l, d_r, disp_mat_gpu)
    return disp_mat_gpu


@cuda.jit('void(float32[:, :, :], float32[:, :, :])')
def grcm(cost_maps, right_cost_maps):
    r, c = cuda.grid(2)
    ndisp, rows, cols = cost_maps.shape

    if r < rows and c < cols:
        for d in range(ndisp):
            right_cost_maps[d, r, c] = cost_maps[d, r, c + d]


def get_right_cost_maps(cost_maps_gpu):
    right_cost_maps = np.full_like(cost_maps_gpu, np.inf, dtype='float32')
    right_cost_maps_gpu = cuda.to_device(right_cost_maps)
    grcm[(30, 30), (30, 30)](cost_maps_gpu, right_cost_maps)
    return right_cost_maps


@cuda.jit
def lrc(left, right, check, t):
    r, c = cuda.grid(2)
    rows, cols = left.shape
    if r < rows and c < cols:
        d = int(left[r, c])
        if c - d >= 0 and abs(d - right[r, c - d]) <= t:
            check[r, c] = True


def left_right_check(left_disp_mat_gpu, right_disp_mat_gpu, thresh=1.0):
    check_mat = np.full_like(left_disp_mat_gpu, False, dtype='bool')
    check_mat_gpu = cuda.to_device(check_mat)
    lrc[(30, 30), (30, 30)](left_disp_mat_gpu, right_disp_mat_gpu, check_mat_gpu, thresh)
    return check_mat_gpu


@cuda.jit
def rgb_interp(img, disp_mat, check_mat):
    r, c = cuda.grid(2)
    rows, cols = disp_mat.shape
    if r < rows and c < cols and not check_mat[r, c]:
        min_dist = np.inf
        d = 0.0
        for i in range(-1, 2):
            if r + i < 0 or r + i >= rows:
                continue
            for j in range(-1, 2):
                if c + j < 0 or c + j >= cols or not check_mat[r+i, c+j]:
                    continue

                dist = dist_l2(img[r, c], img[r+i, c+j])
                if dist < min_dist:
                    min_dist = dist
                    d = disp_mat[r+i, c+j]
        disp_mat[r, c] = d


def rgb_interpolation(img_gpu, disp_mat_gpu, check_mat_gpu):
    rgb_interp[(50, 50), (20, 20)](img_gpu, disp_mat_gpu, check_mat_gpu)
    return disp_mat_gpu


@cuda.jit
def left_interp(disp_mat, check_mat):
    r = cuda.grid(1)
    rows, cols = disp_mat.shape
    if r < rows:
        for c in range(1, cols):
            if not check_mat[r, c] and check_mat[r, c - 1]:
                disp_mat[r, c] = disp_mat[r, c - 1]


def left_interpolation(disp_mat_gpu, check_mat_gpu):
    left_interp[30, 30](disp_mat_gpu, check_mat_gpu)
    return disp_mat_gpu

