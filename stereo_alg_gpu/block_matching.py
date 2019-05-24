import numpy as np
from numba import cuda
from basic_alg import sub, mul, div, absolute, square, sqrt, aggregt, xd


def sad(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    rad = patch_size // 2
    ndisp = maxdisp + 1
    result = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([ndisp, rows, cols], np.inf, dtype='float32')

    result_gpu = cuda.to_device(result)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(ndisp):
        sub[(30, 30), (30, 30)](mat1_gpu[:, d:], mat2_gpu[:, :cols-1-d], result_gpu[:, d:])
        absolute[(30, 30), (30, 30)](result_gpu[:, d:], result_gpu[:, d:])
        aggregt[(30, 30), (30, 30)](result_gpu[:, d:], cost_maps_gpu[d, :, d:], rad)

    return cost_maps_gpu


def ssd(mat1_gpu, mat2_gpu, patch_size, maxdisp):
    rows, cols = mat1_gpu.shape
    rad = patch_size // 2
    ndisp = maxdisp + 1
    result = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([ndisp, rows, cols], np.inf, dtype='float32')

    result_gpu = cuda.to_device(result)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(ndisp):
        sub[(30, 30), (30, 30)](mat1_gpu[:, d:], mat2_gpu[:, :cols-1-d], result_gpu[:, d:])
        square[(30, 30), (30, 30)](result_gpu[:, d:], result_gpu[:, d:])
        aggregt[(30, 30), (30, 30)](result_gpu[:, d:], cost_maps_gpu[d, :, d:], rad)

    return cost_maps_gpu


def sxd(mat1_gpu, mat2_gpu, patch_size, maxdisp, s=255, t=12.5):
    rows, cols = mat1_gpu.shape
    rad = patch_size // 2
    ndisp = maxdisp + 1
    result = np.full([rows, cols], np.inf, dtype='float32')
    cost_maps = np.full([ndisp, rows, cols], np.inf, dtype='float32')

    result_gpu = cuda.to_device(result)
    cost_maps_gpu = cuda.to_device(cost_maps)

    for d in range(ndisp):
        xd[(30, 30), (30, 30)](mat1_gpu[:, d:], mat2_gpu[:, :cols-1-d], result_gpu[:, d:], s, t)
        aggregt[(30, 30), (30, 30)](result_gpu[:, d:], cost_maps_gpu[d, :, d:], rad)

    return cost_maps_gpu

