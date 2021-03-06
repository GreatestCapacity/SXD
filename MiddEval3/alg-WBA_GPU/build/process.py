import cv2 as cv
import stereo_alg as sa
import time
from alg_wba_gpu import alg_wba_gpu


def process(file1_name, file2_name, outfile_name, maxdisp):
    # Record the time when the algorithmn start
    start_time = time.clock()

    # Read the two images
    img1 = cv.imread(file1_name, cv.IMREAD_COLOR)
    img2 = cv.imread(file2_name, cv.IMREAD_COLOR)

    disp_mat = alg_wba_gpu(img1, img2, maxdisp)

    # Output runtime
    end_time = time.clock()
    secs = end_time - start_time
    rows = img1.shape[0]
    cols = img1.shape[1]
    print('runtime: %.2fs  (%.2fs/MP)' % (secs, secs/(rows*cols/1000000.0)))

    # Save disparity image to pfm file
    sa.save_pfm(disp_mat, outfile_name, 1.0/maxdisp)
