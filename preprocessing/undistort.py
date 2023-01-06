from libraries import *

def undistort(img2):
    mtx = [[661.36745612,   0.,         626.09553138],
           [0.,         663.35892418, 354.8585088],
           [0.,           0.,           1.]]

    dist = [[-3.52903264e-01,  1.65315869e-01, -3.39131620e-04, -1.64084879e-06, -4.30240406e-02]]

    # img2 = np.array(img2)
    mtx = np.array(mtx)
    dist = np.array(dist)

    h, w = img2.shape[:2]
    dst = cv2.undistort(img2, mtx, dist, None, mtx)
    return dst
