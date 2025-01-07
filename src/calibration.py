import numpy as np
import cv2 as cv
import glob
import imageio
from typing import List

if __name__ == "__main__":
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    objp = np.zeros((16*16, 3), np.float32)
    objp[:, :2] = np.mgrid[0:16, 0:16].T.reshape(-1, 2) 
    objpoints = [] 
    imgpoints = [] 
    
    def load_images(filenames: List) -> List:
        return [imageio.imread(filename) for filename in filenames]

    left_imgs_path = glob.glob("data/chess_left_images/*.jpg")
    left_imgs = load_images(left_imgs_path)
    right_imgs_path = glob.glob("data/chess_right_images/*.jpg")
    right_imgs = load_images(right_imgs_path)
    images = left_imgs + right_imgs
    
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (16,16))
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (8,8), (-1,-1), criteria)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, (16, 16), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)
