# ======= imports
pass
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import utility_functions as util
import camera_calibration as cc
import mesh_renderer


# calibrate camera using chess board
square_size = 2.95
pattern_size = (7, 4)
img_mask = "./chess_board_frames/*.jpg"

camera_matrix, dist_coeffs  = cc.calibrateCamera(square_size, pattern_size, img_mask)

# template image keypoint and descriptors
book_size = (27.4, 21.7)
t_rgb,t_gray = util.import_frame('template.jpg')

# obj_3d = mesh_renderer.MeshRenderer(camera_matrix, 1920, 1080, "skull/12140_Skull_v3_L2.obj")
obj_3d = mesh_renderer.MeshRenderer(camera_matrix, 1920, 1080, "pickle-rick-rat/Pickle Rick Rat v2.obj")


feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()


def processSingleFrame(frame_rgb,frame_gray):

    frame_kp,desc_f = feature_extractor.detectAndCompute(frame_gray,None)
    template_kp,desc_t = feature_extractor.detectAndCompute(t_gray,None)


    matches = bf.knnMatch(desc_f, desc_t, k=2)

    # Apply ratio test
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance/m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]
    #%%
    #finds the homogriphic matrix
    good_kp_f = np.array([frame_kp[m.queryIdx].pt for m in good_match_arr])
    good_kp_t = np.array([template_kp[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_t, good_kp_f, cv2.RANSAC, 5.0)

    #masking only the inside homography
    inside_homography_masked = np.array(masked.ravel() > 0)
    inside_kp_f = np.array(good_kp_f)[inside_homography_masked,:]
    inside_kp_t = np.array(good_kp_t)[inside_homography_masked,:]

    #rescaling the template in the image to its original size
    rescale_kp_t = np.array([[x * book_size[1] / t_gray.shape[1] , y * book_size[0] / t_gray.shape[0],0] for x,y in inside_kp_t])
    res,rvec,tvec = cv2.solvePnP(rescale_kp_t, inside_kp_f, camera_matrix, dist_coeffs)
    frame_rgb = obj_3d.draw(frame_rgb,rvec,tvec)

    return frame_rgb

# util.runVideo('IMG_8283.MOV','AR_Planar.avi',processSingleFrame)



