# ======= imports
pass
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import utility_functions as util
import camera_calibration as cc
# ======= constants
pass

# calibrate camera using chess board
square_size = 2.95
pattern_size = (7, 4)
img_mask = "./chess_board_frames/*.jpg"

camera_matrix, dist_coeffs  = cc.calibrateCamera(square_size, pattern_size, img_mask)

# template image keypoint and descriptors
book_size = (27.4, 21.7)
t_rgb,t_gray = util.import_frame('template.jpg')

feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()
pass

# ===== video input, output and metadata
pass

def prossesSingleFrame(frame_rgb,frame_gray):

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
    good_kp_l = np.array([frame_kp[m.queryIdx].pt for m in good_match_arr])
    good_kp_r = np.array([template_kp[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)

    ##TODO:
    ## understand this shit

    ##good_homography_mask = (np.array(masked).flatten() > 0)
    #best_kp_template = np.array(good_kp_template)[good_homography_mask, :]
    #best_kp_frame = np.array(good_kp_frame.reshape(good_kp_frame.shape[0], 2))[good_homography_mask, :]
#
    ## ========= solve PnP to get cam pose (r_vec and t_vec)
    ## `cv2.solvePnP` is a function that receives:
    ## - xyz of the template in centimeter in camera world (x,3)
    ## - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    ## - camera K
    ## - camera dist_coeffs
    ## and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    ##
    ## NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    ## because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    ## For this we just need the template width and height in cm.
    #best_kp_template_xyz_cm = np.array([[x[0] / template_im_gray.shape[1] * TEMPLATE_IM_W, x[1] / template_im_gray.shape[0] * TEMPLATE_IM_H, 0] for x in best_kp_template])
    #res, rvec, tvec = cv2.solvePnP(best_kp_template_xyz_cm, best_kp_frame, K, dist_coeffs)




    #lays the cover image
    
    # util.show_image(rgb_r_warped,figsize= (20,20))

    # covered_im = rgb_r_warped.copy()

    #%%
    return res
    # util.show_image(res,figsize= (20,20))

# ========== run on all frames
while True:
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    pass

    # ======== find homography
    # also in SIFT notebook
    pass

    # ++++++++ take subset of keypoints that obey homography (both frame and reference)
    # this is at most 3 lines- 2 of which are really the same
    # HINT: the function from above should give you this almost completely
    pass

    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K
    # - camera dist_coeffs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows
    pass

    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
    pass

    # =========== plot and save frame
    pass

# ======== end all
pass



    # finds corners in the warped image
    # #%%
    # mask = mask.astype(np.float32)
# mask /= 255
# # mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
# corners = cv2.goodFeaturesToTrack(mask, 4, 0.1, 50)

# corners = corners.astype(np.int32)
# for corner in corners:
#     x,y = corner.ravel()
#     cv2.circle(rgb_r_warped,(x,y),30,(36,255,12),-1)
