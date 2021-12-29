#%%
# ======= imports
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import utility_functions as util

#%%
# ======= constants
figsize = (10,10)
pass
# === template image keypoint and descriptors
t_rgb,t_gray = util.import_frame('template.jpg')
cover_im_rgb,cover_im_gray = util.import_frame('Pickle_Rick.jpg')

feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()

def prossesSingleFrame(frame_rgb,frame_gray):

    # frame_rgb ,frame_gray = util.import_frame('frame183.jpg')

    frame_kp,desc_f = feature_extractor.detectAndCompute(frame_gray,None)
    template_kp,desc_t = feature_extractor.detectAndCompute(t_gray,None)

    # test = cv2.drawKeypoints(frame_rgb, frame_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # util.show_image(test,figsize)
    #%%
    matches = bf.knnMatch(desc_f, desc_t, k=2)

    # Apply ratio test
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance/m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

    # show only 30 matches
    # im_matches = cv2.drawMatchesKnn(frame_rgb, frame_kp, t_rgb, template_kp,
                                    # good_and_second_good_match_list[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #%%
    #finds the homogriphic matrix
    good_kp_l = np.array([frame_kp[m.queryIdx].pt for m in good_match_arr])
    good_kp_r = np.array([template_kp[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)

    #lays the cover image
    rgb_r_warped = cv2.warpPerspective(cover_im_rgb, H, (frame_rgb.shape[1], frame_rgb.shape[0]))
    # util.show_image(rgb_r_warped,figsize= (20,20))

    # covered_im = rgb_r_warped.copy()
    warped_gray = cv2.cvtColor(rgb_r_warped,cv2.COLOR_RGB2GRAY)
    mask = rgb_r_warped
    t,mask = cv2.threshold(warped_gray,1,255,cv2.THRESH_BINARY)
    (contours, hierarchy) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    # util.show_image(mask,(20,20))

    #%%
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    mask = cv2.bitwise_not(mask)
    # util.show_image(mask,figsize= (20,20))

    frame_rgb = cv2.bitwise_and(frame_rgb,mask);
    res = cv2.bitwise_or(frame_rgb,rgb_r_warped)

    return res
    # util.show_image(res,figsize= (20,20))



#%%
# ===== video input, output and metadata
def runVideo():
    vidcap = cv2.VideoCapture('IMG_8283.MOV')
    success,frame = vidcap.read()
    h,w,d = frame.shape
    out = cv2.VideoWriter('warping_perspective_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30,(w,h))
    # util.show_image(mask,(10,10))
    # util.show_image(cropped_mask,(10,10))
    cntr=0
    frame_list=[]
    while success:
        frame_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        finished_frame  =prossesSingleFrame(frame_src,frame_gray)
        # cv2.imwrite(path+"\\frame%d.jpg" %t_cntr,finished_frame)
        finished_frame = cv2.cvtColor(finished_frame,cv2.COLOR_RGB2BGR)
        out.write(finished_frame)
        # frame_list.append(fpl.frame_find_lanes(frame_gray,frame,mask,cropped_mask))
        success,frame = vidcap.read()
        print('progress:',cntr,'/451',end='\r')
        cntr+=1

# runVideo();

# frame_rgb,frame_gray = util.import_frame('frame199.jpg')

# util.show_image(prossesSingleFrame(frame_rgb,frame_gray),(10,10))
