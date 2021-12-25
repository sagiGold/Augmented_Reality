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
frame_rgb ,frame_gray = util.import_frame('frame233.jpg')
t_rgb,t_gray = util.import_frame('template.jpg')

feature_extractor = cv2.SIFT_create()

kp_template,desc_t = feature_extractor.detectAndCompute(frame_gray,None)
kp_frame,desc_f = feature_extractor.detectAndCompute(t_gray,None)

test = cv2.drawKeypoints(frame_rgb, kp_template, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
util.show_image(test,figsize)
#%%
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_t, desc_f, k=2)

# Apply ratio test
good_and_second_good_match_list = []
for m in matches:
    if m[0].distance/m[1].distance < 0.5:
        good_and_second_good_match_list.append(m)
good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

# show only 30 matches
im_matches = cv2.drawMatchesKnn(frame_rgb, kp_template, t_rgb, kp_frame,
                                good_and_second_good_match_list[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

util.show_image(im_matches,figsize=(20,20))


#%%

good_kp_l = np.array([kp_template[m.queryIdx].pt for m in good_match_arr])
good_kp_r = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])
H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)


rgb_r_warped = cv2.warpPerspective(t_rgb, H, (frame_rgb.shape[1] + t_rgb.shape[1], frame_rgb.shape[0]))
rgb_r_warped[0:frame_rgb.shape[0], 0:frame_rgb.shape[1]] = frame_rgb

util.show_image(rgb_r_warped,figsize= (20,20))
#%%
# ===== video input, output and metadata
def runVideo():
    vidcap = cv2.VideoCapture('Richard_Szeliski.mp4')
    success,frame = vidcap.read()

    mask = util.createMask(frame)
    
    h,w,d = frame.shape
    out = cv2.VideoWriter('lane_video_final.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30,(w,h))
    
    # util.show_image(mask,(10,10))

    cropped_mask = mask.copy()
    cropped_mask[370:,:] = 255
    cropped_mask = cv2.cvtColor(cropped_mask,cv2.COLOR_GRAY2BGR)
    # util.show_image(cropped_mask,(10,10))
    cntr=0
    frame_list=[]
    path = ''
    # frame = util.import_frame("frame827.jpg")
    prev_l = None
    prev_r = None

    global t_cntr
    t_cntr  = 0

    while success:
        frame_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        finished_frame,prev_l,prev_r,cntr = fpl.frame_find_lanes(frame_gray,frame,mask,cropped_mask,prev_l,prev_r,cntr)

        cv2.imwrite(path+"\\frame%d.jpg" %t_cntr,finished_frame)

        out.write(finished_frame)

        # frame_list.append(fpl.frame_find_lanes(frame_gray,frame,mask,cropped_mask))
        
        success,frame = vidcap.read()
        t_cntr += 1


pass

# ========== run on all frames
while True:
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    pass

    # ======== find homography
    # also in SIFT notebook
    pass

    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    pass

    # =========== plot and save frame
    pass

# ======== end all
pass
