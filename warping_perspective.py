# ======= imports 
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

# ======= constants
pass

# === template image keypoint and descriptors
pass

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
