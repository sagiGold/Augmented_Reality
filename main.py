import utility_functions as util
import cv2
import glob

# util.convertVideo2Frames('Chess_Board_video.MOV',r'C:\Users\idano\Documents\HomeWork\3rd semester\computer vision\Augmented Reality\Augmented_Reality\chess_board_frames')

# im,gray = util.import_frame('Pickle_Rick.jpg')
# util.show_image(im,(10,10))

# row, col = im.shape[:2]
# im = im[50:row-50,50:col-50]
# bottom = im[row-2:row, 0:col]
# mean = cv2.mean(bottom)[0]

# bordersize = 50
# im = cv2.copyMakeBorder(
#     im,
#     top=bordersize,
#     bottom=bordersize,
#     left=bordersize,
#     right=bordersize,
#     borderType=cv2.BORDER_CONSTANT,
#     value=(0, 172, 70)
# )
# util.show_image(im,(10,10))

# cv2.imwrite('PICKLE_RICK',im)