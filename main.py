import utility_functions as util
import cv2
import glob
import AR_planar
import warping_perspective

def main():
    util.runVideo('source_video.mp4','warping_perspective.avi',warping_perspective.processSingleFrame)
    util.runVideo('source_video.mp4','AR_Planar.avi',AR_planar.processSingleFrame)

    # rgb,gray = util.import_frame('frame1.jpg')
    # util.show_image(AR_planar.processSingleFrame(rgb,gray),(10,10))

    # rgb,gray = util.import_frame('frame1.jpg')
    # util.show_image(warping_perspective.processSingleFrame(rgb,gray),(10,10))

if __name__ == "__main__":
    main()
    cv2.waitKey(0)
    print("finished")
