import utility_functions as util
import cv2
import glob
import AR_planar
import warping_perspective

def main():
    util.runVideo('IMG_8283.MOV','AR_Planar.avi',AR_planar.processSingleFrame)
    util.runVideo('IMG_8283.MOV','AR_Planar.avi',warping_perspective.processSingleFrame)

    # rgb,gray = util.import_frame('frame1.jpg')
    # util.show_image(arp.processSingleFrame(rgb,gray),(10,10))
    
if __name__ == "__main__":
    main()
    cv2.waitKey(0)
    print("finished")
