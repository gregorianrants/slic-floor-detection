import numpy as np
import cv2

# define a video capture object
vid = cv2.VideoCapture(0)

vid.set(cv2.CAP_PROP_FRAME_WIDTH,640);
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

while (True):
        # Capture the video frameq
        # by frame
        ret, original_image = vid.read()
        hsv_original = cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)

        

        # Display the resulting frame
  

        roi = cv2.imread('roi.jpg')
        

        hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi],[0, 1], None, [180, 256], [0, 180, 0, 256] )

        


        # normalize histogram and apply backprojection
       
        mask = cv2.calcBackProject([hsv_original],[0,1],roi_hist,[0,180,0,256],1)
     

        cv2.imshow('original',original_image)
        cv2.imshow('result',mask)





        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


