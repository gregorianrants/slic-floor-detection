import cv2


# define a video capture object
vid = cv2.VideoCapture(0)

vid.set(cv2.CAP_PROP_FRAME_WIDTH,640);
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

while (True):
    # Capture the video frameq
    # by frame
    ret, img = vid.read()

    print(img.shape)
  
    # Display the resulting frame
    cv2.imshow('frame', img)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('roi.jpg', img)
        break


# After the loop release the cap object



vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
