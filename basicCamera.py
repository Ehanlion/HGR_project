import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# First prepare the capture device:
cap = cv.VideoCapture(0) # get video capture active
if not cap.isOpened():
    cap.release()
    print("Cannot open video device.")
    exit()

# Enter an infinite while loop to show video:
while True:
    ret, frame = cap.read() # ret is a boolean T/F, frame is video frame
    
    # examine ret to see if cap.read was valid:
    if not ret:
        print("Error reading Frame.")
        break
    
    # not we know frame is valid since ret is true
    frame = cv.flip(frame,1) # flip the frame 
    cv.imshow('frame',cv.cvtColor(frame, cv.COLOR_BGR2GRAY)) # show frame, grayscaled
    if cv.waitKey(1) == ord('q'):
        print("Exit key pressed, exiting.")
        break
    
# at the end, release and destroy frames:
cap.release() # release the capture at the end
cv.destroyAllWindows() # clean and destroy opened windows