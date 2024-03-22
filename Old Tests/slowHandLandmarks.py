import numpy as np
import cv2 as cv
import mediapipe as mp 

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

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
    
    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)

    # Process the image and draw hand landmarks
    results = hands.process(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the image
    cv.imshow('MediaPipe Hands', frame)
    
    # Break the loop when 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break
    
    while False:
        if cv.waitKey(1) == ord('c'):
            break
    
# at the end, release and destroy frames:
cap.release() # release the capture at the end
cv.destroyAllWindows() # clean and destroy opened windows