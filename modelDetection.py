# Ethan Owen, 1/27/2024
# modelDetection.py program draws MediaPipe overlay and classifies gestures.
# Limited to classifying only left or right, whichever is seen first.
# Current and Average FPS counters built in, program limited to 10 FPS (ideally).

import mediapipe as mp
import numpy as np
import datetime as dt
import time
import cv2
import re
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Control Variables:
avg_fps_window = 100 # controls the number of frames to track the average fps over

# Global variables for MediaPipe model:
model_path = "Models/gesture_recognizer.task" # save the model path for the mediapipes GR model
gresult = [""] # do not remove; needed to store results from the recognizer for whatever reason

# Setup the options for the visualizer:
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def getCallbackResult(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    gresult.append(result) # store result in global list because I have no idea how else this works
    return result if result else None # this does exactly nothing as far as I know
    
# Function to extract the gesture class and handedness from the result string:
def getResultAndHandedness(res_str):
    # Pattern to find the gesture's category name
    gesture_pattern = r"gestures=\[\[Category\(.*?category_name='([^']*)'\)\]\]"
    gesture_match = re.search(gesture_pattern, res_str)

    # Pattern to find the handedness category name
    handedness_pattern = r"handedness=\[\[Category\(.*?category_name='([^']*)'\)\]\]"
    handedness_match = re.search(handedness_pattern, res_str)

    # Extracting the gesture and handedness
    gesture_category = gesture_match.group(1) if gesture_match else None
    handedness_category = handedness_match.group(1) if handedness_match else None

    return gesture_category, handedness_category

# Calculate current fps and returns average and previous times:
def calculateCurrentFPS(prev):
    curr = time.time()
    fps = 1 / (curr - prev)
    return fps, curr # return fps and current time as previous time  

# Calculate average fps and return average and list of times:
def calculateAverageFPS(arr):
    arr.append(fps)
    arr = arr[1:] if len(arr) > avg_fps_window else arr
    avg = sum(arr) / len(arr)
    return avg, arr # return average and list of times

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path), # dictate the model path
    running_mode=VisionRunningMode.LIVE_STREAM, # set mode to live stream
    result_callback=getCallbackResult) # set call back function to getResult

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0) # open the cv2 video capture
    if not cap.isOpened(): exit() # kill the program if we cannot open the capture device
    
    # Setup for the hand tracking visualization:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()
    
    # trackers for the fps
    timestamp = 0 # start frame counter
    prev_frame_time = 0 # previous frames time for fps calculation
    total_fps = [] # list of last 'N' fps values for average calculation
        
    while True:
        ret, frame = cap.read() # grabe ret and frame, ret = T/F, frame = cv2 frame
        if not ret: break
        
        # FPS calculation
        fps, prev_frame_time = calculateCurrentFPS(prev_frame_time) # Calculate current fps 
        avg_fps, total_fps = calculateAverageFPS(total_fps) # Calculate average fps
        
        # MediaPipe Hand tracking
        hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # process rgb frame for hand positions
        
        # Setup image and then get result data from model:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # define the mp_image
        result = recognizer.recognize_async(mp_image,timestamp) # get result from the recognizer
        classification, handedness = getResultAndHandedness(str(format(gresult[-1]))) # extract the classification and handedness
        gresult = gresult[-1:] # slice and store last element only for size reasons
        print(f"Ts: {timestamp}, Class: {classification}, Hand: {handedness}")
        timestamp = timestamp + 1 # increment frame counter
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Add the overlay
        frame = cv2.flip(frame, 1) # flip the frame first
        textHand1 = classification if classification and (handedness == "Left")  else ""
        textHand2 = classification if classification and (handedness == "Right") else ""
        cv2.putText(frame, f"LH: {textHand1}", (5,35), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,255), 2) # Draw left hand data
        cv2.putText(frame, f"RH: {textHand2}", (5,60), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,255), 2) # Draw right hand data
        cv2.putText(frame, f'{int(fps)}|{int(avg_fps)}', (5,85), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2) # Draw current FPS

        # Display the image
        cv2.imshow('frame', frame) # show the frame
        if cv2.waitKey(1) == ord('q'): break
        time.sleep(0.05) # this caps the fps at 10 fps (for whatever reason...)
        
# clean up open captures and windows
cap.release() # release the capture at the end
cv2.destroyAllWindows() # clean and destroy opened windows