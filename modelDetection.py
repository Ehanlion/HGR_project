import mediapipe as mp
import numpy as np
import datetime as dt
import time
import cv2
import re
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

debug = False
model_path = "Models/gesture_recognizer.task" # save the model path for the mediapipes GR model

# Setup the options for the visualizer:
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

gresult = [""] # do not remove; needed to store results from the recognizer for whatever reason

# Create a gesture recognizer instance with the live stream mode:
def getCallbackResult(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    gresult.append(result) # store result in global list because I have no idea how else this works
    return result if result else None # this does exactly nothing as far as I know
    
# Function to extract the gesture classification from the result string:
def getResultClassification(res_str):
    pattern = r"category_name='([^']*)'" # Pattern to find the gesture's category name
    matches = re.findall(pattern, res_str)  # Find all matches in the string
    return matches[0] if matches else None # Return the first match or None if no match is found

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path), # dictate the model path
    running_mode=VisionRunningMode.LIVE_STREAM, # set mode to live stream
    result_callback=getCallbackResult) # set call back function to getResult

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0) # open the cv2 video capture
    ts = 0 # start frame counter
    
    # Setup for the hand tracking visualization:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()
    
    # Stop the program if the 
    if not cap.isOpened():
        cap.release()
        print("Cannot open video device.")
        exit()
    
    # trackers for the fps
    prev_frame_time = 0
    frame_count = 0
    total_fps = 0 
        
    while True:
        ret, frame = cap.read() # grabe ret and frame, ret = T/F, frame = cv2 frame
        if not ret: break
        
        # FPS calculation
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Update total FPS and frame count for average calculation
        total_fps += fps
        frame_count += 1
        avg_fps = total_fps / frame_count

        # flip the frame and config the mp_image settings and data
        frame = cv2.flip(frame, 1) # flip the cv2 frame for visualizing purposes
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # define the mp_image
        
        # MediaPipe Hand tracking
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        
        # Attempt to get the result data:
        result = recognizer.recognize_async(mp_image,ts) # get result from the recognizer
        classification = getResultClassification(str(format(gresult[-1]))) # extract the classification using a function
        gresult = gresult[-1:] # slice and store last element only for size reasons
        if debug: print(f"Result {ts} {classification}") # print the results for testing if we in debug mode
        
        ts = ts + 1 # increment frame counter
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Add the overlay
        textHand1 = classification if classification else "Unknown"
        textHand2 = classification if classification else "Unknown"
        cv2.putText(frame, f"LH: {textHand1}", (5,15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        cv2.putText(frame, f"RH: {textHand2}", (155,15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        
        # Draw FPS values:
        cv2.putText(frame, f'CFPS: {int(fps)}', (frame.shape[1] - 80, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2) # Draw current FPS
        cv2.putText(frame, f'AFPS: {int(avg_fps)}', (frame.shape[1] - 80, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2) # Draw average FPS

        # Display the image and handle breaking
        cv2.imshow('frame', frame) # show the frame
        if cv2.waitKey(1) == ord('q'): 
            cap.release() # release the capture at the end
            cv2.destroyAllWindows() # clean and destroy opened windows
            break