# Ethan Owen, 02/01/2024
# This program performs model-based gesture classification
# It can also average the gestures it sees in a window to give a certainty about what gesture is presented.
# The classified geture is determined off of a lower certainty bound and average gesture in a window.
# In my eyes, this 'smooths' rough, raw detection into a slicker classifier.

import mediapipe as mp
import numpy as np
import datetime as dt
import cv2
import time
import re
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter

# Configure global data:
path = "Models/gesture_recognizer.task"  # set the path
results = [""] # global tracker for results
last_N_results = []
max_N_results = 40
min_acceptable_certainty = 50 # integer percentage, max 100
certainty_weight = 40

# Short hands for base options:
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def store_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    results.append(str(format(result))) # append and format result to results

# Function to extract the gesture class and handedness from the result string:
def classify():
    gesture_pattern = r"gestures=\[\[Category\(.*?category_name='([^']*)'\)\]\]"
    gesture_match = re.search(gesture_pattern, results[-1])
    gesture_category = gesture_match.group(1) if gesture_match else None
    return gesture_category

# Calculate current fps and returns average and previous times:
def get_fps(prev):
    curr = time.time()
    fps = 1 / (curr - prev)
    return int(fps), curr

# Get last non-none element in a list:
def get_certainty(lst):
    filtered_list = [item for item in lst if item is not None]
    count = Counter(filtered_list)
    if not count:
        return None, 0
    most_common, repetitions = count.most_common(1)[0]
    total_values = len(lst)
    certainty = int((repetitions / total_values) * 100)
    return most_common, certainty
    
# Configure Gesture Recognizer Options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=store_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened(): 
        capture.release()
        cv2.destroyAllWindows()
        exit()
    
    # Start Variable Declaration
    
    ts = 0
    prev_ft = 0
    classification = ""
    certainty = 0.0
    
    # End Variable Declaration
    
    while True:
        ret, frame = capture.read()
        if not ret: break
        
        # Start MP Model Processing
        
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = recognizer.recognize_async(image,ts)
        classification = classify()
        last_N_results.append(classification)
        results = results[-1:]
        ts = ts + 1
        
        if len(last_N_results) > max_N_results: last_N_results = last_N_results[1:]
        common_class, certainty = get_certainty(last_N_results)
        if common_class == last_N_results[-1]: certainty = certainty
        if certainty < min_acceptable_certainty: classification = "Bad Average"
        else: classification = common_class
        
        # End MP Model Processing
        
        # Start Frame Preparation
        
        fps, prev_ft = get_fps(prev_ft)
        frame = cv2.flip(frame,1)
        cv2.putText(frame, f"Class: {classification} {certainty}", (5,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps}", (5,55), cv2. FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
        
        # End Frame Preparation
        
        cv2.imshow('test', frame)
        if cv2.waitKey(1) == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break

# Close all open releases and captures:
capture.release()
cv2.destroyAllWindows()