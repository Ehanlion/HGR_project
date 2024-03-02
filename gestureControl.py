# Ethan Owen, 2/29/2024
# Program to test the ability to control devices using a hue bridge
# Uses the huesdk package to master the controls to the bridge
# No gui output, only console output from the program code
# Does NOT display the altered frame to the user since on-board there is no need to do this

from huesdk import Discover
from huesdk import Hue
import mediapipe as mp
import numpy as np
import datetime as dt
import cv2
import time
import re
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter

# Configure master, debug variables
DEBUG = True

# Configure global data:
path = "Models/gesture_recognizer.task"  # set the path
results = [""] # global tracker for results
last_N_results = []
max_N_results = 40
min_acceptable_certainty = 50 # integer percentage, max 100
certainty_weight = 40

# Color codes for hue color setting:
HUE_RED = 65535
HUE_BLUE = 21845
HUE_GREEN = 43690

# Start Configure Gesture Sequences

gs_main_names = ['Unknown',
                 'Closed_Fist',
                 'Open_Palm',
                 'Pointing_Up',
                 'Thumb_Up',
                 'Thumb_Down',
                 'Victory',
                 'ILoveYou']
gs_lights_on = []
gs_lights_off = []
gs_lights_color_rotate = []

# End Configure Gesture Sequences

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
def classify_gesture():
    gesture_pattern = r"gestures=\[\[Category\(.*?category_name='([^']*)'\)\]\]"
    gesture_match = re.search(gesture_pattern, results[-1])
    gesture_category = gesture_match.group(1) if gesture_match else None
    return gesture_category

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

# ==========================================
# Start Device Control Function Definitions:

# state pass for function takes on/off: 0 = off, 1 = on, other = fail
# bridge pass takes bridge: pass a bridge object that was created
# color pass takes pre-def int: pass either HUE_RED, HUE_BLUE, HUE_GREEN

# Function to set the state of a light to either on or off:
def set_light_state(bridge: Hue, light_id: int, state: int):
    try:
        light = bridge.get_light(light_id) # get the light id
    
        # Set light states:
        if state == 0:
            light.off()
        else:
            light.on()
    except Exception as e:
        print(f'Encounter Exception: {e}')
        return 1
    return 0
        
# Function to set both state and color of a light: 
def set_light_color(bridge: Hue, light_id: int, state: int, color: int):
    try:
        light = bridge.get_light(light_id) # get the light id
    
        # Set light states:
        if state == 0:
            light.off()
        else:
            light.on()
            
        # Set light color:
        light.set_color(hue=color)
    except Exception as e:
        print(f'Encounter Exception: {e}')
        return 1
    return 0

# End Device Control Function Definitions
# ==========================================

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
        frame_cpy = frame
        if not ret: break
        
        # Start MP Model Processing
        
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = recognizer.recognize_async(image,ts)
        classification = classify_gesture()
        last_N_results.append(classification)
        results = results[-1:]
        ts = ts + 1
        
        if len(last_N_results) > max_N_results: last_N_results = last_N_results[1:]
        common_class, certainty = get_certainty(last_N_results)
        if common_class == last_N_results[-1]: certainty = certainty
        if certainty < min_acceptable_certainty: classification = "Failure"
        else: classification = common_class
        
        # End MP Model Processingq
        
        if DEBUG:
            print(f'Classed Gesture: {classification}')

        cv2.imshow('test', cv2.flip(frame, 1))
        if cv2.waitKey(1) == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break

# Close all open releases and captures:
capture.release()
cv2.destroyAllWindows()