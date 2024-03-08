# Ethan Owen, 3/7/2024
# Master program for the HGR project.
# Implements gesture recognition and complex sequence recognition
# Implements various light related functions

from huesdk import Discover
from huesdk import Hue
import mediapipe as mp
import numpy as np
import datetime as dt
import cv2
import time
import re
import string
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter

# Configure master, debug variables
DEBUG = True

# Configure global data:
path = "Models/gesture_recognizer.task"  # set the path
results = [""] # global tracker for results
last_N_results = [] # list containing last classed results
max_N_results = 40
min_acceptable_certainty = 50 # integer percentage, max 100

# declare empty hue objects for redundancy
hue = None
light_corner = None
light_bed = None
bridge_ip = "192.168.1.148"
bridge_username = "ysFHipKKPahizAwVKB8zYJlpPbVc4tyFBLF6MJDg"

# Configure Variables for Hue Lights Interaction:
try:
    hue = Hue(bridge_ip=bridge_ip, username=bridge_username) # create the hue object
    light_corner = hue.get_light(id_=1) # get light 1 created
    light_bed = hue.get_light(id_=2) # get light 2 created
except Exception as e:
    print(f"Encountered Exception loading Hue Object: {e}")

# Light Hue Values
hue_white_hex = "#E8E3E3"
hue_red_hex= "#FF0000"
hue_orange_hex = "#FF5500"
hue_yellow_hex = "#FFD500"
hue_blue_hex = "#0055FF"
hue_green_hex = "#00FF2B"
hue_purple_hex = "#8800CC"
hue_pink_hex = "#FF00AA"
hue_max_bright = 254
hue_min_bright = 1
hue_max_saturation = 254
hue_min_saturation = 1

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
def getCertainty(lst):
    filtered_list = [item for item in lst if item is not None]
    count = Counter(filtered_list)
    if not count:
        return None, 0
    most_common, repetitions = count.most_common(1)[0]
    total_values = len(lst)
    certainty = int((repetitions / total_values) * 100)
    return most_common, certainty

# Function that extracts the desired index of x,y,z normalized coordinates from the result string:
def getCoordinates(index : int):
    # Find all matches for NormalizedLandmark entries
    matches = re.findall(r"NormalizedLandmark\(x=([-\d\.]+),\s*y=([-\d\.]+),\s*z=([-\d\.]+)", results[0])
    if matches and len(matches) > index:
        # Extract coordinates for the specified index
        x,y,z = matches[index]
        x = float(x)
        y = float(y)
        z = float(z)
    else:
        x = -1
        y = -1
        z = -1
    return [x,y,z] # return the results

# Function to return the direction that a hand last moved in the x direction
def getXMovement(lst : list, threshold : float):
    movement = 0
    if lst[0][0] - lst[-1][0] < -1*threshold:
        movement = -1
    elif lst[0][0] - lst[-1][0] > threshold:
        movement = 1
    return movement
    
# Function to return the direction that a hand last moved in the y direction
def getYMovement(lst : list, threshold : float):
    movement = 0
    if lst[0][1] - lst[-1][1] < -1*threshold:
        movement = -1
    elif lst[0][1] - lst[-1][1] > threshold:
        movement = 1
    return movement

# Function that returns a 2 element list with direction movement
# [0,0] means no movement, [-1,1] means left and up while [1,-1] means right and down
# first element is the x position (-1=left, 1=right) and second element is the y position (-1=down, 1=up)
def getMovement(lst : list, threshold : float):
    return [getXMovement(lst,0.05),getYMovement(lst,0.05)]

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

# Function to set the color of a light
def setHueColor(bridge: Hue, light_id: int, color : string):
    try:
        light = bridge.get_light(id_=light_id) # create the light from the bridge
        light.on() # make sure the light is on 
        light.set_color(hexa=color) # set the color
        light.set_brightness(hue_max_bright) # default to max brightness
    except Exception as e:
        print(f"Exception encountered: {e}")
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
    classification = ""
    last_N_landmarks = [[-1,-1,-1]]
    max_N_landmarks = 5
    movement = []
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
        common_class, certainty = getCertainty(last_N_results)
        if common_class == last_N_results[-1]: certainty = certainty
        if certainty < min_acceptable_certainty: classification = "Failure"
        else: classification = common_class
        
        if len(last_N_landmarks) > max_N_landmarks: last_N_landmarks = last_N_landmarks[1:]
        coordinates = getCoordinates(index=5) # 5th index for INDEX_FINGER_MCP position on hand
        last_N_landmarks.append(coordinates)
        movement = getMovement(last_N_landmarks,0.05)
        
        # End MP Model Processingq
        
        if DEBUG:
            # print(f'Classed Gesture: {classification}')
            print(f"Movement={movement}")
            
        # Start Hue Light Control
        
        # End Hue Light Control
        
        cv2.imshow('test', cv2.flip(frame, 1))
        if cv2.waitKey(1) == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break

# Close all open releases and captures:
capture.release()
cv2.destroyAllWindows()