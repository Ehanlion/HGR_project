# Ethan Owen, 3/12/2024
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
DEBUG_GENERAL = False # controls general debug outputs
DEBUG_LOCATION = False # controls the location-specific debug outputs
DEBUG_COLORS = True # control the color/brightness/saturation changing debug outputs
COMPLEX_GUI = True # controls whether or not to draw augmented fields on the output frame
DRAW_HANDS = False # controls whether or not to draw hand landmarks CAUTION: severe fps impact

# Configure global data:
path = "Models/gesture_recognizer.task"  # set the model's path
rawModelResults = [""] # global tracker for raw results directly extract from the GestureRecognizer model from MediaPipes
classifiedResults = [] # global tracker for all classified results

maxClassifiedResults = 20 # max amount of classified gestures to keep stored at one time
if DRAW_HANDS:
    maxClassifiedResults = 10 # change if drawing hands since frame time effect this
    
minClassifyThreshold = 35 # threshold for calling a gesture a gesture, values between 0-100 (integer)
if DRAW_HANDS:
    minClassifyThreshold = 20 # change if drawing hands since frame time effect this

# ``
# ``
# ``

# Start Setup of Hue Light Objects

light_corner = None # hue bulb in corner of room
light_bed = None # hue bulb by bedside
light_computer = None # hue spotlight by computer
light_bookcase = None # hue spotlight by bookcase
hue = None # empty hue bridge object for redundancy
lightList = []

# define the bridge information, local just to my project
bridge_ip = "192.168.1.148"
bridge_username = "ysFHipKKPahizAwVKB8zYJlpPbVc4tyFBLF6MJDg"

try:
    hue = Hue(bridge_ip=bridge_ip, username=bridge_username) # create the hue bridge object
    
    # Get and print light data for debug purposes
    if DEBUG_GENERAL:
        lights = hue.get_lights()
        for light in lights:
            print(f"ID: {light.id_}")
            print(f"    Name: {light.name}")
            print(f"    Brightness: {light.bri}")
            print(f"    Hue: {light.hue}")
            print(f"    Saturation: {light.sat}")
    
    # Create all of the lights
    light_corner = hue.get_light(id_=1)
    light_bed = hue.get_light(id_=2)
    light_computer = hue.get_light(id_=3)
    light_bookcase = hue.get_light(id_=4)
    
    # form the light list
    lightList.append(light_corner)
    lightList.append(light_bed)
    lightList.append(light_computer)
    lightList.append(light_bookcase)
    
except Exception as e:
    print(f"Encountered Exception loading Hue Object: {e}")
    exit()

# End Setup of Hue Light Objects

# ``
# ``
# ``

# Start Pre-declared Hue Values:

# color hex codes, alternate main color then 'linking' color in declarations
hue_white_hex = "#E5E5E5"
hue_lightRed = "#FF6666"
hue_red_hex= "#FF0000"
hue_amber_hex = "#FF5500"
hue_orange_hex = "#FF8000"
hue_burntSienna_hex = "#996600"
hue_yellow_hex = "#CCAA00"
hue_limeGreen_hex = "#AACC00"
hue_green_hex = "#669900"
hue_teal_hex = "#00CC88"
hue_blue_hex = "#004C99"
hue_darkLavender_hex = "#330099"
hue_purple_hex = "#6600CC"
hue_darkPurple_hex = "#550066"
hue_pink_hex = "#990066"
hue_lightRose_hex = "#FF99BB"

# brightness and saturation values
hue_max_bright = 254
hue_min_bright = 1
hue_max_saturation = 254
hue_min_saturation = 1

# list of all hue colors for color swiping:
hueColors = [hue_white_hex, hue_lightRed, 
             hue_red_hex, hue_amber_hex,
             hue_orange_hex, hue_burntSienna_hex,
             hue_yellow_hex, hue_limeGreen_hex,
             hue_green_hex, hue_teal_hex ,
             hue_blue_hex, hue_darkLavender_hex,
             hue_purple_hex, hue_darkPurple_hex,
             hue_pink_hex, hue_lightRose_hex]

# End Pre-declared Hue Values

# ``
# ``
# ``

# Start Configure Gesture Sequences
# - Idea: use Unknown class of gesture as a 'break' between gestures
#         If we see 'Unknown,' look for next gesture in a sequence and re-check?

# master list of all gestures names according to output of the MediaPipe GestureRecognizer Model
gesture_unknown = "Unknown"
gesture_closedFist = "Closed_Fist"
gesture_openPalm = "Open_Palm"
gesture_pointingDown = "Pointing_Down"
gesture_pointingUp = "Pointing_Up"
gesture_thumbUp = "Thumb_Up"
gesture_thumbDown = "Thumb_Down"
gesture_victory = "Victory"
gesture_iLoveYou = "ILoveYou"

gs_lights_on = [] # gs for turning all lights on
gs_lights_off = [] # gs for turning all lights off
gs_lights_brighten = [] # gs for brightening the lights
gs_lights_dim = [] # gs for dimming the lights

# End Configure Gesture Sequences

# ``
# ``
# ``

# Configure base options for the MediaPipes framework:
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ``
# ``
# ``

# ==================================
# Start Generic Function Definitions

# Create a gesture recognizer instance with the live stream mode:
def store_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    rawModelResults.append(str(format(result))) # append and format result to results

# Function to extract the gesture class and handedness from the result string:
def classify_gesture():
    gesture_pattern = r"gestures=\[\[Category\(.*?category_name='([^']*)'\)\]\]"
    gesture_match = re.search(gesture_pattern, rawModelResults[-1])
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
    matches = re.findall(r"NormalizedLandmark\(x=([-\d\.]+),\s*y=([-\d\.]+),\s*z=([-\d\.]+)", rawModelResults[0])
    if matches and len(matches) > index:
        # Extract coordinates for the specified index
        x,y,z = matches[index]
        x = float(x)
        y = float(y)
        z = float(z)
    else:
        return None
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

# Function to calculate current fps and returns average and previous times:
def get_fps(prev):
    curr = time.time()
    fps = 1 / (curr - prev)
    return int(fps), curr

# End Generic Function Definitions
# ==================================

# ``
# ``
# ``

# ==========================================
# Start Device Control Function Definitions:

# state pass for function takes on/off: 0 = off, 1 = on, other = fail
# bridge pass takes bridge: pass a bridge object that was created
# color pass takes pre-def int: pass either HUE_RED, HUE_BLUE, HUE_GREEN

# Function to set the state of a light to either on or off:
def setLightState(bridge: Hue, light: Hue.get_light, state: int):
    try:
        # Set light states:
        if state == 0:
            light.off()
        else:
            light.on()
            light.set_brightness(hue_max_bright) # default to max brightness
    except Exception as e:
        print(f'Exception encountered in setLightState: {e}')
        return 1
    return 0

# Function to set the color of a light
def setLightColor(bridge: Hue, light: Hue.get_light, color : string):
    try:
        light.on() # make sure the light is on 
        light.set_color(hexa=color) # set the color
        light.set_brightness(hue_max_bright) # default to max brightness
    except Exception as e:
        print(f"Exception encountered in setColorLight: {e}")
        return 1
    return 0

# Function to set the brightness of a light (0=off, 254=max_bright)
def setLightBrightness(bridge : Hue, light : Hue.get_light, brightness : int, state : int):
    try:
        if state == 0:
            light.off()
        else:
            light.on()
        light.set_brightness(value=brightness) # set the light to brightness 
    except Exception as e:
        print(f"Exception encountered in setLightBrightness: {e}")
        return 1
    return 0

# Function to set the saturation of a light (0=min, 254=max_saturation)
def setLightSaturation(bridge : Hue, light : Hue.get_light, saturation : int, state : int):
    try:
        if state == 0:
            light.off()
        else:
            light.on()
        light.set_saturation(value=saturation) # set the light to brightness 
    except Exception as e:
        print(f"Exception encountered in setLightSaturation: {e}")
        return 1
    return 0

# End Device Control Function Definitions
# ==========================================

# ``
# ``
# ``+

# Configure local GestureRecognizerOptions for our gesture recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=store_result)

# ``
# ``
# ``

# Start the main iteration loop:
# - This loop basically holds all the program function and iteration

with GestureRecognizer.create_from_options(options) as recognizer:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened(): 
        capture.release()
        cv2.destroyAllWindows()
        exit()
    
    # Start Variable Declaration
    
    # general fields:
    mpTimestamp = 0 # stores only-increasing amount of timestamps for OpenCV and MediaPipe purposes 
    gestureClassification = "" # stores the last classification result of looking for a hand gesture
    gestureCertainty = 0.0 # stores the certainty in any given gesture on a scale of     
    
    # hand landmark fields:
    landmarks = [[0,0,0]] # stores the last hand location
    maxLandmarks = 3 # sets the max allowed store hand locations
    
    # movement-related fields:
    lastMovement = [0,0] # store information about which direction the hand last moved in
    movements = [[0,0]] # store last N hand location movements
    maxMovements = 20 # stores the maximum allowed number of hand movements (10 worked well)
    averageMovement_x = 0.0 # stores the average of the last N x movements
    averageMovement_y = 0.0 # store the average of the last N y movements
    
    # color/brightness fields:
    colorIndex = 0 # stores the index of hueLights list of colors that can be selected from 
    colorBrightness = light_corner.bri # get current brightness
    colorSaturation = hue_max_saturation # set color saturation as max saturation for ease 
    colorSwipingThreshold = 0.101 # set the color sweeping threshold for when we actually swipe between colors
    brightnessIncrement = 50 # set the brightness increment for changing brightness
    saturationIncrement = 50 # set the saturation increment for changing light saturation 
    
    # fps fields:
    prev_ft = 0 # stores the prior frame time
    fps = 0 # stores the frames per second that the program is running at
    
    # conditional fields to draw hands:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils    
    
    # End Variable Declaration
    
    while True:
        ret, frame = capture.read()
        frame_cpy = frame
        if not ret: break
        
        # Start MP Model Processing
        
        # extract the image and store the raw results from the GestureRecognitionModel
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = recognizer.recognize_async(image,mpTimestamp)
        gestureClassification = classify_gesture()
        classifiedResults.append(gestureClassification)
        rawModelResults = rawModelResults[-1:]
        mpTimestamp = mpTimestamp + 1
        
        # determine the class of gesture detected in this loop cycle
        if len(classifiedResults) > maxClassifiedResults: classifiedResults = classifiedResults[1:]
        common_class, gestureCertainty = getCertainty(classifiedResults)
        if common_class == classifiedResults[-1]: gestureCertainty = gestureCertainty
        if gestureCertainty < minClassifyThreshold: gestureClassification = "Unknown" # set classification to "Unknown" if we don't get good certainty
        else: gestureClassification = common_class
        
        # This section draws the MediaPipe hands to the frame if the flag is true, very slow render time
        if DRAW_HANDS:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame color
            mpHandResults = hands.process(frame) # process the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # reconvert frame color
            
            # if hand landmarks are found, iterate through landmarks and draw them on the frame
            if mpHandResults.multi_hand_landmarks:
                for hand_landmarks in mpHandResults.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # analyze the hand landmarks that the model saves and use it for color/brightness calculations
        if len(landmarks) > maxLandmarks: landmarks = landmarks[1:]
        single_coordinate = getCoordinates(index=5) # 5th index for INDEX_FINGER_MCP position on hand
        if single_coordinate is not None: landmarks.append(single_coordinate)
        lastMovement = getMovement(landmarks,0.05)
        movements.append(lastMovement) # add the last movement to the list 
        
        # Conditional branch to resolve interactions.
        if len(movements) > maxMovements - 1: 
            x_sum = 0.0 # track the sum of x values
            y_sum = 0.0 # track the sum of y values
            for m in movements:
                x_sum += m[0] # add x value to x list
                y_sum += m[1] # add y value to y list
            averageMovement_x = x_sum / len(movements) # compute the x average
            averageMovement_y = y_sum / len(movements) # compute the y average
            movements.clear() # clear the list
            movements.append(lastMovement) # re-append the movement
        else:
            averageMovement_x = 0.0 # reset the x average movement to 0
            averageMovement_y = 0.0 # reset the y average movement to 0
        
        # End MP Model Processing
        
        # printing fields for debug work
        if DEBUG_GENERAL:
            if averageMovement_x != 0.0 or averageMovement_y != 0.0:
                print(f"x_avg_mov={averageMovement_x}, y_avg_mov={averageMovement_y}")
            # print(f"Gesture_Class={classification}, Movement={movement}, avg_x_mov={x_averageMovement}, avg_y_mov={y_averageMovement}, lenLastNMovement={len(last_N_movements)}")
            
        if DEBUG_LOCATION:
            print(f"coord={single_coordinate}")
            
        # ========================
        # Start Hue Light Control
        # gesture mapping:
        #   - thumbs_up = light on
        #   - thumbs_down = light off
        #   - open_palm = control colors
        #   - closed_fist = control brightness
        
        # Enter here to simply turn the lights on:
        if gestureClassification == gesture_thumbUp: # enter if turn lights on 
            for l in lightList:
                setLightState(Hue,l,1)
        
        # Enter here to simply turn the lights off:
        elif gestureClassification == gesture_thumbDown: # enter if turn lights off
            for l in lightList:
                setLightState(Hue,l,0)
                
        # Enter here for color swiping:
        elif gestureClassification == gesture_openPalm and averageMovement_x != 0.0: # enter color changing section
            # Check to see if the x movement is negative = hand moves LEFT
            if averageMovement_x < -1*colorSwipingThreshold:
                colorIndex = (colorIndex-1)%len(hueColors)  # calculate index accounting for wrapping edges
                for l in lightList:
                    setLightColor(Hue,l,color=hueColors[colorIndex]) # set color based on new index
                    
                # debug code for seeing changes
                if DEBUG_COLORS:
                    print(f"CLR: Move_X={averageMovement_x}, Hand_LEFT, newColor={hueColors[colorIndex]}")
                    
                averageMovement_x = 0.0 # reset movement
            
            # Check to see if the x movement is positive = hand moves RIGHT
            elif averageMovement_x > colorSwipingThreshold:
                colorIndex = (colorIndex+1)%len(hueColors) # calculate index accounting for wrapping edges
                for l in lightList:
                    setLightColor(Hue,l,color=hueColors[colorIndex]) # set color based on new index
                
                # debug code for seeing changes
                if DEBUG_COLORS:
                    print(f"CLR: Move_X={averageMovement_x}, Hand_RIGHT, newColor={hueColors[colorIndex]}")
                    
                averageMovement_x = 0.0 # reset movement
        
        # Enter here for Brightness modulation:
        elif gestureClassification == gesture_closedFist and averageMovement_y != 0.0:
            # Check to see if the y movemenent is negative = hand moved DOWN
            if averageMovement_y < -1*colorSwipingThreshold:
                colorBrightness-=brightnessIncrement # decrement the brightness by the brightness increment
                
                # Check to see if we go below min allowed value
                if colorBrightness < hue_min_bright: 
                    colorBrightness = hue_min_bright # if we did, set requested brightness to min
                    
                    # instead now, interate over lights and simply turn them off if a value below min requested
                    for l in lightList:
                        setLightState(hue,l,0)
                        
                # Otherwise, brightness value valid and we should decrease brightness of each light
                else:
                    for l in lightList:
                        setLightBrightness(hue,l,colorBrightness,1) # decrease the brightness here
                        
                # debug code for seeing changes
                if DEBUG_COLORS:
                    print(f"BRI: Move_Y={averageMovement_y}, Hand_DOWN, newBrightness={colorBrightness}")
                    
                averageMovement_y = 0.0 # reset movement
                
            # Check to see if the y movement is positive = hand moved UP
            elif averageMovement_y > colorSwipingThreshold:
                colorBrightness+=brightnessIncrement # increment the brightness by the brightness increment
                
                # Check to see if we go above the max allowed value
                if colorBrightness > hue_max_bright: 
                    colorBrightness = hue_max_bright # if we did, set current brightness to max value
                    
                # unlike for low brightness, for high just set as max value if higher than max value suggested
                for l in lightList:
                    setLightBrightness(hue,l,colorBrightness,1) # increase brightness here
                    
                # debug code for seeing changes
                if DEBUG_COLORS:
                    print(f"BRI: Move_Y={averageMovement_y}, Hand_UP, newBrightness={colorBrightness}")
                    
                averageMovement_y = 0.0 # reset movement
        
        # Enter here for the Saturation modulation:
        elif gestureClassification == gesture_iLoveYou and averageMovement_y != 0.0:
            # Check to see if the y movemenent is negative = hand moved DOWN
            if averageMovement_y < -1*colorSwipingThreshold:
                colorSaturation-=saturationIncrement # decrement the saturation by the saturation increment
                
                # Check to see if we go below min allowed value
                if colorSaturation < hue_min_saturation: 
                    colorSaturation = hue_min_saturation # if we did, set requested brightness to min
                    
                # iterate through lights and change the saturation 
                for l in lightList:
                    setLightSaturation(hue,l,colorSaturation,1) # decrease the saturation here
                    
                # debug code for seeing changes
                if DEBUG_COLORS:
                    print(f"SAT: Move_Y={averageMovement_y}, Hand_DOWN, newSaturation={colorBrightness}")
                    
                averageMovement_y = 0.0 # reset movement
                
            # Check to see if the y movement is positive = hand moved UP
            elif averageMovement_y > colorSwipingThreshold:
                colorSaturation+=saturationIncrement # increment the saturation by the saturation increment
                
                # Check to see if we go above the max allowed value
                if colorSaturation > hue_max_saturation: 
                    colorSaturation = hue_max_saturation # if we did, set current saturation to max value
                    
                # iterate through lights and change the saturation
                for l in lightList:
                    setLightSaturation(hue,l,colorSaturation,1) # increase saturation here

                # debug code for seeing changes
                if DEBUG_COLORS:
                    print(f"SAT: Move_Y={averageMovement_y}, Hand_DOWN, newSaturation={colorBrightness}")
                    
                averageMovement_y = 0.0 # reset movement
            
        # End Hue Light Control
        # ========================
        
        # Show the frame
        
        frame = cv2.flip(frame,1)
        
        # only waste time calculating the fps if we are displaying complex gui
        if COMPLEX_GUI:
            fps, prev_ft = get_fps(prev_ft)
            cv2.putText(frame, f"FPS: {fps}", (5,55), cv2. FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
        
        cv2.putText(frame, f"Class: {gestureClassification} {gestureCertainty}", (5,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Home Gesture Recognition', frame)
        if cv2.waitKey(1) == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break
        
        while False:
            if cv2.waitKey(1) == ord('c'):
                break
            if cv2.waitKey(1) == ord('q'):
                capture.release()
                cv2.destroyAllWindows()
                break

# End the main iteration loop:

# ``
# ``
# ``

# Finally, clean up all the OpenCV windows once the program ends to make sure nothing is left open
# - Also make sure to clear any lists just in case, idk how Python3 handles memory leaks but this has to run on RPI

capture.release()
cv2.destroyAllWindows()
rawModelResults.clear()
classifiedResults.clear()
landmarks.clear()