from huesdk import Discover
from huesdk import Hue
import mediapipe as mp
import numpy as np
import datetime as dt
import string
import cv2
import time
import re
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter

# Initialize MediaPipe Hands for color swiping:
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Configure Variables for Hue Lights Interaction:
bridge_ip = "192.168.1.148"
bridge_username = "ysFHipKKPahizAwVKB8zYJlpPbVc4tyFBLF6MJDg"
# hue = Hue(bridge_ip=bridge_ip, username=bridge_username) # create the hue object
# light_corner = hue.get_light(id_=1) # get light 1 created
# light_bed = hue.get_light(id_=2) # get light 2 created

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

# Test the code with visualizer here:
# First prepare the capture device:
cap = cv2.VideoCapture(0) # get video capture active
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
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process the image and draw hand landmarks
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Display the image
    cv2.imshow('MediaPipe Hands', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
# at the end, release and destroy frames:
cap.release() # release the capture at the end
cv2.destroyAllWindows() # clean and destroy opened windows