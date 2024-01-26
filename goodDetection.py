import mediapipe as mp
import numpy as np
import datetime as dt
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "gesture_recognizer.task" # assume .task file in same folder as program

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

gresult = ["hello test"]

# Create a gesture recognizer instance with the live stream mode:
def getResult(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    gresult.append(result)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=getResult)

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0) 
    ts = 0 # start frame counter
    if not cap.isOpened():
        cap.release()
        print("Cannot open video device.")
        exit()
        
    while True:
        ret, frame = cap.read() 
        if not ret: break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = recognizer.recognize_async(mp_image,ts)
        local_result = format(gresult[-1])
        print(local_result.gestures[0].categoryName)
        ts = ts + 1 # increment frame counter
        
        # Add the overlay
        text1 = "Unknown"
        text2 = "Unknown"
        cv2.putText(frame, text1, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(frame, text1, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        # Display the image and handle breaking
        cv2.imshow('goodDetection', frame)
        if cv2.waitKey(1) == ord('q'): 
            cap.release() # release the capture at the end
            cv2.destroyAllWindows() # clean and destroy opened windows
            break