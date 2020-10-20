# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:59:39 2020

@author: 11037
"""

import argparse
import base64
import json
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from PIL import ImageGrab
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
from imutils.video import VideoStream
from directkeys import PressKey,ReleaseKey, W, A, S, D

import numpy as np
import os 


import pyautogui
import cv2
import numpy as np

model = None

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ##PressKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    #PressKey(W)
   # ReleaseKey(D)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def telemetry(img):
    #imgString = img
    #image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(img)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    image_array = image_array[200:700, 100:900]
    #cv2.imshow("Image2", image_array)
    #cv2.waitKey(1)
    image_array = cv2.resize(image_array, (128, 128)) / 255. - 0.5
    transformed_image_array = image_array[None, :, :, :]
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    return round(steering_angle*10)


    

#vs = cv2.VideoCapture(0)
#time.sleep(2.0)
if __name__ == '__main__':    
    args={'confidence':0.5,'threshold':0.3}
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    
    # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join(["yolo-coco ", "yolov3.weights"])
    weightsPath = "yolo-coco/yolov3.weights"
    # configPath = os.path.sep.join(["yolo-coco ", "yolov3.cfg"])
    configPath = "yolo-coco/yolov3.cfg"
    
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    #-------------------------------------
    model = load_model('best_model777.h5')    
    last_time = time.time()
    while True:
        #ReleaseKey(W)
        #printscreen =  np.array(ImageGrab.grab(bbox=(0,400,1500,1200)))
        img = pyautogui.screenshot(region=[0, 500, 1250, 350])  # x,y,w,h
        printscreen = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        
        #print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        #cv2.imshow('window',printscreen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        decision=telemetry(printscreen)
        print(decision)
        if decision > 0:
            PressKey(D)
            ReleaseKey(A)
            #ReleaseKey(W)
          
            #ReleaseKey(W)
            #PressKey(W)
            #ReleaseKey(D)
            print("right")
        #ReleaseKey(W)
        if decision < 0:
            PressKey(A)
            #ReleaseKey(W)
            ##PressKey(W)
            ReleaseKey(D)
            #ReleaseKey(W)
            ##PressKey(W)
            #ReleaseKey(A)
            print("left")

       # show the output image
        img2=cv2.resize(printscreen,(500,140))
        cv2.imshow("Image", img2)
       # ReleaseKey(S)
        #YOLO--------------------
       # ReleaseKey(D)
       # ReleaseKey(A)
       
       # PressKey(W)
       #--------

