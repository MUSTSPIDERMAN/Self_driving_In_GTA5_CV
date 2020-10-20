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
        #img = pyautogui.screenshot(region=[0, 500, 1250, 350])  # x,y,w,h    1280*960 resolution
        img = pyautogui.screenshot(region=[0, 10, 1800, 840])  # x,y,w,h
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
        #    PressKey(D)
        #    ReleaseKey(A)
            #ReleaseKey(W)
          
            #ReleaseKey(W)
            #PressKey(W)
            #ReleaseKey(D)
            print("right")
        #ReleaseKey(W)
        if decision < 0:
        #    PressKey(A)
            #ReleaseKey(W)
            ##PressKey(W)
            #ReleaseKey(D)
            #ReleaseKey(W)
            ##PressKey(W)
            #ReleaseKey(A)
            print("left")
        #YOLOY---------------------
        #image=printscreen
        printscreen=cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)
        image = printscreen[200:700, 100:900]
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
	     	# extract the class ID and confidence (i.e., probability) of# the current object detection 
                scores = detection[5:] 
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
		           # scale the bounding box coordinates back relative to the
		        	# size of the image, keeping in mind that YOLO actually
		    	    # returns the center (x, y)-coordinates of the bounding
		    	    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    print(centerX, centerY, width, height)
		        	# use the center (x, y)-coordinates to derive the top and
		        	# and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
		        	# update our list of bounding box coordinates, confidences,
		        	# and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            
	    # loop over the indexes we are keeping
            for i in idxs.flatten():
	        	# extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

		        # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                print("ckassIDs:",classIDs[i],"i",i)
                if classIDs[i] == 2 or classIDs[i] == 7 :
                    if x<390 and y<130 and (x+w)>700 and (y+w)>450:
                        PressKey(S)
                        ReleaseKey(W)
                        color=(0,0,255)
                        print("Warning!!!!!!!!")
                    else:
                        ReleaseKey(S)
                          
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)    
                print("x y:",x,y,"x+w y+w",x+w,y+w)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

       # show the output image
        image=cv2.resize(image,(625,420))
        cv2.imshow("Image", image)
       # ReleaseKey(S)
        #YOLO--------------------
       # ReleaseKey(D)
       # ReleaseKey(A)
       
       # PressKey(W)
       

        

