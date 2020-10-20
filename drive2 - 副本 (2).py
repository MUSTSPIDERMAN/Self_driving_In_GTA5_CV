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

model = None

def telemetry(img):
    #imgString = img
    #image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(img)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    image_array = image_array[300:400, 0:1000]
    cv2.imshow("Image2", image_array)
    image_array = cv2.resize(image_array, (128, 128)) / 255. - 0.5
    transformed_image_array = image_array[None, :, :, :]
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    return round(steering_angle*10)


#vs = cv2.VideoCapture(0)
#time.sleep(2.0)
if __name__ == '__main__':
    model = load_model('best_model999.h5')    
    last_time = time.time()
    while True:
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window',printscreen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print(telemetry(printscreen))
        

