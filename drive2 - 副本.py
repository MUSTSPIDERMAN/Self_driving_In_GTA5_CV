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
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
from imutils.video import VideoStream
#sio = socketio.Server()
#app = Flask(__name__)
model = None
def atoi(s):
   s = s[::-1]
   num = 0
   for i, v in enumerate(s):
      for j in range(0, 10):
         if v == str(j):
            num += j * (10 ** i)
   return num
#@sio.on('telemetry')
def telemetry(img):

    imgString = img
    #image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    image_array = image_array[300:400, 0:1000]
    cv2.imshow("Image2", image_array)
    image_array = cv2.resize(image_array, (128, 128)) / 255. - 0.5
    transformed_image_array = image_array[None, :, :, :]

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
	
    
    return round(steering_angle*10)


import PIL.Image as Image

 



vs = cv2.VideoCapture(0)
time.sleep(2.0)
if __name__ == '__main__':
    model = load_model('best_model777.h5')    
    while True:
        ret, image=vs.read()
       # cv2.imwrite("image_read2.jpg", image)
       # image=cv2.imread("image_read2.jpg")
        #image= transparent_back(image)
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        print(telemetry(image))
        
   # app = socketio.Middleware(sio, app)    
   # eventlet.wsgi.server(eventlet.listen(('', 4567)), app)   
