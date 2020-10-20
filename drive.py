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

sio = socketio.Server()
app = Flask(__name__)
model = None
def atoi(s):
   s = s[::-1]
   num = 0
   for i, v in enumerate(s):
      for j in range(0, 10):
         if v == str(j):
            num += j * (10 ** i)
   return num
@sio.on('telemetry')
def telemetry(sid, data):
    steering_angle = data["steering_angle"]
    throttle = data["throttle"]
    speed = data["speed"]
   # print("speed:",speed)
    imgString = data["image"]
    #cv2.imshow('window',imgString)
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    image_array = image_array[80:140, 0:320]
    #cv2.imshow('window',image_array)
    image_array = cv2.resize(image_array, (256, 256)) / 255. - 0.5
    transformed_image_array = image_array[None, :, :, :]

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
	#设置自动驾驶时的油门量
    if (atoi(speed)/100000) < 26:
        throttle = 0.5
    if abs(steering_angle) > 0.12:
         if (atoi(speed)/100000) > 26:
             throttle = -1
    if (atoi(speed)/100000) > 26:
        throttle = -1
   # print("atoi",atoi(speed)/100000)
   # print("test!!!!!", atoi(int(steering_angle)))
    print("angle",int(steering_angle) )
   # print(int(steering_angle) > 0)
    #print(steering_angle, throttle)
    print(steering_angle*100)
    print(round(steering_angle*100))
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    model = load_model('best_modelzzz.h5')    
    app = socketio.Middleware(sio, app)    
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)   
