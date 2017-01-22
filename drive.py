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

from model import preprocess_image

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

class PID:
    def __init__(self, sp, ksp=0.01, kp=0.2, ki=0.0001, kd=0.0, max_i=300.):
        self.sp = sp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ksp = ksp
        self.i = 0.0
        self.max_i = max_i
        self.last_v = 0.0
        
    def update(self, v):
        e = self.sp - v
        d = v - self.last_v
        self.last_v = v
        self.i = clamp(self.i + e,-self.max_i, self.max_i)
        
        return self.sp*self.ksp + self.kp * e + self.ki * self.i + self.kd * d
        
pid = PID(sp = 25)
        
        

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array=preprocess_image(image_array)
    transformed_image_array = image_array[None, :, :, :]
    #print('image size',transformed_image_array.shape)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    #print(".")
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #throttle = 0.3
    throttle = pid.update(speed)
    
    print(steering_angle, throttle)
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
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
