# Behavioral Cloning Project
## Udacity Self Driving Car Class
Author: Brian Erickson

This project uses a neural network to learn to predict a drivers steering input given the camera images of the view from the car onto the road.  A driving simulator is used to record images and steering angles.  These are used to train the model. The model can be used to control the simulator.


## Files

| file          |                                                |
|---------------|------------------------------------------------
 [model.py](model.py)    | The script used to create and train the model. 
 [drive.py](drive.py)      | The script used to drive the car                                                 
 [model.json](model.json)    | Model architecture                             
 model.h5      | Model weights                                  
 README.md     | This file                                      

## Installation

In addition to the instructions provided in the courseware, I found that the following are required by drive.py

    conda install -c conda-forge flask-socketio=2.7.1
    conda install -c conda-forge eventlet=0.19.0

## Datasets Used


## Training
