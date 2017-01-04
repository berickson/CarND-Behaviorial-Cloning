# Behavioral Cloning Project
## Udacity Self Driving Car Class
Author: Brian Erickson

This project uses a neural network to learn to predict a drivers steering input given the camera images of the view from the car onto the road.  A driving simulator is used to record images and steering angles.  These are used to train the model. The model can be used to control the simulator.


## Files
model.py - The script used to create and train the model.
drive.py - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications and submit your modified version.
model.json - The model architecture.
model.h5 - The model weights.
README.md - explains the structure of your network and training approach. While we recommend using English for good practice, writing in any language is acceptable (reviewers will translate). There is no minimum word count so long as there are complete descriptions of the problems and the strategies. See the rubric for more details about the expectations.


## Installation
In addition to the instructions provided in the courseware, I found that the following are required by drive.py

    conda install -c conda-forge flask-socketio=2.7.1
    conda install -c conda-forge eventlet=0.19.0