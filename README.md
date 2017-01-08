# Behavioral Cloning Project
## Udacity Self Driving Car Class
Author: Brian Erickson

This project uses a neural network to learn to predict a drivers steering input given the camera images of the view from the car onto the road.  A driving simulator is used to record images and steering angles.  These are used to train the model. The model can be used to control the simulator.


## Files

| file          |                                                |
|---------------|------------------------------------------------
 [model.py](model.py)    | The script used to create and train the model. 
 [processing.py](processing.py) | Image preprocessing routine used by model.py and drive.py
 [drive.py](drive.py)      | The script used to drive the car, modified to include preprocessing
 [model.json](model.json)    | Model architecture                             
 [model.h5](model.h5)      | Model weights                                  
 README.md     | This file                                      

## Installation

In addition to the instructions provided in the courseware, I found that the following are required by drive.py

    conda install -c conda-forge flask-socketio=2.7.1
    conda install -c conda-forge eventlet=0.19.0

## Datasets Used

As suggested in the project assignment page, I decided to use the center images only from the simulator generated training sequences.  To handle drift, special training data were used that steer the car from off-center back to on-center.

The datasets that I used are not published as part of this repository as they are too large.

| Data Set | # of Samples   | Description|
|----------|----------------|------------|
| sample_data | 8036  |Dataset provided by Udacity along with the project assignment, includes both center and corrective steering |
| Track1Center1 | 3033 | Center driving along track 1 |
| Track1Recovery1 | 921 | Recovery driving along track 1 |
| Track1Center2 | 2736 | Center driving along track 1 |
| Track1Recovery2 | 273 | Recovery driving along track 1 |

## Preprocessing



## Training
