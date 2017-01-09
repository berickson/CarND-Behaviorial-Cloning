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

The following preprocessing steps are used:

The colorspace is changed from RGB to HLS.  The idea is that these are easier for the neural network to understand, especially with differences in lighting. Antedoctal testing indicated that HLS outperformed RGB in training.

The image tops and bottoms are cropped.  The idea is to get the network to concentrate on the road.  The top of the image above the horizon doesn't contain any road was removed and the bottom portion which contains the car itself was removed.  I removed the car portion in preparation for possible Left / Right / Center processing so the model wouldn't get any hints of which camera was being used.

The image is downsampled at 2:1 in both horizontal and vertical dimensions.

The preprocessing lowers the image size from 160x320x3 to 45x160x3.  The resulting images take less than 15% memory and let me get about 7 times more images in memory at once.

## Training
The following were considerations during training:
- Prevent over-fitting
- Keep training time down
- Normalization

### Preventing Overfitting
To prevent overfitting, I split my data into training and validation sets and I also applied dropout.

The training set was 80% of the samples and the validation set was 20% of the samples.  I used a training method that would keep the best model based on the loss from the validation set, even if the training set continued to show lower loss due to overfitting.  This is done automatically using the "ModelCheckpoint" callback provided by Keras.

In the model itself, I added a dropout layer with 50% retention.  This will help to reinforce more general learning and reduce overfitting.

### Keep Training Time Down
To keep the training time down, I reduced the image size by first downsizing 2:1 and then removing the sky from the image so the training could concentrate on the road.  The original image size was 160x320x3 and after manipulation, it became 50x160x3, this reduces the number of pixels from 51,200 to 8000.  With this smaller image size, I can keep more images in memory and don't need to use a slower generator function that would need to read the images from disk.