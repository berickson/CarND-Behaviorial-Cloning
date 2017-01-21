# Behavioral Cloning Project
## Udacity Self Driving Car Class
Author: Brian Erickson

This project uses a neural network to learn to predict a drivers steering input given the camera images of the view from the car onto the road.  A driving simulator is used to record images and steering angles.  These are used to train the model. The model can be used to control the simulator.

Video recording of successfully navigating both tracks [here](https://youtu.be/5NqPmhm5s3I)


## Files

| file          |                                                |
|---------------|------------------------------------------------
  [model.py](model.py)    | The script used to create and train the model. 
 [preprocess.py](preprocess.py) | Image preprocessing routine used by model.py and drive.py
 [drive.py](drive.py)      | The script used to drive the car, modified to include preprocessing
 [model.json](model.json)    | Model architecture
 [model.h5](model.h5)      | Model weights
 README.md     | This file
[project.ipynb](project.ipynb)    | I python notebook that I would use to command training.
 
## Installation

In addition to the instructions provided in the courseware, I found that the following are required by drive.py

    conda install -c conda-forge flask-socketio=2.7.1
    conda install -c conda-forge eventlet=0.19.0

## Datasets Used

I used the simulator to record my own images.  I recorded on both tracks.  I recorded a combination of center driving and recovery driving where I would start recording while the camera was off-center, and drive to center to correct.  This gave the model example of what to do when the car would invariablyd drift off center.

To aid in training, I used the left and right images with a small offset angle, so that the car would know what to do if it was too far right or too far left.  This basically tripled my training data.

The datasets that I used are not published as part of this repository as they are too large.


| Data Set | # of Samples   | Description|
|----------|----------------|------------|
| Track1Center3 | 15789 | Center driving along track 1 |
| Track1Recovery3 | 7380 | Recovery driving along track 1 |
| Track2Center1 | 1091 | Center driving along track 2 |
| Track2Center2 | 19267 |  Center driving along track 2 |
| Track2Center3 | 11403 |  Center driving along track 2 |
| Track2Recovery1 | 7899 | Recovery driving along track 1 |
| Track2Recovery2 | 7137 | Recovery driving along track 1 |

In total, 69966 training samples were used.

## Preprocessing

The following preprocessing steps are used:

The colorspace is changed from RGB to HLS.  The idea is that these are easier for the neural network to understand, especially with differences in lighting. Antedoctal testing indicated that HLS outperformed RGB in training.

The image tops and bottoms are cropped.  The idea is to get the network to concentrate on the road.  The top of the image above the horizon doesn't contain any road was removed and the bottom portion which contains the car itself was removed.  I removed the car portion in preparation for possible Left / Right / Center processing so the model wouldn't get any hints of which camera was being used.

The image is downsampled at 2:1 in both horizontal and vertical dimensions.

The preprocessing lowers the image size from 160x320x3 to 45x160x3.  I also normalized the images in the pre-processing step and saved them into float16 numpy arrays.  

## Training
The following were considerations during training:
- Prevent over-fitting
- Keep training time down
- Normalization

### Preventing Overfitting
To prevent overfitting, I split my data into training and validation sets and I also applied dropout.

The training set was 80% of the samples and the validation set was 5% of the samples.  I used a training method that would keep the best model based on the loss from the validation set, even if the training set continued to show lower loss due to overfitting.  This is done automatically using the "ModelCheckpoint" callback provided by Keras.

In the model itself, I added a dropout layer with 50% retention.  This helped to reduce overfitting for a more robust model.

During training, I never really saw indications of overfitting, such as decreasing training loss with increasing validation loss, even if I ran the model for many epochs.  I suspect this is mainly because I used 50% dropout and a relatively large input set that would tend to resist "memorization" by the model.

### Keep Training Time Down
To keep the training time down, I reduced the image size by first downsizing 2:1 and then removing the sky from the image so the training could concentrate on the road.  The original image size was 160x320x3 and after manipulation, it became 50x160x3, this reduces the number of pixels from 51,200 to 8000.  With this smaller image size, I can keep more images in memory and don't need to use a slower generator function that would need to read the images from disk.

## Maintaining Speed
While it wasn't required, I wanted my model to work on both track 1 and track 2.  Track 2 is very hilly and I chose to implement a PID for speed control so a more constant speed could be maintained whether going uphill, downhill, or on a flat surface.
