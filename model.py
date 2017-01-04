# manually transcribed and modified from 
# https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.u8zq6ghon
def nvidia_model():
    model = Sequential()
    model.add(Lamda(lambda x: x/127.5 -1., input_shape = input_shape))
    model.add(Convolution2d(24, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init="he_normal"))
    model.add(RELU())
    model.add(Dense(1164, init="he_normal"))
    model.add(RELU())
    model.add(Dense(100, init="he_normal"))
    model.add(RELU())
    model.add(Dense(50, init="he_normal"))
    model.add(RELU())
    model.add(Dense(10, init="he_normal"))
    model.add(RELU())
    model.add(Dense(1, init="he_normal"))
    return model

