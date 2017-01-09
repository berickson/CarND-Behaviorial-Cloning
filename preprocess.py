import cv2

def preprocess_image(im):
    #return im
    new_size=(im.shape[1]//2,im.shape[0]//2)
    im=cv2.resize(im,(new_size),interpolation=cv2.INTER_AREA)
    im=im[30:,:,:]
    #im=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    return  ((im.astype(float) - 128.)/255.)
