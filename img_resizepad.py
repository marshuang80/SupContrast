import pandas as pd
import numpy as np
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)

TEST1 = cv2.imread('CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg', 0) #width longest
TEST2 = cv2.imread('CheXpert-v1.0/train/patient00002/study1/view2_lateral.jpg', 0) #height longest
SCALE=300


def resize_img(img, scale):
    """
    args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    #Resizing
    if max_ind == 0:
        #image is heigher
        wpercent = (scale / float(size[0]))
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        #image is wider
        hpercent = (scale / float(size[1]))
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(img, desireable_size[::-1]) #this flips the desireable_size vector
    #print(resized_img.shape)

    #Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size/2))
        right = int(np.ceil(pad_size/2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size/2))
        bottom = int(np.ceil(pad_size/2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(resized_img,[(top, bottom), (left, right)], 'constant', constant_values=0)
    #print(resized_img.shape)

    return resized_img


if __name__ == "__main__":
    resize_img(img=TEST2, scale=SCALE)



