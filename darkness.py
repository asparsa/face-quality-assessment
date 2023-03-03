import cv2
import numpy as np
def darker(img,val):
    factor=np.ones(img.shape,dtype='uint8')*val
    return cv2.subtract(img,factor)
def brighter(img, val):
    factor=np.ones(img.shape,dtype='uint8')*val
    return cv2.add(img,factor)
if __name__=='__main__':
    image_path="train\\1\\live\\197650.jpg"
    a=cv2.imread(image_path)
    cv2.imshow('original',a)
    cv2.imshow('light',brighter(a,100))
    cv2.imshow('darker',darker(a,100))
    cv2.waitKey(0)
    cv2.destroyAllWindows()