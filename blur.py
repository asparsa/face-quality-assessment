import cv2
import numpy as np
import sys
import logging
import os
def bluringall(img,ksize=[5,5]):
    return cv2.blur(img,ksize)

def cropface(img, text):
    real_h,real_w,c = img.shape
    try:
        x,y,w,h,score = text.strip().split(' ')
    except:
        logging.info('file is not correct.')
    try:
        w = int(float(w))
        h = int(float(h))
        x = int(float(x))
        y = int(float(y))
        w = int(w*(real_w / 224))
        h = int(h*(real_h / 224))
        x = int(x*(real_w / 224))
        y = int(y*(real_h / 224))
        y1 = 0 if y < 0 else y
        x1 = 0 if x < 0 else x 
        y2 = real_h if y1 + h > real_h else y + h
        x2 = real_w if x1 + w > real_w else x + w
        img = img[y1:y2,x1:x2,:]
        return img,y1,y2,x1,x2
    except:
        print('cropping is not correct.')
        return []

def focusbulr(img,material, ksize=[5,5]):
    img2,y1,y2,x1,x2=cropface(img,material)
    img=bluringall(img,ksize)
    img[y1:y2,x1:x2,:]=img2
    return img

def wrongfocus(img,material, ksize=[5,5]):
    img2,y1,y2,x1,x2=cropface(img,material)
    img2=bluringall(img2,ksize)
    img[y1:y2,x1:x2,:]=img2
    return img




########################testing###################
if __name__=='__main__':
    image_path="train\\1\\live\\197650.jpg"
    image_path="train\\10160\\live\\113482.jpg"
    a=cv2.imread(image_path)
    assert os.path.exists(image_path[:-4] + '_BB.txt'),'path not exists' + ' ' + image_path
    with open(image_path[:-4] + '_BB.txt','r') as f:
        material = f.readline()
    cv2.imshow('original',a)
    nb=focusbulr(a,material)
    cv2.imshow('focusblur',nb)
    df=wrongfocus(a,material)
    allblur=bluringall(a)
    cv2.imshow('all blur',allblur)
    cv2.imshow('wrong focus',df)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
