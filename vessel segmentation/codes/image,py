import numpy as np
import cv2
import cv2.cv as cv
##from PIL import Image
##import scipy
##import scipy.io
##from matplotlib import pyplot as plt
from computebgimg import computebgimg
from LumConDrift import LumConDrift

##image = cv2.imread('C:\Users\AK PUJITHA\Desktop\iiit h\semester 6\honors project 2\Image Enahancement-Matlab\output\image010.png',1)
def imageenhancement(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    bgImg,fundusMask = computebgimg(image)
    bgImg = cv2.multiply(image[:,:,1].astype(float),bgImg.astype(float))
    ldrift,cdrift = LumConDrift(bgImg,fundusMask)

    g = image[:,:,1].astype(float)

    imgCorr = cv2.divide(cv2.subtract(g,ldrift),(cv2.add(cdrift,0.0001)))
    imgCorr = cv2.multiply(imgCorr,fundusMask.astype(float))

    imgCorr = cv2.add(imgCorr,np.abs(np.min(imgCorr)))
    imgCorr = cv2.divide(imgCorr,np.max(imgCorr))
    imgCorr = cv2.multiply(imgCorr,fundusMask.astype(float))


    image = image.astype(float)
    image[:,:,0] = cv2.divide(cv2.multiply(imgCorr,image[:,:,0]),hsv[:,:,2].astype(float))
    image[:,:,1] = cv2.divide(cv2.multiply(imgCorr,image[:,:,1]),hsv[:,:,2].astype(float))
    image[:,:,2] = cv2.divide(cv2.multiply(imgCorr,image[:,:,2]),hsv[:,:,2].astype(float))


    fundusMask = fundusMask.astype(float)
    image[:,:,0] = cv2.multiply(image[:,:,0],fundusMask)
    image[:,:,1] = cv2.multiply(image[:,:,1],fundusMask)
    image[:,:,2] = cv2.multiply(image[:,:,2],fundusMask)
    out = image[:,:,1]*255
    return out 

##cv2.namedWindow('image',1)
##cv2.imshow('image',image)
##cv2.imwrite('C:\Users\AK PUJITHA\Desktop\iiit h\semester 6\honors project 2\Image Enahancement-Matlab\output\image010_corrected.jpg',image*255)
##cv2.waitKey(0)
