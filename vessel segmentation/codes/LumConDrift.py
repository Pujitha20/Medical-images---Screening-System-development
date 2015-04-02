import numpy as np
import cv2
import cv2.cv as cv
##from PIL import Image
##import scipy
##import scipy.io
##from matplotlib import pyplot as plt
##
def LumConDrift(bgImg,fundusMask): 
    m,n = bgImg.shape
    tsize = 50
    indx=0
    indy=0
    i = tsize
   
    ldrift = np.zeros((int(m/tsize),int(n/tsize)),np.float)
    cdrift = np.zeros((int(m/tsize),int(n/tsize)),np.float)
    while(i<m):
        j = tsize
        while(j<n):           
            if (i+tsize>=m and j+tsize<n):
                block = bgImg[i-tsize:m, j-tsize:j+tsize]
            elif (i+tsize<m and j+tsize>=n):
                block = bgImg[i-tsize:i+tsize, j-tsize:n]
            elif (i+tsize>=m and j+tsize>=n):
                block = bgImg[i-tsize:m, j-tsize:n]
            else :
                block = bgImg[i-tsize:i+tsize, j-tsize:j+tsize]
            mean,std = cv2.meanStdDev(block)
            ldrift[indx,indy] = mean
            cdrift[indx,indy] = std
            indy = indy+1
            j = j+tsize
        indy = 0
        indx = indx+1
        i = i+tsize
    ldrift = cv2.resize(ldrift,(n,m),interpolation = cv2.INTER_CUBIC)
    cdrift = cv2.resize(cdrift,(n,m),interpolation = cv2.INTER_CUBIC)
    ldrift = cv2.multiply(ldrift,fundusMask.astype(float))
    cdrift = cv2.multiply(cdrift,fundusMask.astype(float))
    return ldrift,cdrift

##image = cv2.imread('C:\Users\AK PUJITHA\Desktop\iiit h\semester 6\honors project 2\e1.png',1)
##bgImg,fundusMask = LumConDrift(bgImg,fundusMask)

