import numpy as np
import cv2
import cv2.cv as cv
##from PIL import Image
##import scipy
##import scipy.io
##from matplotlib import pyplot as plt

def computebgimg(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    fundusMask = v > 20
    b,g,r = cv2.split(image)
    m,n = g.shape
    tsize = 50
    indx=0
    indy=0
    i = tsize
   
    tmean = np.zeros((int(m/tsize),int(n/tsize)),np.float)
    tstd = np.zeros((int(m/tsize),int(n/tsize)),np.float)
    cnt = 1
    while(i<m):
        j = tsize
        while(j<n):
            cnt = cnt +1 
            if (i+tsize>=m and j+tsize<n):
                block = g[i-tsize:m, j-tsize:j+tsize]
            else:
                if (i+tsize<m and j+tsize>=n):
                    block = g[i-tsize:i+tsize, j-tsize:n]
                else:
                    if (i+tsize>=m and j+tsize>=n):
                        block = g[i-tsize:m, j-tsize:n]
                    else : 
                        block = g[i-tsize:i+tsize, j-tsize:j+tsize]
                    
            mean,std = cv2.meanStdDev(block)
            tmean[indx,indy] = mean
            tstd[indx,indy] = std
            indy = indy+1
            j = j+tsize
        indy = 0
        indx = indx+1
        i = i+tsize
    tmean = cv2.resize(tmean,(n,m),interpolation = cv2.INTER_CUBIC)
    tstd = cv2.resize(tstd,(n,m),interpolation = cv2.INTER_CUBIC)
    bgImg = np.abs(cv2.divide(cv2.subtract(g.astype(float),tmean),tstd))
    bgImg = bgImg < 1
    return bgImg,fundusMask

##image = cv2.imread('C:\Users\AK PUJITHA\Desktop\iiit h\semester 6\honors project 2\e1.png',1)
##bgImg,fundusMask = computebgimg(image)

