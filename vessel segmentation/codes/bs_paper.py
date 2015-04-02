import numpy as np
import cv2
import cv2.cv as cv
##from PIL import Image
##import scipy
##import scipy.io
##from matplotlib import pyplot as plt
from dilate import dilate

def gauss2D(shape,sigma):
    m = round(sigma*3)
    n = round(sigma*3)    
    y,x = np.ogrid[-m:m,-n:n]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return h

def blurshift(img_blur,sigmai,rhoi,phii):
    alpha = 0.1167
    sigma0 = 0.5
    rows,cols = img_blur.shape
    sigma1 = sigma0+alpha*(rhoi)
    filter_gaussian = gauss2D((rows,cols),sigma1)
    
    delta_xi = -1*rhoi*np.cos(phii)
    delta_yi = -1*rhoi*np.sin(phii)

    rows1,cols1 =img_blur.shape
    M = np.float32([[1,0,delta_xi],[0,1,delta_yi]])
    Dogresp_trans = cv2.warpAffine(img_blur,M,(cols1,rows1))
    blurshift_img = dilate(Dogresp_trans,filter_gaussian)
    return blurshift_img

##img_blur = cv2.imread('c.jpg')
##img_blur = img_blur[:,:,1]
#input (sigmai,rhoi,phii)
##image_bs = blurshift(img_blur,2.6,4,1.57)
##cv2.imshow('img_bs',image_bs)
##cv2.waitKey(0)
##cv2.imwrite('img_bs_c6.jpg',image_bs)

