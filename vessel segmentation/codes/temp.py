import numpy as np
import cv2
import math
from imageenhancement import imageenhancement

def params():
    params={'ht':1,'COSFIRE':{'rholist':[0,2,4,6,8],'eta':[2.6180],'t1':[0],'t2':[0.4],'sigma0':[0.5],'alpha':[0.1167],\
                          'mintupleweight':[0.5],'ouputfunction':'geomentricmean','blurringfuction':'max',\
                          'weightingsigma':[6.7946],'t3':[0]},'invariance':{'rotation':{'psilist':[0,0.261799387799149,\
                                                                                               0.523598775598299,0.785398163397448,1.04719755119660,\
                                                                                               1.30899693899575,1.57079632679490,1.83259571459405,\
                                                                                               2.09439510239320,2.35619449019235,2.61799387799149,\
                                                                                               2.87979326579064]},'scale':{'upsilonlist':1},'reflection':0}\
        ,'detection':{'mindistance':8},'inputfilter':{'name':'DoG','DoG':{'polaritylist':[1],'sigmalist':[2.4],'sigmaratio':[0.5],'halfwaverct':[0]}},'symmetric':1}

    return(params)

def normalize(img):
    max_val=np.amax(img)
    r,c=img.shape
    for i in xrange(r):
        for j in xrange(c):
            if (img[i][j]==0):
                img[i][j]=0;
            else:
                img[i][j]=img[i][j]/max_val


    return(img)

def DoGBankResponse(img,params):
    sz1,sz2=img.shape;
    sigma=params['inputfilter']['DoG']['sigmalist'][0]
    sigma_ratio=params['inputfilter']['DoG']['sigmaratio'][0]
    onff=0
    DoGBank=getDoG(img,sigma,1,sigma_ratio)       
    return(DoGBank)

def fspecial(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def getDoG(img,sigma,onoff,sigmaRatio):
    ##create diff of Gaussian kernal
    sz=round((sigma*3) * 2 + 1)
    g1 = fspecial((sz,sz),sigma)
    g2 = fspecial((sz,sz),sigma*sigmaRatio);

    if (onoff==1):
        G=g2-g1;  ### difference of guassians
    else:
        G=g1-g2;

    #print(np.amax(G))
    dst = cv2.filter2D(img*255,-1,G) ###convolution of image
    return(dst,sigma)

def configcosfire(img,params,x,y,sig):
    angle=np.linspace(1,360,360)*(math.pi/180);
    rho=[2,4,6,8]
    rho1=[0,2,4,6,8]
    max1=np.amax(img)
    
    symparams={'sigma':sig,'0':[0,0],'2':[],'4':[],'6':[],'8':[]}
    for r in rho:
        for theta in angle:
            cnt=theta*180/math.pi
            x1=x+(r*math.cos(theta))
            y1=y+(r*math.sin(theta))
            #print(x1,y1)
            x_1=math.modf(x1)
            y_1=math.modf(y1)

            #print(x1,y1)
        
            if((x_1[0]==float(0)) & (y_1[0]==float(0))):
                if(img[x_1[1],y_1[1]]==max1):
                    symparams[str(r)].append(theta-(math.pi/2))
                
                
    symparams['rho']=rho1     
           
    return(symparams)
    
    
def configcosfire1(img,params,x,y,sig):
    angle=np.linspace(1,360,360)*(math.pi/180);
    rho=[2,4,6,8]
    rho1=[0,2,4,6,8]
    max1=np.amax(img);
    asymparams={'sigma':sig,'0':[0,0],'2':[],'4':[],'6':[],'8':[]}
    for r in rho:
        for theta in angle:
            cnt=theta*180/math.pi
            x1=x+(r*math.cos(theta))
            y1=y+(r*math.sin(theta))

            
            x_1=math.modf(x1)
            y_1=math.modf(y1)
            
            #print(img[x_1,y1])

            #print(x1,y1)
        

            if((x_1[0]==float(0)) & (y_1[0]==float(0))):
                if((img[x_1[1],y_1[1]]==43)or img[x_1[1],y_1[1]]==45 or img[x_1[1],y_1[1]]==42):
                    asymparams[str(r)].append(theta-(math.pi/2))
                    
                    
                
                
            
    asymparams['rho']=rho1  
    return(asymparams)
    


        
image = cv2.imread("F:/3-2/honors-2/fundus_extension/4.jpg",1)

image = imageenhancement(image)
image=normalize(image)
image=255-image


syparams=params()
img1,sigma=DoGBankResponse(image,syparams)


x=100;
y=100;
line1=np.zeros((201,201), dtype=np.uint8);
line1[:,x]=255;
line2=normalize(line1);
template_r,sigma=DoGBankResponse(line2,syparams)
symm_filter=configcosfire(template_r,syparams,x,y,sigma,)


line2[100:201,100]=0
line3=line2
template_ra,sigma=DoGBankResponse(line3,syparams)
asymm_filter=configcosfire1(template_ra,syparams,x,y,1.8)

