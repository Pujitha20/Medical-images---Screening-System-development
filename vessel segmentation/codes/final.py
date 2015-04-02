from temp import *
from bs_paper import *
from Bcosfire_response import *
from rotation_invarience import *
bsf=np.zeros((10,img1.shape[0],img1.shape[1]))
resp=np.zeros((12,img1.shape[0],img1.shape[1]))
bsf1=np.zeros((5,img1.shape[0],img1.shape[1]))
resp1=np.zeros((12,img1.shape[0],img1.shape[1]))


print symm_filter
for orien in range(2):
    cnt=0;
    for i in xrange(5):
        for j in xrange(2):
            
            bs = blurshift(img1,symm_filter['sigma'],symm_filter['rho'][i],symm_filter[str(2*i)][j]+math.pi*orien/2)
            bsf[cnt,:,:]=bs
            cnt  = cnt + 1
            print cnt

        ##        cv2.imwrite("F:/v1.jpg",bs)
        ##        cv2.imshow("v",bs*255)
        ##        cv2.waitKey(0)
        
    resp[orien,:,:] = Bcosfire_response(bsf,symm_filter,0);

    
    cnt1=0;
    for i1 in xrange(5):
        for j1 in xrange(1):
            
            bs1 = blurshift(img1,symm_filter['sigma'],symm_filter['rho'][i1],asymm_filter[str(2*i1)][j1]+math.pi*orien/2)
            bsf1[cnt1,:,:]=bs1
            cnt1  = cnt1 + 1
            print cnt1

        ##        cv2.imwrite("F:/v1.jpg",bs)
        ##        cv2.imshow("v",bs*255)
        ##        cv2.waitKey(0)
        
    resp1[orien,:,:] = Bcosfire_response(bsf1,asymm_filter,1);





final = invarience(resp)
final1=invarience(resp1)
p=(final*255)+(final1*255)
q=p.astype(np.uint8)

ret,thresh1 = cv2.threshold(q,10,255,cv2.THRESH_BINARY)
cv2.imshow("w",thresh1)


#cv2.imwrite("F:/img12345.jpg",final*255)
#cv2.imshow("w",final)
cv2.waitKey(0)


