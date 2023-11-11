'''
Developed by : AIT-ALI HASSNA
Verfied by AIT ALI HASSNA
'''

import cv2
import numpy as np
import imutils
import math
list_image=[]
list_y_distance=[]
list_x_distance=[]
list_distance=[]
def myfunction(frame,count,fps):
    img=frame
    img = cv2.GaussianBlur(img, (11, 11), 0)
    #img=cv2.resize(img, (500,200))
    img2=img.copy()
    img=cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    SJ=np.mean(img)
    h,s,v=cv2.split(img)
    b,g,r=cv2.split(img2)
    S=img
    height = S.shape[0]
    width = S.shape[1]
# Cut the image in half
    width_cutoff = width // 2
#print("width",width)
#print("width_ctoff",width_cutoff)
    left1 = S[:, :width_cutoff]
    right1 = S[:, width_cutoff:]
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    l1 = img[:, :width_cutoff]
    l2 = img[:, width_cutoff:]
# finish vertical devide image
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
#rotate image to 90 COUNTERCLOCKWISE
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    r1 = img[:, :width_cutoff]
    r2 = img[:, width_cutoff:]
#rotate image to 90 COUNTERCLOCKWISE
    r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
#rotate image to 90 COUNTERCLOCKWISE
    r2 = cv2.rotate(r2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    b1=l1[:, :, 1].mean()
    b2=l2[:, :, 1].mean()
    b3=r1[:, :, 1].mean()
    b4=r2[:, :, 1].mean()
    mylist=[b1,b2,b3,b4]
    SI=sum(mylist)/len(mylist)
    gray5=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    blurred5 = cv2.GaussianBlur(gray5, (11, 11), 0)
    SD=abs(SI-SJ)
    if SJ>SI:
        a1=SJ
    else:
       a1=SI
    Sa=SD/(a1)
    #print("SI",SI)
    #print("SJ",SJ)
    #print("SD",SD)
    #print("Sa",Sa)
    #print("maxlist",I)

    if  SD<10 and Sa<0.1 and Sa>0.08:

         val=True #Black
         if SJ>SI:

             a=(SI+SJ)/(SJ)
         elif SI>SJ:
             a=(SI+SJ)/(SI)
    
    else :

        val=False #white
        if SJ>SI:
            a=(SI)/(SJ)
        elif SI>SJ:
            a=(SJ)/(SI)
    #print("mya",a)
    agdb2=a*h-s
    blur = cv2.GaussianBlur(agdb2,(5,5),0)
    blur=blur*255
    blur.astype("uint8")
    ret,blur = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    blur=blur.astype("uint8")
    if count==1:
        list_image.append(blur)
        
    if count==2:
        im3=blur
        count=1
        result=cv2.absdiff(im3,list_image[0])
        list_image.clear()
        cv2.imshow("sub-res",result)
        

    contours=cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_TREE #,hierachie
    #image=cv2.drawContours(img2, contours, -1, (255,255,0),2) #img2
   # cnt = contours[0]
    #M = cv2.moments(blur)
    cnts = imutils.grab_contours(contours)
    #
    list_x_distance.clear()
    list_y_distance.clear()
    list_distance.clear()
    for c in cnts:
        M=cv2.moments(c)
        if int(M["m00"]!=0):
            cX=int(M["m10"]/M["m00"])
            cY=int(M["m01"]/M["m00"])
            list_x_distance.append(cX)
            list_y_distance.append(cY)
            cv2.drawContours(img2, [c], -1, (255,0,0),2)
            cv2.circle(img2,(cX,cY),7,(0,255,0),-1)
            cv2.putText(img2,"Cloud",(cX-20,cY-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            cv2.imshow("Image",img2)
            #print(c)

	# compute the center of the contour
	    #M = cv2.moments(c)
        
	    #cX = int(M["m10"] / M["m00"])
	    #cY = int(M["m01"] / M["m00"])
	# draw the contour and center of the shape on the image
	    #cv2.drawContours(img2, [c], -1, (0, 255, 0), 2)
	    #cv2.circle(img2, (cX, cY), 7, (255, 255, 255), -1)
	    #cv2.putText(img2, "center", (cX - 20, cY - 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	# show the image
	    #cv2.imshow("Image", img2)
      
	    #cv2.waitKey(0)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    #cv2.circle(img2, (cx, cy), 7, (255, 255, 255), -1)
	#cv2.putText(img2, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2))
    #cv2.putText(img2, "center", (cx + 20, cy + 20),
		#.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    h1,w1 =blur.shape
    
    #print("agd2",blur)
    gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    ret,thresh = cv2.threshold(blurred, 253, 255, cv2.THRESH_BINARY)
    gray1= cv2.GaussianBlur(gray, (21,21), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray1)
    image = frame.copy()
    cv2.circle(image, maxLoc, 21, (255, 0, 0), 2)
    print("maxloc",maxLoc)
    if val :
       blr=(255-blur)
       rest=blr-thresh
       nc=sum(rest==255)
       perc=100*(nc/(h1*w1))
    else :
    #white
       rest=blur-thresh
       nc=sum(rest==255)
       perc=100*(nc/(h1*w1))
    M1=cv2.moments(thresh)
    if M1['m00']!=0:
        cx = int(M1['m10']/M1['m00'])
        cy = int(M1['m01']/M1['m00'])
        cv2.circle(img2, (cx, cy), 7, (0, 0, 255), -1)
        cv2.putText(img2,"Sun",(cx-20,cy-20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for v in range(len(list_x_distance)):
            x=abs(list_x_distance[v]-cx)
            y=abs(list_y_distance[v]-cy)
            d=math.sqrt(pow(x,2)+pow(y,2))
            list_distance.append(d)
            print("list_x",list_x_distance)
            print("list_y",list_y_distance)
            print("distance",list_distance)
        if  list_distance:
            distance=min(list_distance)
            index=list_distance.index(distance)
            cv2.line(img2, pt1=(cx,cy), pt2=(list_x_distance[index],list_y_distance[index]), color=(0,0,255), thickness=5)
            t=int(distance/fps)
            cv2.putText(img2,'T: '+str(t),(cx+40,cy+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 40, 255), 2)



    cv2.imshow("originale", img2)
    cv2.imshow("image_with_sun", image)
    cv2.imshow("blur", blur)
    cv2.imshow("Sun",thresh)
    cv2.imshow("Rest",rest)
    #cv2.waitKey(0)
    
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('exemple_video.mp4')#'E:/Projet_fin_etude/opencv/mysky.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
  
# Loop untill the end of the video
count=0
while (cap.isOpened()):
  
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)
  
    # Display the resulting frame
    count+=1
    if count==3:
        count=1
    cv2.imshow('Frame', frame)
   # print("count",count)
    myfunction(frame,count,fps)
    
    # conversion of BGR to grayscale is necessary to apply this operation
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # adaptive thresholding to use different threshold 
    # values on different regions of the frame.
    #Thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          # cv2.THRESH_BINARY_INV, 11, 2)
  
    #cv2.imshow('Thresh', Thresh)
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()
