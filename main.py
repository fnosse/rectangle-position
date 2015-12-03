import cv2
import numpy as np
from matplotlib import pyplot as plt

#def to_binary_img(img):

def find_hull(img,thresh):
    contours,hier = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    img = img.copy()
    
    hulls = []
    for cnt in contours:
        if True:#cv2.contourArea(cnt)>5000:  # remove small areas like noise etc
            hull = cv2.convexHull(cnt)    # find the convex hull of contour
            hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
            if len(hull)==4:
                #print('hull {0} area {1}'.format(hull, cv2.contourArea(cnt)))
                cv2.drawContours(img,[hull],0,(0,0,255),30)
                hulls.append((cnt,hull))
                #return hull,img
    max_elem = max(hulls, key=lambda h: cv2.contourArea(h[0]))
    print(max_elem[1])
    cv2.drawContours(img,[max_elem[1]],0,(0,255,0),30)
    return None,img

img = cv2.imread('../laptop.jpeg')

img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (7, 7), 2.0, 2.0);

canny = cv2.Canny(img, 10, 200)

#cannypic = cv2.Canny( src, dst, 50, 200, 3 );

# global thresholding
ret1,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

hull,hull_img = find_hull(img,thresh)

# plot all the images and their histograms
images = [img, 0, thresh,
          hull_img,
          canny
]

titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)','Hull', 'Canny']

i=0
#for i in range(0, len(images):
plt.subplot(3,3,1+i),plt.imshow(images[i],'gray')
plt.title(titles[i]), plt.xticks([]), plt.yticks([])
i = i+1

plt.subplot(3,3,1+i),plt.hist(images[0].ravel(),256)
plt.title(titles[i]), plt.xticks([]), plt.yticks([])
i = i+1

plt.subplot(3,3,1+i),plt.imshow(images[i],'gray')
plt.title(titles[i]), plt.xticks([]), plt.yticks([])
i = i+1

plt.subplot(3,3,1+i),plt.imshow(images[i],'gray')
plt.title(titles[i]), plt.xticks([]), plt.yticks([])
i = i+1

plt.subplot(3,3,1+i),plt.imshow(images[i],'gray')
plt.title(titles[i]), plt.xticks([]), plt.yticks([])
i = i+1

plt.show()

