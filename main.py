import cv2
import numpy as np
from matplotlib import pyplot as plt
import corners
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
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
    hull = []
    for d in max_elem[1]:
        hull.append([d[0][0],d[0][1]])
        #hull.append(d[0])
    return hull,img

def handle_frame(img, plot_graph):
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (7, 7), 2.0, 2.0);
    
    canny = cv2.Canny(img, 10, 200)
    
    #cannypic = cv2.Canny( src, dst, 50, 200, 3 );
    
    # global thresholding
    ret1,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    
    hull,hull_img = find_hull(img,thresh)
    
    camera_matrix = np.array([[  2.85718343e+03,   0.00000000e+00,   1.63754626e+03],
                              [  0.00000000e+00,   2.86448210e+03,   1.21382910e+03],
                              [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    distortion_coefficients = np.array([  5.07668186e-02,   2.31285979e-01,  -3.40214009e-04,  -6.56692742e-04,  -1.08762549e+00], np.float64)

    print('hull{0}'.format(hull))
    hull_start_top = hull #corners.corners(hull) # hull[1:] + hull[:1]
    print('hull_start_top{0}'.format(hull_start_top))

    #[x[0],x[1],0]
    hull3d = []
    for e in hull_start_top:
        print('loop{0}, {1}'.format(e[0], e[1]))
        hull3d.append([e[0],e[1],0])
    print('hull3d{0}'.format(hull3d))
    hull3d = np.array(hull3d, np.float64)

    print('hull3d{0}'.format(hull3d))

    #world_coords = np.array([np.array([-1600.,-900.]), np.array([1600.,-900.]),
    #                         np.array([1600.,900.]),   np.array([-1600.,900.])])
    world_coords = np.array([[-1600.,-900.], [1600.,-900.],
                             [1600.,900.],  [-1600.,900.]], np.float64)
    
    pixel_coords = hull3d #np.array(hull,np.float64)
    
    print('distcoeff={0}'.format(distortion_coefficients))

    retval,rvec,tvec = cv2.solvePnP(pixel_coords, world_coords, camera_matrix, distortion_coefficients)
    print("retval{0},rvec{1},tvec{2}".format(retval,rvec,tvec))

    rotation_matrix,jac = cv2.Rodrigues(rvec.transpose())
    print('rotation_matrix{0}'.format(rotation_matrix))
    camera_rotation_vector,jac = cv2.Rodrigues(rotation_matrix.transpose())
    print('camera_rotation_vector{0}'.format(camera_rotation_vector))
    camera_translation_vector = -rotation_matrix.transpose()*np.matrix(tvec)
    
    print('camera_translation_vector={0}'.format(camera_translation_vector))

    if plot_graph:
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
    return (tvec,camera_translation_vector)

if False:
    img = cv2.imread('laptop.jpeg')
    handle_frame(img, True)
else:
    cap = cv2.VideoCapture("IMG_1832.MOV")

    cmap = cm.jet
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cnt = 0
    points = []
    while cap.isOpened():
        flag, frame = cap.read()
        tvec,ctv = handle_frame(frame, False)
        norm = np.linalg.norm(ctv)
        print('ctv len{0}'.format(norm))
        if norm>10000: #ctv[2] > 0:
            continue
        ax.scatter(*ctv, cmap=cmap)
        #points.append([ctv[0][0], ctv[1][0], ctv[2][0]])
        #points.append(np.asarray(1).reshape((1,-1)))
        point = np.array(ctv.transpose())[0].tolist()

        #if len(points)>0 and points
        points.append(point)
        cnt = cnt + 1
        if cnt > 400:
            break
    print(points)
    #ax.plot(points[,1], points[,2], points[,3])
    x,y,z = zip(*points)
    #ax.scatter(x,y,z)
    plt.show()
