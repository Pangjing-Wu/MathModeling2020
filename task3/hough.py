import cv2
import numpy as np
import matplotlib.pyplot as plt  
 
f = 100


img= cv2.imread('./data/highway-snapshot/original_frame%s.bmp' % f)
img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 10, 40, apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,118) 
result = img.copy()
for line in lines:
	rho = line[0][0]
	theta= line[0][1]
	if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): 
		pt1 = (int(rho/np.cos(theta)),0)
		pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
		cv2.line( result, pt1, pt2, (255))             
	else:                                                
		pt1 = (0,int(rho/np.sin(theta)))              

		pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
		cv2.line(result, pt1, pt2, (255), 1)


fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7,3))
ax[0].imshow(edges, cmap='gray')
ax[0].set_title('canny')
ax[1].imshow(result)
ax[1].set_title('hough')
fig.tight_layout()
fig.savefig('./fig/task3-canny/frame%s-canny-hough.eps' % f)