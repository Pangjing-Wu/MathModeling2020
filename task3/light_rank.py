import cv2
import numpy as np
import matplotlib.pyplot as plt


f = 1
rank = 10
file = './fig/task3-trans/original_frame%s-trans.bmp' % f
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img,(5,5),0)
img = img // rank * rank
cv2.imwrite('./fig/task3-rank/frame%s-rank.bmp' % f, img)
threshold = (10, 40)
img = cv2.Canny(img, *threshold)
cv2.imwrite('./fig/task3-rank/frame%s-rank-canny.bmp' % f, img)