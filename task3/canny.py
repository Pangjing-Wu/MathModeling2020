import cv2

i_fig = 75
file  = './data/highway-snapshot/original_frame%s.bmp' % i_fig
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
threshold = (10, 40)
cv2.imwrite('./fig/original_frame%s.bmp' % i_fig, cv2.Canny(img, *threshold))