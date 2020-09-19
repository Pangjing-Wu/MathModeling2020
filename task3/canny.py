import cv2

for f in range(1, 101):
    file = './data/highway-snapshot/original_frame%s.bmp' % f
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img,(5,5),0)
    threshold = (10, 40)
    cv2.imwrite('./fig/task3-canny/original_frame%s-canny.bmp' % f, cv2.Canny(img, *threshold))