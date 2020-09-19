import cv2
import numpy as np


H = 720
W = 1280

road_l_side_l_a = 2.4062
road_l_side_l_b = -431.22

road_l_side_r_a = 0.8694
road_l_side_r_b = 24.128

road_l_line_l_a = 1.272
road_l_line_l_b = -91.272

road_l_line_r_a = 1.2464
road_l_line_r_b = -83.602

road_r_side_r_a = 0.4533
road_r_side_r_b = 147.52

road_r_line_l_a = 0.5308
road_r_line_l_b = 129.45

road_r_line_r_a = 0.5351
road_r_line_r_b = 121.84

list_a = [road_l_side_l_a, road_l_side_r_a, road_l_line_l_a, road_l_line_r_a, road_r_side_r_a, road_r_line_l_a, road_r_line_r_a]
list_b = [road_l_side_l_b, road_l_side_r_b, road_l_line_l_b, road_l_line_r_b, road_r_side_r_b, road_r_line_l_b, road_r_line_r_b]


def line_point(a, b):
    x1 = 0
    y1 = int(x1 * a + b)
    x2 = W
    y2 = int(x2 * a + b)
    if y1 < 0:
        y1 = 0
        x1 = int((y1 - b) / a)
    elif y1 > H:
        y1 = H
        x1 = int((y1 - b) / a)
    else: 
        pass
    if y2 < 0:
        y2 = 0
        x2 = int((y2 - b) / a)
    elif y2 > H:
        y2 = H
        x2 = int((y2 - b) / a)
    else: 
        pass
    return (x1, y1), (x2, y2)


file = './data/highway-snapshot/original_frame100.bmp'
img = cv2.imread(file)
# img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

for a, b in zip(list_a, list_b):
    img = cv2.line(img, *line_point(a, b), color=(0, 255, 0), lineType=4)
cv2.imwrite('./fig/original_frame100-drawroad.bmp', img)