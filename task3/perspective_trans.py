import numpy as np
import cv2

dist = lambda a, b: np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - bl[1]) ** 2))

def perspective_transform(image, tl, tr, br, bl):
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    max_width = max(int(dist(br, bl)), int(dist(tr, tl)))
    max_height = max(int(dist(tr, br)), int(dist(tl, bl)))
    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
        ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    print(M)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

if __name__ == '__main__':
    for f in range(1,2):
        file = './data/highway-snapshot/original_frame%s.bmp' % f
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        tl, tr = [316, 499], [520, 476]
        bl, br = [335, 719], [748, 673]
        img = perspective_transform(img, tl, tr, br ,bl)
        print(img)
        cv2.imwrite('./fig/task3-trans/original_frame%s-trans.bmp' % f, img)