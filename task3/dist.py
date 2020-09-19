import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


block = lambda x, y, size=3: (slice(y, y+size), slice(x, x+size))

dist_start = 0
PIX_PER_MIN = 10
BLOCKSIZE = 5


def contrast(front_pixs, back_pixs):
    front = np.mean(np.array(front_pixs))
    back  = np.mean(np.array(back_pixs))
    k = abs(front - back) / max(front, back)
    return k


def l_side(img, f):
    l_side = [img[block(139, y, size=BLOCKSIZE)] for y in range(0, img.shape[0], BLOCKSIZE)]
    l_side_sur_l = np.array([img[block(132, y, size=BLOCKSIZE)] for y in range(0, img.shape[0], BLOCKSIZE)], dtype=np.object)
    l_side_sur_r = np.array([img[block(148, y, size=BLOCKSIZE)] for y in range(0, img.shape[0], BLOCKSIZE)], dtype=np.object)
    l_side_sur = (l_side_sur_l + l_side_sur_r) / 2
    k = np.array([contrast(front, back) for front, back in zip(l_side, l_side_sur)])
    k = k[::-1]
    first_i = np.where(k < 0.1)[0][0]
    vertical_dist = first_i * BLOCKSIZE / PIX_PER_MIN
    plt.figure()
    plt.plot(k)
    plt.title('frame %s' % f)
    length = list(range(0, len(k), int(10 * PIX_PER_MIN / BLOCKSIZE)))
    labels = ['%sm' % (dist_start+10*l) for l in range(len(length))]
    plt.xticks(length, labels)
    plt.savefig('./fig/task3-k/frame%s-lside.png' % f)
    plt.close()
    return vertical_dist


def r_side(img, f):
    r_side = [img[block(408, y, size=BLOCKSIZE)] for y in range(0, img.shape[0], BLOCKSIZE)]
    r_side_sur_l = np.array([img[block(400, y, size=BLOCKSIZE)] for y in range(0, img.shape[0], BLOCKSIZE)], dtype=np.object)
    k = np.array([contrast(front, back) for front, back in zip(r_side, r_side_sur_l)])
    k = k[::-1]
    first_i = np.where(k < 0.1)[0][0]
    vertical_dist = first_i * BLOCKSIZE / PIX_PER_MIN
    plt.figure()
    plt.plot(k)
    plt.title('frame %s' % f)
    length = list(range(0, len(k), int(10 * PIX_PER_MIN / BLOCKSIZE)))
    labels = ['%sm' % (dist_start+10*l) for l in range(len(length))]
    plt.xticks(length, labels)
    plt.savefig('./fig/task3-k/frame%s-rside.png' % f)
    plt.close()
    return vertical_dist


if __name__ == '__main__':
    results = dict(lsideverticaldist=list(), rsideverticaldist=list())
    for f in range(1, 101):
        file = './fig/task3-trans/original_frame%s-trans.bmp' % f
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        l_vertical_dist = l_side(img, f)
        r_vertical_dist = r_side(img, f)
        results['lsideverticaldist'].append(l_vertical_dist)
        results['rsideverticaldist'].append(r_vertical_dist)
        
    results = pd.DataFrame(results)
    results.to_csv('./fig/task3-k/results.csv', index_label='frame')
