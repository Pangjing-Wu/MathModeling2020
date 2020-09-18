import cv2
import numpy as np
import matplotlib.pyplot as plt
file = ['./data/vedio/vedio%s.png' % (i+1) for i in range(3)]



fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 4))

for i in range(3):
    img = cv2.imread(file[i], cv2.IMREAD_GRAYSCALE)
    ax[0,i].imshow(img,'gray')
    ax[0,i].set_title('original')
    colormap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    ax[1,i].imshow(colormap)
    ax[1,i].set_title('spectral filter')

plt.tight_layout()
plt.savefig('./fig/spectral-filter.eps')