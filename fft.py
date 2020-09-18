import cv2
import numpy as np
import matplotlib.pyplot as plt


file = ['./data/vedio/vedio%s.png' % (i+1) for i in range(3)]



fig, ax = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(7, 9))

for i in range(3):
    img = plt.imread(file[i])
    img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    ax[0,i].imshow(img,'gray')
    ax[0,i].set_title('original')

    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)
    shift2center = np.fft.fftshift(dft)
    log_fft2 = np.log(1 + np.abs(dft))
    ax[1,i].imshow(log_fft2[:,:,0], 'Blues')
    ax[1,i].set_title('log_fft2')

    log_shift2center = np.log(1 + np.abs(shift2center))
    ax[2,i].imshow(log_shift2center[:,:,0], 'Blues')
    ax[2,i].set_title('log shift spectrum')

    width = 30
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2) #中心位置
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-width:crow+width, ccol-width:ccol+width] = 0
    #在频域上显示滤波器位置
    mask_img = log_shift2center
    mask_img[crow-width:crow+width, ccol-width:ccol+width] = 0
    ax[3,i].imshow(mask_img[:,:,0], 'Blues')
    ax[3,i].set_title('filter shifted spectrum')


    f = fshift * mask
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
    ax[4,i].imshow(res, 'gray')
    ax[4,i].set_title('output')

plt.tight_layout()
plt.savefig('./fig/fft.eps')