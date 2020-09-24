import os
import cv2
import tqdm
import shutil
import numpy as np
from argparse import ArgumentParser


def fft_transfrom(img, width=50):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    x, y = img.shape
    cx, cy = int(x/2), int(y/2)
    mask = np.ones((x, y, 2), np.uint8)
    mask[cx-width:cx+width, cy-width:cy+width] = 0
    ishift = np.fft.ifftshift(fshift*mask)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    return np.uint8(res)


def colormap_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    return img


def gen_one_interval(mode, cap, path, idx, fps, start_frame):
    for n in range(15):
        position = (idx * 15 + n) * fps + start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        if mode == 'fft':
            frame = fft_transfrom(frame, 50)
        elif mode == 'color_map':
            frame = colormap_transform(frame)
        else:
            pass
        cv2.imwrite(os.path.join(path, f'{idx}_{n}.png'), frame)

    return True


def main(args):
    start_frame = 290
    end_frame = 698361
    fps = 24.2892
    total_interval = 1915
    mode = args.trans

    dataset_path = "./dataset_"+mode
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.mkdir(dataset_path)

    cap = cv2.VideoCapture("./Fog20200313000026.mp4")

    for i in tqdm.tqdm(range(total_interval+1)):
        _ = gen_one_interval(mode, cap, dataset_path, i, fps, start_frame)
    cap.release()


def param_loader():
    parser = ArgumentParser()
    parser.add_argument(
            "--trans", type=str, choices=['normal', 'color_map', 'fft'],
            help="Select a type of transformation for generated dataset")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    p = param_loader()
    main(p)
