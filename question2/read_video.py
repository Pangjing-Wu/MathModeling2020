import os
import cv2
import tqdm
import shutil


def gen_one_interval(cap, path, idx, fps, start_frame):
    for n in range(15):
        position = (idx * 15 + n) * fps + start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(path, f'{idx}_{n}.png'), frame)

    return True


def main():
    start_frame = 290
    end_frame = 698361
    fps = 24.2892
    total_interval = 1915

    dataset_path = "./dataset"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.mkdir(dataset_path)

    cap = cv2.VideoCapture("./Fog20200313000026.mp4")

    pbar = tqdm.tqdm(range(total_interval+1))
    for i in pbar:
        _ = gen_one_interval(cap, dataset_path, i, fps, start_frame)
    cap.release()


if __name__ == "__main__":
    main()
