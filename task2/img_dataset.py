import os
import torch
import random
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as trans


def transform_func(mode, img_size):
    resize_size = (int(img_size[0]*1.143), int(img_size[1]*1.143))
    if mode == 'train':
        # TODO Add others helpful augmentation for training
        return trans.Compose([
                trans.CenterCrop((640, 640)),
                trans.Resize(resize_size),
                trans.CenterCrop(img_size),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:
        return trans.Compose([
                trans.CenterCrop((640, 640)),
                trans.Resize(resize_size),
                trans.CenterCrop(img_size),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


class VidDataset(torch.utils.data.Dataset):
    def __init__(self, img_pt, csv_pt, base,
                 mode='train', img_size=(224, 224)):
        super(VidDataset, self).__init__()
        self.img_pt = img_pt
        self.base = base

        self.dataframe = self.get_dataframe_match_video(csv_pt)
        self.mor_range = self.dataframe.MOR_1A.max()
        self.rvr_range = self.dataframe.RVR_1A.max()

        self.trans = transform_func(mode, img_size)
        train_list, val_list = self.split_dataset(len(self.dataframe), 0.8)
        # self.file_list = train_list if mode == 'train' else val_list
        self.file_list = list(range(1915))

    def __getitem__(self, index):
        if self.base == 'visnet':
            video_interval = self.get_mix_interval(self.file_list[index])
        else:
            video_interval = self.get_one_interval(self.file_list[index])
        mor = np.log10(self.dataframe.MOR_1A.iloc[index]+1)
        rvr = np.log10(self.dataframe.RVR_1A.iloc[index]+1)
        labels = torch.FloatTensor((mor, rvr))
        return video_interval, labels

    def __len__(self):
        return len(self.file_list)

    def get_one_interval(self, idx):
        interval = list()
        for n in range(15):
            img = Image.open(os.path.join(self.img_pt, f'{idx}_{n}.png'))
            img = self.trans(img)
            interval.append(img)
        return torch.stack(interval)

    def get_mix_interval(self, idx):
        interval = list()
        for n in range(15):
            trans_imgs = list()
            for pt in self.img_pt:
                img = Image.open(os.path.join(pt, f'{idx}_{n}.png'))
                img = self.trans(img)
                trans_imgs.append(img)
            interval.append(torch.stack(trans_imgs))
        return torch.stack(interval)

    def get_dataframe_match_video(self, pt, start=3840, end=-1):
        df = pd.read_csv(pt)
        # Totally 1915 rows
        df = df.iloc[start:(None if end == -1 else end+1)]
        df = df.reset_index(drop=True)
        return df

    def split_dataset(self, tot, ratio=0.8):
        whole = list(range(tot))
        train = random.sample(whole, int(0.8*tot))
        val = list(set(whole) - set(train))
        return train, val


def main():
    ds = VidDataset(vid_pt="./Fog20200313000026.mp4",
                    csv_pt="./20200313.csv", mode="train")
    for i, l in ds:
        print(i.shape)
        print(l)
        break


if __name__ == "__main__":
    main()
