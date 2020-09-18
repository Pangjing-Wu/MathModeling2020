import os
import torch
import random
import pandas as pd
from PIL import Image
import torchvision.transforms as trans


def transform_func(mode, img_size):
    resize_size = (int(img_size[0]*1.143), int(img_size[1]*1.143))
    if mode == 'train':
        # TODO Add others helpful augmentation for training
        return trans.Compose([
                trans.Resize(resize_size),
                trans.CenterCrop(img_size),
                trans.ToTensor(),
                trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return trans.Compose([
                trans.Resize(resize_size),
                trans.CenterCrop(img_size),
                trans.ToTensor(),
                trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class VidDataset(torch.utils.data.Dataset):
    def __init__(self, img_pt, csv_pt, mode='train', img_size=(224, 224)):
        super(VidDataset, self).__init__()
        self.img_pt = img_pt

        self.dataframe = self.get_dataframe_match_video(csv_pt)
        self.mor_range = self.dataframe.MOR_1A.max()
        self.rvr_range = self.dataframe.RVR_1A.max()

        self.trans = transform_func(mode, img_size)
        train_list, val_list = self.split_dataset(len(self.dataframe), 0.8)
        self.file_list = train_list if mode == 'train' else val_list

    def __getitem__(self, index):
        video_interval = self.get_one_interval(self.file_list[index])
        mor = self.dataframe.MOR_1A.iloc[index] / self.mor_range
        rvr = self.dataframe.RVR_1A.iloc[index] / self.rvr_range
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
