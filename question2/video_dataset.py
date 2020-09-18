import cv2
import torch
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
    def __init__(self, vid_pt, csv_pt, mode='train', img_size=(224, 224)):
        super(VidDataset, self).__init__()
        self.vid_pt = vid_pt
        self.mode = mode
        self.img_size = img_size

        # NOTE index 3840 in dataframe: time 00:00:45
        #      index 5754(last) in dataframe: time 07:59:45
        # NOTE range: 00:00:45 ~ 07:59:45
        #      63-42=21 91-64=27 113-92=21 140-114=26 162-141=21 190-163=27
        #      time slice = -7 ~ +8, total 15f(or s)
        # Next frame is 00:00:38
        self.start_frame = 290
        # 698340 ~ 698361 next is the last frame of 07:59:52
        self.end_frame = 698361
        # (end_frame-start_frame) / (07:59:52 - 00:00:38)second
        self.fps = 24.2892

        self.dataframe = self.get_dataframe_match_video(csv_pt)
        self.mor_range = self.dataframe.MOR_1A.max()
        self.rvr_range = self.dataframe.RVR_1A.max()

    def __getitem__(self, index):
        video_interval = self.extract_frame_interval(index)
        mor = self.dataframe.MOR_1A.iloc[index] / self.mor_range
        rvr = self.dataframe.RVR_1A.iloc[index] / self.rvr_range
        labels = torch.FloatTensor((mor, rvr))
        return video_interval, labels

    def __len__(self):
        return len(self.dataframe)

    def extract_frame_interval(self, idx):
        cap = cv2.VideoCapture(self.vid_pt)
        interval_list = list()
        for i in range(15):
            current_frame = (idx * 15 + i) * self.fps + self.start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            trans_frame = transform_func(self.mode, self.img_size)(pil_frame)
            interval_list.append(trans_frame)
        cap.release()
        return torch.stack(interval_list)

    def get_dataframe_match_video(self, pt, start=3840, end=-1):
        df = pd.read_csv(pt)
        # Totally 1915 rows
        df = df.iloc[start:(None if end == -1 else end+1)]
        df = df.reset_index(drop=True)
        return df


def main():
    ds = VidDataset(vid_pt="./Fog20200313000026.mp4",
                    csv_pt="./20200313.csv", mode="train")
    for i, l in ds:
        print(i.shape)
        print(l)
        break


if __name__ == "__main__":
    main()
