import os
import tqdm
import torch
import img_dataset
import numpy as np
from model import ResNet18Rnn
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    return True


def get_dataset(img_pt, mode='train', bs=8):
    dataset = img_dataset.VidDataset(img_pt=img_pt,
                                     csv_pt="./20200313.csv",
                                     mode=mode, img_size=(224, 224))
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=bs, shuffle=True, num_workers=4)

    return dataloader


def main(params):
    fix_seed(params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if params['trans'] == 'normal':
        ds_pt = "./dataset_normal"
    elif params['trans'] == 'fft':
        ds_pt = "./dataset_fft"
    elif params['trans'] == 'color_map':
        ds_pt = "./dataset_color_map"
    print(f"[ INFO ] Using {params['trans']} dataset")

    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    train_dl = get_dataset(ds_pt, mode='train', bs=8)
    val_dl = get_dataset(ds_pt, mode='val', bs=8)

    # Model
    model = ResNet18Rnn(params)
    model = model.to(device)
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir)

    for e in range(params['epochs']):
        print()
        print(f"[ INFO ] No.{e} epoch:")
        # Train
        running_size, running_loss = 0, 0
        t_pbar = tqdm.tqdm(train_dl)
        for i, (intervals, labels) in enumerate(t_pbar):
            intervals, labels = intervals.to(device), labels.to(device)
            outputs = model(intervals)
            loss = criterion(outputs, labels)

            # Update model
            model.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_size += intervals.size(0)
                running_loss += loss
                avg_loss = running_loss / running_size

            t_pbar.set_postfix(avg_loss=avg_loss.item())
            writer.add_scalar('Train/loss', loss, len(t_pbar)*e+i)
            writer.add_scalar('Train/avg_loss', avg_loss, len(t_pbar)*e+i)

        # Val
        running_size, running_loss = 0, 0
        v_pbar = tqdm.tqdm(val_dl)
        for i, (intervals, labels) in enumerate(v_pbar):
            intervals, labels = intervals.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(intervals)
                loss = criterion(outputs, labels)

                running_size += intervals.size(0)
                running_loss += loss
                avg_loss = running_loss / running_size

            v_pbar.set_postfix(avg_loss=avg_loss.item())
            writer.add_scalar('Val/loss', loss, len(v_pbar)*e+i)
            writer.add_scalar('Val/avg_loss', avg_loss, len(v_pbar)*e+i)
        print(outputs)
        print(labels)


def param_loader():
    parser = ArgumentParser()
    # Model arguments
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dr_rate", type=float, default=0.5)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--rnn_hidden_size", type=int, default=256)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)

    # Dataset type
    parser.add_argument("--trans", choices=['normal', 'color_map', 'fft'],
                        help="Select which dataset to use")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=666)

    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == "__main__":
    p = param_loader()
    main(p)
