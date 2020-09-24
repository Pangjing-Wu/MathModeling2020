import os
import tqdm
import torch
import img_dataset
import numpy as np
from convrnn import ConvRnn
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    return True


def get_dataset(img_pt, base, size, mode='train', bs=8):
    dataset = img_dataset.VidDataset(
            img_pt=img_pt, base=base, csv_pt="./20200313.csv",
            mode=mode, img_size=(size, size))
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=bs, shuffle=True, num_workers=16)

    return dataset, dataloader


def main(params):
    fix_seed(params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if params['trans'] == 'normal':
        ds_pt = "./dataset_normal"
    elif params['trans'] == 'fft':
        ds_pt = "./dataset_fft"
    elif params['trans'] == 'color_map':
        ds_pt = "./dataset_color_map"
    elif params['trans'] == 'mix':
        ds_pt = ["./dataset_normal", "./dataset_fft", "./dataset_color_map"]
    else:
        raise KeyError("Choose a dataset type!")
    print()
    print(f"[ INFO ] Using {params['trans']} dataset")

    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    train_ds, train_dl = get_dataset(
            ds_pt, base=params['base'], mode='train', bs=params['bs'],
            size=params['img_size'])
    val_ds, val_dl = get_dataset(
            ds_pt, base=params['base'], mode='val', bs=params['bs'],
            size=params['img_size'])

    # Model
    model = ConvRnn(params).to(device)
    model = torch.nn.DataParallel(model)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    writer = SummaryWriter(log_dir)

    for e in range(params['epochs']):
        print(f"[ INFO ] No.{e} epoch:")
        # Train
        model.train()
        running_size, running_loss = 0, 0
        t_pbar = tqdm.tqdm(train_dl)
        for i, (intervals, labels) in enumerate(t_pbar):
            intervals, labels = intervals.to(device), labels.to(device)
            outputs = model(intervals)
            loss_mor = criterion(outputs[:, 0], labels[:, 0])
            loss_rvr = criterion(outputs[:, 1], labels[:, 1])
            loss = loss_mor + loss_rvr

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
        model.eval()
        running_size, running_loss = 0, 0
        v_pbar = tqdm.tqdm(val_dl)
        for i, (intervals, labels) in enumerate(v_pbar):
            intervals, labels = intervals.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(intervals)
                loss_mor = criterion(outputs[:, 0], labels[:, 0])
                loss_rvr = criterion(outputs[:, 1], labels[:, 1])
                loss = loss_mor + loss_rvr

                running_size += intervals.size(0)
                running_loss += loss
                avg_loss = running_loss / running_size

            v_pbar.set_postfix(avg_loss=avg_loss.item())
            writer.add_scalar('Val/loss', loss, len(v_pbar)*e+i)
            writer.add_scalar('Val/avg_loss', avg_loss, len(v_pbar)*e+i)
        print("[ INFO ] MOR pred: ",
              [f'{n-1:.2f}' for n in
               torch.pow(10, outputs[:5, 0]).tolist()])
        print("         MOR gt  : ",
              [f'{n-1:.2f}' for n in
               torch.pow(10, labels[:5, 0]).tolist()])
        print("[ INFO ] RVR pred: ",
              [f'{n-1:.2f}' for n in
               torch.pow(10, outputs[:5, 1]).tolist()])
        print("         RVR gt  : ",
              [f'{n-1:.2f}' for n in
               torch.pow(10, labels[:5, 1]).tolist()])
        print()


def param_loader():
    parser = ArgumentParser()
    # Model arguments
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dr_rate", type=float, default=0.5)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--rnn_hidden_size", type=int, default=512)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--base", choices=['resnet18', 'visnet'])

    # Dataset type
    parser.add_argument(
            "--trans", choices=['normal', 'color_map', 'fft', 'mix'],
            help="Select which dataset to use")

    # Training hyper-parameter
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=666)

    args, _ = parser.parse_known_args()
    print(args)

    return vars(args)


if __name__ == "__main__":
    p = param_loader()
    main(p)
