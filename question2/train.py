import os
import tqdm
import torch
# import video_dataset
import img_dataset
from model import ResNet18Rnn
from torch.utils.tensorboard import SummaryWriter


def get_dataset(mode='train', bs=8):
    dataset = img_dataset.VidDataset(img_pt="./dataset",
                                     csv_pt="./20200313.csv",
                                     mode=mode, img_size=(224, 224))
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=bs, shuffle=True, num_workers=4)

    return dataloader


def main():
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    train_dl = get_dataset(mode='train', bs=8)
    val_dl = get_dataset(mode='val', bs=1)

    # Model
    model = ResNet18Rnn(param_loader())
    model = model.to(device)
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir)

    for e in range(epochs):
        print(f"[ INFO ] No.{e} epoch:")
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

        v_pbar = tqdm.tqdm(val_dl)
        for i, (intervals, labels) in enumerate(v_pbar):
            intervals, labels = intervals.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(intervals)
                loss = criterion(outputs, labels)

                running_size += intervals.size(0)
                running_loss += loss
                avg_loss = running_loss / running_size

            if i % 100 == 0:
                print(outputs)
                print(labels)
            v_pbar.set_postfix(avg_loss=avg_loss.item())
            writer.add_scalar('Val/loss', loss, len(v_pbar)*e+i)
            writer.add_scalar('Val/avg_loss', avg_loss, len(v_pbar)*e+i)


def param_loader():
    args = {'num_classes': 2,
            'dr_rate': 0.5,
            'pretrained': True,
            'rnn_hidden_size': 256,
            'rnn_num_layers': 1}

    return args


if __name__ == "__main__":
    main()
