import torch
import torchvision
from torch import nn


class ResNet18Rnn(nn.Module):
    def __init__(self, params_model):
        super(ResNet18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate = params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]

        self.base = torchvision.models.resnet18(pretrained=pretrained)
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, 256)

        self.rnn = nn.GRU(256, rnn_hidden_size,
                          rnn_num_layers, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, 64)
        self.dropout = nn.Dropout(dr_rate)
        self.fc2 = nn.Linear(64, num_classes)
        self.clipper = nn.Sigmoid()

    def forward(self, x):
        hid = self.init_hidden(x)

        # Run ResNet alone sequence -> concatenate features -> RNN
        features = list()
        for seq in range(x.size(1)):
            feat = self.base(x[:, seq].squeeze())
            features.append(feat.unsqueeze(1))
        features = torch.cat(features, dim=1)
        out, _ = self.rnn(features, hid)
        out = out[:, -1]

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.clipper(out)
        return out

    def init_hidden(self, inp):
        hidden = torch.zeros((1, inp.size(0), self.rnn.hidden_size),
                             device=inp.device)
        return hidden
