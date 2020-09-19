import torch
import torchvision
from torch import nn

import visnet


class ConvRnn(nn.Module):
    def __init__(self, params_model):
        super(ConvRnn, self).__init__()
        base_model = params_model['base']
        num_classes = params_model["num_classes"]
        dr_rate = params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]

        if base_model == 'resnet18':
            if params_model['trans'] == 'mix':
                raise KeyError("Base model ResNet18 should not mixed datasets")
            resnet = torchvision.models.resnet18(pretrained=pretrained)
            # num_features = self.base.fc.in_features
            resnet.fc = Identity()
            self.base = resnet
        elif base_model == 'visnet':
            if params_model['trans'] != 'mix':
                raise KeyError("Base model VisNet must use mixed datasets")
            self.base = visnet.VisNet()

        self.rnn = nn.GRU(rnn_hidden_size, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, 64)
        self.dropout = nn.Dropout(dr_rate)
        self.fc2 = nn.Linear(64, num_classes)
        self.clipper = nn.Sigmoid()

    def forward(self, x):
        hid = self.init_hidden(x)
        # features = self.base_forward(x)
        features = list()
        for seq in range(x.size(1)):
            feat = self.base(x[:, seq])
            features.append(feat)
        features = torch.stack(features)

        out, _ = self.rnn(features, hid)
        out = out[-1]

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.clipper(out)
        return out

    def base_forward(self, inp):
        features = list()
        # Run Base Net alone sequence -> concatenate features -> RNN
        if self.base.__class__.__name__ == 'VisNet':
            for seq in range(inp.size(1)):
                feat = self.base(inp[0][:, seq].squeeze(),
                                 inp[1][:, seq].squeeze(),
                                 inp[2][:, seq].squeeze())
                features.append(feat.unsqueeze(1))
        else:
            for seq in range(inp.size(1)):
                feat = self.base(inp[:, seq].squeeze())
                features.append(feat.unsqueeze(1))

        return torch.cat(features, dim=1)

    def init_hidden(self, inp):
        hidden = torch.zeros((1, inp.size(0), self.rnn.hidden_size),
                             device=inp.device)
        return hidden


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
