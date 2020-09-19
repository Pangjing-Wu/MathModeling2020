import torch
import torch.nn as nn


class ConvBlock1(nn.Module):
    def __init__(self):
        super(ConvBlock1, self).__init__()
        conv_block = [nn.Conv2d(3, 64, kernel_size=1, stride=(1, 1)),
                      nn.ReLU(),
                      nn.Conv2d(64, 64, kernel_size=3, stride=(1, 1)),
                      nn.ReLU(),
                      nn.MaxPool2d(2, stride=(2, 2))]

        self.model = nn.Sequential(*conv_block)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ConvBlock2(nn.Module):
    def __init__(self):

        super(ConvBlock2, self).__init__()
        conv_block = [nn.Conv2d(64, 128, kernel_size=1, stride=(1, 1)),
                      nn.ReLU(),
                      nn.Conv2d(128, 128, kernel_size=3, stride=(1, 1)),
                      nn.ReLU(),
                      nn.MaxPool2d(2, stride=(2, 2))]

        self.model = nn.Sequential(*conv_block)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ConvBlock3(nn.Module):
    def __init__(self):
        super(ConvBlock3, self).__init__()
        conv_block = [nn.Conv2d(128, 256, kernel_size=1, stride=(1, 1)),
                      nn.ReLU(),
                      nn.Conv2d(256, 256, kernel_size=3, stride=(2, 2)),
                      nn.ReLU(),
                      nn.Conv2d(256, 256, kernel_size=1, stride=(1, 1)),
                      nn.ReLU(),
                      nn.MaxPool2d(2, stride=(2, 2)),
                      nn.AdaptiveAvgPool2d((1, 1))]
        self.model = nn.Sequential(*conv_block)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class VisNet(nn.Module):
    def __init__(self):
        super(VisNet, self).__init__()
        self.STREAM_1_block1 = ConvBlock1()
        self.STREAM_1_block2 = ConvBlock2()
        self.STREAM_1_block3 = ConvBlock3()

        self.STREAM_2_block1 = ConvBlock1()
        self.STREAM_2_block2 = ConvBlock2()
        self.STREAM_2_block3 = ConvBlock3()

        self.STREAM_3_block1 = ConvBlock1()
        self.STREAM_3_block2 = ConvBlock2()
        self.STREAM_3_block3 = ConvBlock3()

        # self.fc_1 = nn.Linear(256, 1024)
        # self.fc_2 = nn.Linear(256, 2048)
        # self.drop = nn.Dropout(0.4)
        # self.fc_3 = nn.Linear(1024+2048, 4096)
        # self.classfy = nn.Linear(4096, 7)

    def forward(self, inp):
        input1, input2, input3 = inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]
        # generate output image given the input data_A
        output1_1 = self.STREAM_1_block1(input1)
        output1_2 = self.STREAM_2_block1(input2)
        output1_3 = self.STREAM_3_block1(input3)

        sum = torch.add(output1_2, output1_3)
        sum = torch.add(sum, output1_1)

        output2_1 = self.STREAM_1_block2(sum)
        output2_2 = self.STREAM_2_block2(output1_2)
        output2_3 = self.STREAM_3_block2(output1_3)

        sum = torch.add(output2_2, output2_3)
        sum = torch.add(sum, output2_1)

        output3_1 = self.STREAM_1_block3(sum)
        output3_2 = self.STREAM_2_block3(output2_2)
        output3_3 = self.STREAM_3_block3(output2_3)
        sum = torch.add(output3_2, output3_3)

        output3_1 = torch.flatten(output3_1, 1)
        sum = torch.flatten(sum, 1)

        return torch.cat([output3_1, sum], dim=1)

        # self.output_4_1 = self.drop(self.fc_1(self.output3_1))
        # self.output_4_2 = self.drop(self.fc_2(self.sum))

        # self.concated = torch.cat((self.output_4_1, self.output_4_2), dim=1)
        # self.output5 = self.fc_3(self.concated)
        # self.output = self.classfy(self.output5)
        # return self.output
