import torch
import torch.nn as nn
from config import cfg


class Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        ans = self.block(inputs)
        # print('ans shape: ', ans.shape)
        return ans


class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
        )

    def forward(self, inputs):
        ans = torch.add(inputs, self.block(inputs))
        # print('ans shape: ', ans.shape)
        return ans
        # return inputs + ans


class Block3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, inputs):
        ans = torch.add(self.branch1(inputs), self.branch2(inputs))
        # print('ans shape: ', ans.shape)
        return ans


class Block4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block4, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

    def forward(self, inputs):
        temp = self.block(inputs)
        ans = torch.mean(temp, dim=(2, 3))
        # print('ans shape: ', ans.shape)
        return ans


class Steganalyzer(nn.Module):
    def __init__(self, data_format='NCHW', init_weights=True):
        super(Steganalyzer, self).__init__()
        self.inputs = None
        self.outputs = None
        self.data_format = data_format

        # 第一种结构类型
        self.layer1 = Block1(3, 64)
        self.layer2 = Block1(64, 16)

        # 第二种结构类型
        self.layer3 = Block2()
        self.layer4 = Block2()
        self.layer5 = Block2()
        self.layer6 = Block2()
        self.layer7 = Block2()

        # 第三种类型
        self.layer8 = Block3(16, 16)
        self.layer9 = Block3(16, 64)
        self.layer10 = Block3(64, 128)
        self.layer11 = Block3(128, 256)

        # 第四种类型
        self.layer12 = Block4(256, 512)

        # 最后一层，全连接层
        self.layer13 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)
        if init_weights:
            self._init_weights()

    def forward(self, inputs):

        # 第一种结构类型
        x = self.layer1(inputs)
        x = self.layer2(x)

        # 第二种结构类型
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # 第三种类型
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        # 第四种类型
        x = self.layer12(x)
        # 全连接层
        x = x.view(cfg.BATCH_SIZE, -1)
        x = self.layer13(x)
        x = self.softmax(x)
        # 最后一层全连接
        # self.outputs = self.layer13(x)
        # print('self.outputs.shape: ', self.outputs.shape)
        # final_logits = torch.mean(torch.sigmoid(x.view(cfg.BATCH_SIZE, -1)), 1)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.001)
