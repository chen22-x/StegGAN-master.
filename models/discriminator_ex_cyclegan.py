import os

import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# cyclegan的discriminator

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # print(x.size())
        # Average pooling and flatten
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        final_logits = torch.mean(torch.sigmoid(x.view(1, -1)), 1)
        return final_logits

# if __name__ == '__main__':
    # path = 'D:/pycharm/StegGAN-master/StegGAN-master/data/cover'
    # # message = cv2.imread('D:/pycharm/StegGAN-master/StegGAN-master/data/cover/image_00001.jpg')
    #
    # message = cv2.imread(os.path.join(path, os.listdir(path)[0]))
    #
    # # resize为（256,256）
    # if message.shape[0] > 256 or message.shape[1] > 256:
    #     message = cv2.resize(message, 256, interpolation=cv2.INTER_AREA)
    # else:
    #     message = cv2.resize(message, 256, interpolation=cv2.INTER_CUBIC)
    # # (256,256,3)
    # message = message.reshape((256, 256, 3))
    #
    # message = message.astype(np.float32)
    # message = torch.from_numpy(message)
    # # print(message.size())
    # message = torch.transpose(torch.transpose(message, 2, 0), 1, 2)
    # message = message / 255.0
    #
    #
    # print(message.shape)
    #
    # mse_loss = nn.MSELoss()


    # 秘密信息送入extractor的discriminator
    # dx_real_logit = Discriminator(message)
    # # dx_real_logit = torch.tensor(dx_real_logit, dtype=float)
    # print(message.shape)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # # 填充真实标签1
    # dx_real_label = torch.full((2,), 1., device=device)
    #
    # # 计算真实标签1和秘密信息损失
    # dx_loss_real = mse_loss(dx_real_logit, dx_real_label)  # 换成了mse