import cv2
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

'''

img = cv2.imread("/image/img.jpg")

print(img.shape) # (500, 667, 3)
image_size = (256,256)

if img.shape[0] > image_size[0] or img.shape[1] > image_size[0]:
    cover_im = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
else:
    cover_im = cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)



img = cover_im.reshape((256, 256, 3))
print(img.shape)


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转为灰度图
plt.imshow(img)
plt.show()
print(img.shape)

_, im1 = cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(im1.shape)
plt.imshow(im1)
plt.show()

'''

import torch
from config import cfg

# 这个是滤波器使用的模板矩阵
# kernel_3x3 = np.array([[-1, -1, -1],
#                        [-1, 8, -1],
#                        [-1, -1, -1]])
#
# kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
#                        [-1, 1, 2, 1, -1],
#                        [-1, 2, 4, 2, -1],
#                        [-1, 1, 2, 1, -1],
#                        [-1, -1, -1, -1, -1]])


# 显示原始图像
# img = Image.open("./image/img.jpg")
# GREY = img.convert('L') # 转灰度图
# img.show()
# GREY.show()

# 以灰度的方式加载图片
# grey_img = img.convert('L')
# grey_img.show()

from torch.nn.functional import conv2d


# resize - transfrom -hpf -
#
#
# cover_image = img.astype(np.float32)
# cover_image = torch.from_numpy(cover_image)
# cover_image = torch.transpose(torch.transpose(cover_image, 2, 0), 1, 2)
# cover_image = cover_image / 255.0
#
# img_x = cover_image.view(-1, 1, cover_image.shape[2], cover_image.shape[3])
# print(img_x)

'''
# 高通滤波器
def hpf(x):
    KV = torch.tensor([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]]) / 12.
    KV = KV.view(1, 1, 5, 5).to(device='cuda', dtype=torch.float)
    # print(KV)
    # print(KV.shape)
    KV = torch.autograd.Variable(KV, requires_grad=False)

    # with torch.no_grad():
    img_x = img.view(-1, 1, img.shape[2], img.shape[3])
    out = conv2d(img_x, KV, padding=2).view(64, 3, x.shape[2], x.shape[3]).to(device='cuda')
    return out


hpf_img = hpf(img)
print(hpf_img)
'''


import os
import re


def latest_checkpoint(checkpoints_dir):
	''' If you get ValueError: max() arg is an empty sequence. Means your checkpoints_dir is empty just delete it.'''
	if os.path.exists(checkpoints_dir):
		if len(os.listdir(checkpoints_dir)) > 0:
			all_chkpts = "".join(os.listdir(checkpoints_dir))
			latest = max(map(int, re.findall('\d+', all_chkpts)))
		else:
			latest = None
	else:
		latest = None
	return latest


latest = latest_checkpoint("D:/pycharm/StegGAN-master/StegGAN-master/checkpoints")



print(latest)