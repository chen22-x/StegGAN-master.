import os
import re

import cv2
import torch
import numpy as np


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


def threshold(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转灰度图

	# 实现Otsu方式的阈值分割。 需要说明的是，在使用Otsu方法时，要把阈值设为0。
	# 此时的函数cv2.threshold（）会自动寻找最优阈值，并将该阈值返回(即_)
	# img是返回处理后的图像
	# 返回值_是Otsu方法计算得到并使用的最优阈值。 需要注意，如果采用普通的阈值分割，返回的阈值就是设定的阈值
	_,im1 = cv2.threshold(np.array(img),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# im1 = cv2.cvtColor(im1,cv2.COLOR_GRAY2RGB)
	return im1


def time_taken(tt):
	if tt>60:
		tt=round(tt/60.,2)
		return str(tt)+'mins.'
	elif tt>3600:
		tt=round(tt/3600.,2)
		return str(tt)+'hrs.'
	else: return str(round(tt,2))+'sec.'



