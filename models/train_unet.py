''' training gan with xunet as discriminator for generator and an extractor with discriminator. num_images=17496 to 
make it divisible by batch_size.'''
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
import time
from utils.utils import threshold, time_taken, latest_checkpoint
from utils.hpf import hpf
from utils.normalize import normalize_imagenet
from utils.vgg16 import VGG16_intermediate

# from models.embedder import Embedder
# from models.extractor import Extractor
from models.discriminator_em import Steganalyzer
from models.discriminator_ex import Discriminator_ex

from models.embedder_unet import Embedder
from models.extractor_unet import Extractor



from config import cfg
from dataset import Dataset_Load
import torchvision.utils as vutils
#
# from models.discriminator_em_sagan import Steganalyzer
# from models.discriminator_ex_sagan import Discriminator_ex
# from models.embedder_sagan import embedder
# from models.extractor_sagan import extractor

cfg.merge_from_file('./experiment.yaml')

logging.basicConfig(filename='training.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

print("Embedder learning rate: ", cfg.EM_LR)
print("Steganalyzer learning rate: ", cfg.ST_LR)
print("Extractor  learning rate: ", cfg.EX_LR)
print("Extractor's discriminator learning rate: ", cfg.DI_LR)
print("\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    batches = int(cfg.NUM_IMAGES / cfg.BATCH_SIZE)

    real_label = 1.
    fake_label = 0.

    # Embedder network
    netG = Embedder()
    # netG = embedder()
    mse_loss = nn.MSELoss()
    netG.to(device)
    optim_g = optim.Adam(netG.parameters(), lr=cfg.EM_LR)

    # Embedder's discriminator netowrk
    netD = Steganalyzer()
    # netD = Steganalyzer()

    bce_loss = nn.BCELoss()
    netD.to(device)
    optim_d = torch.optim.SGD(netD.parameters(), cfg.ST_LR, momentum=0.9)

    # VGG features for perceptual loss
    vgg = VGG16_intermediate()
    vgg.to(device)
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    # Extractor network
    netE = Extractor()
    # netE = extractor()
    e_mse_loss = nn.MSELoss()
    netE.to(device)
    optim_e = optim.Adam(netE.parameters(), lr=cfg.EX_LR)

    # Extractor's discriminator network
    netDX = Discriminator_ex()
    # netDX = Discriminator_ex()
    bcex_loss = nn.BCELoss()
    netDX.to(device)
    optim_dx = optim.Adam(netDX.parameters(), lr=cfg.DI_LR)

    # dataset and dataloader
    dataset = Dataset_Load(msg_path=cfg.SECRET_PATH, cover_path=cfg.COVER_PATH)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    latest = latest_checkpoint(cfg.CHECKPOINT_PATH) #300
    latest = None

    if latest is None:
        # 从头开始训练
        start_epoch = 1
        print("No checkpoints found!! \nRetraining...\n")
        logging.debug("No checkpoints found!!\nRetraining\n")
        if not os.path.exists(cfg.CHECKPOINT_PATH):
            os.makedirs(cfg.CHECKPOINT_PATH)
    else:
        # 加载与训练模型
        checkpoint_g = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netG_" + str(latest) + ".pt"))
        checkpoint_d = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netD_" + str(latest) + ".pt"))
        checkpoint_e = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netE_" + str(latest) + ".pt"))
        checkpoint_dx = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netDX_" + str(latest) + ".pt"))

        start_epoch = checkpoint_g['epoch'] + 1

        netG.load_state_dict(checkpoint_g['model_state_dict'])
        optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])

        netD.load_state_dict(checkpoint_d['model_state_dict'])
        optim_d.load_state_dict(checkpoint_d['optimizer_state_dict'])

        netE.load_state_dict(checkpoint_e['model_state_dict'])
        optim_e.load_state_dict(checkpoint_e['optimizer_state_dict'])

        netDX.load_state_dict(checkpoint_dx['model_state_dict'])
        optim_dx.load_state_dict(checkpoint_dx['optimizer_state_dict'])

        print("Restoring model from checkpoint " + str(start_epoch))
        logging.debug("Restoring model from checkpoint " + str(start_epoch))

    netG.train()
    netD.train()
    netE.train()
    netDX.train()

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):

        total_Emse_loss = 0.0
        total_G_loss = 0.0
        total_mse_loss = 0.0
        total_EP_loss = 0.0
        total_D_loss = 0.0
        total_E_loss = 0.0
        total_DX_loss = 0.0
        total_adv_loss = 0.0

        s1 = time.time()
        for i_batch, sample_batched in enumerate(dataloader):
            s = time.time()

            cover_batch = sample_batched['cover'].to(device)  # 未处理的载体图像 cover
            msg_batch = sample_batched['msg'].to(device)  # 秘密信息
            th_batch = sample_batched['th'].to(device)  # 处理后的载体图像 Tc
            gt_batch = sample_batched['gt'].to(device)  # 未处理的载体图像

            # max ED first  先计算extractor的discriminator的损失（max）
            # extractor's discriminator
            for p in netDX.parameters():
                p.requires_grad = True   # netDX梯度更新

            optim_dx.zero_grad()  #梯度置为0

            # extractor's generator
            for p in netE.parameters():
                p.requires_grad = False

            # embedder's generator
            for p in netG.parameters():
                p.requires_grad = False

            # 秘密信息送入extractor的discriminator
            dx_real_logit = netDX(msg_batch)
            # dx_real_logit = torch.tensor(dx_real_logit, dtype=float)

            # 填充真实标签1
            dx_real_label = torch.full((cfg.BATCH_SIZE,), real_label, device=device)

            # 计算真实标签1和秘密信息损失
            dx_loss_real = bce_loss(dx_real_logit, dx_real_label)
            dx_loss_real.backward(retain_graph=True)
            # cover 秘密信息 阈值覆盖的cover   送入Embedder里面的generator--->S    （通过Tc将M嵌入到C）
            stego_batch = netG(cover_batch, msg_batch, th_batch)
            # 将生成的 S 送入网络extractor --> 提取出来的M
            pred_msg = netE(stego_batch)

            # 将经过extractor的generator出来的，即提取出来的secret message 送入extractor的discriminator
            dx_fake_logit = netDX(pred_msg)
            # 假标签
            dx_fake_label = torch.full((cfg.BATCH_SIZE,), fake_label, device=device)

            # 计算假标签0和 提取出来的secret message 的损失
            dx_loss_fake = bce_loss(dx_fake_logit, dx_fake_label)
            dx_loss_fake.backward()

            optim_dx.step()

            # Discriminator of Extractor loss： M和提取出来的M
            batch_DX_loss = float(dx_loss_fake) + float(dx_loss_real)  #
            total_DX_loss += batch_DX_loss
            # max ED done

            # 开始计算extractor的generator损失
            for p in netE.parameters():
                p.requires_grad = True

            optim_e.zero_grad()
            # 将生成的S-->提取出的M
            ext_msg_pred = netE(stego_batch)
            # 计算提取出来的M 和 原始的M的损失  L2_loss  MSE
            batch_Emse_loss = e_mse_loss(ext_msg_pred, msg_batch)
            batch_Emse_loss.backward(retain_graph=True)

            # extractor的mse总损失
            total_Emse_loss += batch_Emse_loss.item()
            # print("======",netE.conv1.weight.grad[0][0])

            #
            for p in netDX.parameters():
                p.requires_grad = False

            # 将提取出来的M 送入 extractor的discriminator
            dx_real_logit = netDX(ext_msg_pred)
            # 真实标签
            dx_real_label = torch.full((cfg.BATCH_SIZE,), real_label, device=device)
            # WT_EXT_ADV: 0.1  BCE_LOSS：提取出来的M和真实标签的损失
            dx_loss_real = torch.mul(cfg.WT_EXT_ADV, bce_loss(dx_real_logit, dx_real_label))
            dx_loss_real.backward(retain_graph=True)
            # print("******",netE.conv1.weight.grad[0][0])

            feats_orig_msg = vgg(normalize_imagenet(msg_batch))  # M
            feats_pred_msg = vgg(normalize_imagenet(ext_msg_pred)) # 提取的M

            # VGG层的perceptual loss   WT_PERCEPT: 0.0004    MSE_LOSS
            ext_perceptual = torch.mul(cfg.WT_PERCEPT, mse_loss(feats_pred_msg[1], feats_orig_msg[1]))
            ext_perceptual.backward()


            batch_EP_loss = ext_perceptual.item()
            # perceptual 的总 loss
            total_EP_loss += batch_EP_loss
            # print("-------",netE.conv1.weight.grad[0][0])

            optim_e.step()

            # extractor的loss: VGG层的perceptual loss + 提取出来的M和real_label 1 的损失（BCE） + M和提取出来的M损失（L2_MSE）
            batch_E_loss = float(batch_Emse_loss) + float(dx_loss_real) + batch_EP_loss  # Extractor loss

            total_E_loss += batch_E_loss  # extractor loss = discriminator的loss和generator的loss
            # Extractor done

            # embedder的discriminator部分
            for p in netD.parameters():
                p.requires_grad = True

            optim_d.zero_grad()

            for p in netG.parameters():
                p.requires_grad = False

            '''
                cover_batch = sample_batched['cover'].to(device)  # 未处理的载体图像 cover
                msg_batch = sample_batched['msg'].to(device)  # 秘密信息
                th_batch = sample_batched['th'].to(device)  # 处理后的载体图像 Tc
                gt_batch = sample_batched['gt'].to(device)  # 未处理的载体图像
            '''
            # 高通滤波器处理 cover
            c_batch = hpf(cover_batch)
            # cover-->embedder的discriminator
            d_real_logit = netD(c_batch)
            # 真实标签 1
            d_real_label = torch.full((cfg.BATCH_SIZE,), real_label, device=device)

            d_real_logit = torch.max(d_real_logit, 1)[0].float()
            # 计算cover和1的loss  BCE
            d_loss_real = bce_loss(torch.sub(1, d_real_logit), d_real_label)  #计算阈值处理后的cover和真实标签的bce损失
            d_loss_real.backward(retain_graph=True)

            # 生成S
            fake_images = netG(cover_batch, msg_batch, th_batch)
            # 高通滤波器处理S
            fake_images = hpf(fake_images)
            # 将S --> embedder的discriminator
            d_fake_logit = netD(fake_images)
            #
            d_fake_logit = torch.max(d_fake_logit, 1)[0].float()
            d_fake_label = torch.full((cfg.BATCH_SIZE,), fake_label, device=device)
            # 计算S和假标签0的损失
            d_loss_fake = bce_loss(torch.sub(1, d_fake_logit), d_fake_label)  # 计算生成的S和假标签的bce损失

            d_loss_fake.backward()

            optim_d.step()
            # Discriminator loss    cover和标签1.的损失+stego(高通滤波处理)与标签0.的损失
            batch_D_loss = float(d_loss_fake) + float(d_loss_real)    #相当于计算的是XuNet的损失
            total_D_loss += batch_D_loss
            # embedder的discriminator部分结束


            # min G now   （embedder部分开始）
            for p in netG.parameters():
                p.requires_grad = True

            optim_g.zero_grad()

            # get stego images
            pred_batch = netG(cover_batch, msg_batch, th_batch)

            # mse loss ： S和cover的损失
            batch_mse_loss = torch.mul(cfg.WT_MSE, mse_loss(pred_batch, cover_batch))  # 生成的S和cover的mse损失
            batch_mse_loss.backward(retain_graph=True)

            batch_mse_loss = float(batch_mse_loss)
            total_mse_loss += batch_mse_loss

            # for adv loss
            for p in netD.parameters():
                p.requires_grad = False

            # 高通滤波器处理 S
            pred_batch_f = hpf(pred_batch)
            # S送入embedder的discriminator
            stego_logits = netD(pred_batch_f)
            #
            stego_logits = torch.max(stego_logits, 1)[0].float()
            # 真实标签 1
            stego_labels = torch.full((cfg.BATCH_SIZE,), real_label, device=device)
            # BCE_LOSS:真实标签1和S的二分类交叉熵损失
            batch_adv_loss = torch.mul(cfg.WT_ADV, bce_loss(torch.sub(1, stego_logits), stego_labels))
            batch_adv_loss.backward(retain_graph=True)


            batch_adv_loss = float(batch_adv_loss)
            total_adv_loss += batch_adv_loss

            # min E loss
            for p in netE.parameters():
                p.requires_grad = False

            # S
            stego = netG(cover_batch, msg_batch, th_batch)
            # 提取出来的M
            ext_msg_pred = netE(stego)
            # 提取出来的M'和M 的MSE_LOSS
            batch_E_loss_g = torch.mul(cfg.WT_EXT,
                                       e_mse_loss(ext_msg_pred, msg_batch))  # extractor loss going to generator
            batch_E_loss_g.backward()

            # optim G
            optim_g.step()
            # S和cover的MSE损失 + S和真实标签1的损失（经过高通滤波处理） + 提取出来的M'和M
            batch_G_loss = batch_mse_loss + batch_adv_loss + float(batch_E_loss_g)   #Embedder的损失
            total_G_loss += batch_G_loss
            # （embedder部分结束）
            nb = float(cfg.NUM_IMAGES / cfg.BATCH_SIZE)  # total no. of batches
            f = time.time()
            batch_tt = f - s

            print(
                '| Eopch: %d| %d/%d batch| G_mse: %.6f| G_adv: %.6f| G_loss: %.6f| D_loss: %.6f| E_mse: %.6f| E_adv: %.6f| E_loss: %.6f| EP_loss:%.6f| DX_loss: %.6f|'
                % (epoch, i_batch, batches, batch_mse_loss, batch_adv_loss, batch_G_loss, batch_D_loss, batch_Emse_loss,
                   dx_loss_real, batch_E_loss, batch_EP_loss, batch_DX_loss))

        f1 = time.time()
        epoch_tt = f1 - s1 # 训练一个batch

        '''
            total_G_loss:
            total_mse_loss:
            total_EP_loss:
            total_D_loss:
            total_E_loss:
            total_Emse_loss:
            total_DX_loss:
        '''

        # 生成stego
        # show_result(epoch,netG,netE,cover_batch, msg_batch, th_batch,device)
        # # 提取message
        # stego_batch = netG(cover_batch, msg_batch, th_batch)
        # # 将生成的 S 送入网络extractor --> 提取出来的M
        # pred_msg = netE(stego_batch)
        # show_result(epoch,netG,netE,cover_batch,msg_batch,th_batch,device)

        netG.eval()
        netE.eval()
        save_stego = netG(cover_batch,msg_batch,th_batch)
        save_m = netE(save_stego)
        # 存储cover
        vutils.save_image(cover_batch.data, '%s/cover_epoch_%03d.png' % ('D:/pycharm/StegGAN-master/StegGAN-master/results/cover', epoch),
                          normalize=True)
        # 嵌入的secret
        vutils.save_image(msg_batch.data, '%s/M_epoch_%03d.png' % ('D:/pycharm/StegGAN-master/StegGAN-master/results/M', epoch),
                          normalize=True)
        # 存储生成的S
        vutils.save_image(save_stego.data,
                          '%s/stego_image_epoch_%03d.png' % ('D:/pycharm/StegGAN-master/StegGAN-master/results/stego', epoch),
                          normalize=True)
        # 存储提取出来的的M
        vutils.save_image(save_m.data,
                          '%s/extract_M_epoch_%03d.png' % ('D:/pycharm/StegGAN-master/StegGAN-master/results/extrac_secret', epoch),
                          normalize=True)
        netG.train()
        netE.train()

        print(
            '\n| Epoch: %d over | Gen_loss: %.6f | G_MSE: %.6f |percept_loss: %.6f | Disc_loss: %.6f | E_loss: %.6f | E_mse:%.6f | DisEx_loss: %.6f | T_ep:%s |\n'
            % (epoch, total_G_loss / nb, total_mse_loss / nb, total_EP_loss / nb, total_D_loss / nb, total_E_loss / nb,
               total_Emse_loss / nb, total_DX_loss / nb, time_taken(epoch_tt)))
        logging.debug(
            '\n| Epoch: %d over | Gen_loss: %.6f | G_MSE: %.6f | perc_loss: %.6f | Disc_loss: %.6f | E_loss: %.6f |E_mse: %.6f | DisEx_loss: %.6f |T_ep:%s |\n'
            % (epoch, total_G_loss / nb, total_mse_loss / nb, total_EP_loss / nb, total_D_loss / nb, total_E_loss / nb,
               total_Emse_loss / nb, total_DX_loss / nb, time_taken(epoch_tt)))

        # embedder的generator模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optim_g.state_dict(),
            'mse_loss': total_mse_loss,
            'entropy_loss': total_adv_loss,
            'total_loss': total_G_loss
        }, os.path.join(cfg.CHECKPOINT_PATH, 'netG_' + str(epoch) + '.pt'))

        # embedder的discriminator
        torch.save({
            'epoch': epoch,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optim_d.state_dict(),
            'entropy_loss': total_D_loss
        }, os.path.join(cfg.CHECKPOINT_PATH, 'netD_' + str(epoch) + '.pt'))

        # Extractor的generator
        torch.save({
            'epoch': epoch,
            'model_state_dict': netE.state_dict(),
            'optimizer_state_dict': optim_e.state_dict(),
            'e_mse_loss': total_E_loss,
            'e_perceptual_loss': total_EP_loss
        }, os.path.join(cfg.CHECKPOINT_PATH, 'netE_' + str(epoch) + '.pt'))

        # extractor的discriminator模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': netDX.state_dict(),
            'optimizer_state_dict': optim_dx.state_dict(),
            'dx_entropy_loss': total_DX_loss
        }, os.path.join(cfg.CHECKPOINT_PATH, 'netDX_' + str(epoch) + '.pt'))
