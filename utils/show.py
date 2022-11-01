import torch
import matplotlib.pyplot as plt
import itertools

# show_result(epoch, netG, netE, cover_batch, msg_batch, th_batch, device)
# stego_batch = netG(cover_batch, msg_batch, th_batch)
# # 将生成的 S 送入网络extractor --> 提取出来的M
# pred_msg = netE(stego_batch)

def show_result(num_epoch, G_net, E_net,cover_batch, msg_batch, th_batch, cuda):
    with torch.no_grad():
        # randn_in = torch.randn((5 * 5, 100))
        # if cuda:
        #     randn_in.cuda()

        G_net.eval()
        E_net.eval()
        #生成stego
        stego = G_net(cover_batch,msg_batch,th_batch)
        # 提取的m
        pred_m = E_net(stego)
        # test_images = G_net(randn_in) # (25,100)
        G_net.train()
        E_net.train()

        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(stego[k].cpu().data.numpy().transpose(1, 2, 0) * 0.5 + 0.5)


        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig("results/train_result/epoch_" + str(num_epoch) + "_results.png")
        plt.close('all')  #避免内存泄漏