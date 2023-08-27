"""
@Project: PICR_Net
@File: train.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""

import time
import os
import logging
from datetime import datetime
from mindspore import dataset,context,nn
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
# from torch import einsum
# from einops import rearrange,  repeat
import mindspore.ops as ops
import numpy as np
from math import exp
from tensorboardX import SummaryWriter

from setting.dataLoader import get_loader
from setting.utils import clip_gradient, adjust_lr
from setting.utils import create_folder, random_seed_setting
from setting.options_ms import opt
from setting.loss import IOU, SSIM
from model.build_model_ms import PICR_Net

random_seed_setting()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
# Logs
save_path = create_folder(opt.save_path)
logging.basicConfig(filename=save_path + 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info(f'Config--epoch:{opt.epoch}; lr:{opt.lr}; batch_size:{opt.batchsize}; image_size:{opt.trainsize}')
writer = SummaryWriter(save_path + 'summary')

# load data
train_loader, train_num = get_loader(opt.rgb_root, opt.depth_root, opt.gt_root, opt.batchsize, opt.trainsize)
val_loader, val_num = get_loader(opt.val_rgb_root, opt.val_depth_root, opt.val_gt_root, 1, opt.trainsize)
print(f'Loading data, including {train_num} training images and {val_num} validation images.')
logging.info(f'Loading data, including {train_num} training images and {val_num} validation images.')
# model
model = PICR_Net()


n_parameters = sum(p.numel() for p in model.trainable_params())
logging.info(f"number of params: {n_parameters}")
# check model size
# if not os.path.exists('module_size'):
#     os.makedirs('module_size')
# for name, module in model.named_children():
#     mindspore.save_checkpoint(module, 'module_size/' + '%s' % name + '.pth')
# optimizer
optimizer = nn.optim.Adam(model.trainable_params(), opt.lr)

# Loss function
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = SSIM(window_size=11, size_average=True)
iou_loss = IOU(size_average=True)


def loss_bce_ssim_iou(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out  + iou_out + ssim_out

    return loss

def loss_bce_iou(pred, target):
    bce_out = bce_loss(pred, target)
    # ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out  + iou_out

    return loss

# # Restore training from checkpoints
# if opt.load is not None:
#     checkpoint = torch.load(opt.load)
#     opt.epoch = checkpoint['epoch']
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print(f'Load model from [{opt.load}]')


# train function
class ComputeLoss(nn.Cell):
    def __init__(self, network):
        super(ComputeLoss, self).__init__(auto_prefix=False)
        self.network = network
    def construct(self,images, depths, gts):
        s, sides = self.network(images, depths)
        # for side in sides:
        #     print(side.shape)
        # s = torch.sigmoid(s)
        gts_s1 = F.interpolate(gts, (7, 7), mode='bilinear', align_corners=True)
        gts_s2 = F.interpolate(gts, (14, 14), mode='bilinear', align_corners=True)
        gts_s3 = F.interpolate(gts, (28, 28), mode='bilinear', align_corners=True)
        gts_s4 = F.interpolate(gts, (56, 56), mode='bilinear', align_corners=True)
        loss_s1 = loss_bce_ssim_iou(sides[0].sigmoid(), gts_s1)
        loss_s2 = loss_bce_ssim_iou(sides[1].sigmoid(), gts_s2)
        loss_s3 = loss_bce_ssim_iou(sides[2].sigmoid(), gts_s3)
        loss_s4 = loss_bce_ssim_iou(sides[3].sigmoid(), gts_s4)

        loss_side = loss_s4 / 2 + loss_s3 / 4 + loss_s2 / 8 + loss_s1 / 16
        # loss_side = loss_s4  + loss_s3  + loss_s2  + loss_s1
        loss_main = loss_bce_ssim_iou(s.sigmoid(), gts)
        loss = loss_main + loss_side
        return loss

def train(train_loader, model, optimizer, epoch, save_path, iteration):
    model.set_train(True)
    loss_all = 0
    net = ComputeLoss(model)
    T_net = nn.TrainOneStepCell(net, optimizer)
    # print(iteration)

    # iteration = len(train_loader)
    try:
        for i, data in enumerate(train_loader, start=1):


            data["rgb"] = F.squeeze(data["rgb"], axis=(1))
            data["gt"] = F.squeeze(data["gt"], axis=(1))
            data["d"] = F.squeeze(data["d"], axis=(1))
            images, depths, gts = data['rgb'], data['d'], data['gt']
            loss_step = T_net(images, depths, gts)

            loss_all += loss_step.asnumpy()
            if i % (iteration // 4) == 0 or i == iteration:
                print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                      f'Step [{i:04d}/{iteration:04d}], loss_step: {loss_step.asnumpy():.4f}')
                logging.info(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                             f'Step [{i:04d}/{iteration:04d}], loss_step: {loss_step.asnumpy():.4f}')
        # loss_avg = loss_all / iteration
        # print(f'Epoch [{epoch:03d}/{opt.epoch:03d}]:Loss_train_avg={loss_avg:.4f}')
        # logging.info(f'Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_train_avg: {loss_avg:.4f}')
        # writer.add_scalar('Loss-train-avg', loss_avg, global_step=epoch)

        if  (epoch % 5 == 0 or epoch == opt.epoch):
            mindspore.save_checkpoint(model, save_path + 'PICR_Net_epoch_{}'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        mindspore.save_checkpoint(state, save_path + 'PICR_Net_epoch_{}_checkpoint.pth.tar'.format(epoch + 1))
        print('Save checkpoint successfully!')
        raise


# test function
best_loss = 100
best_epoch = 1


def test(val_loader, model, epoch, save_path):
    global best_loss, best_epoch
    model.set_train(False)
    loss_sum = 0
    for i, (image, depth, gt) in enumerate(val_loader, start=1):

        pre = model(image, depth)
        loss = loss_bce_ssim_iou(pre, gt)
        loss_sum += loss.detach()
    loss_epoch = loss_sum / val_num
    if loss_epoch < best_loss:
        best_loss, best_epoch = loss_epoch, epoch
        mindspore.save_checkpoint(model, save_path + 'PICR_Net_epoch_best.pth')
    print(f'Epoch [{epoch:03d}/{opt.epoch:03d}]:Loss_val={loss_epoch:.4f},'
          f' Best_loss={best_loss:.4f}, Best_epoch:{best_epoch:03d}')



if __name__ == '__main__':
    print("-------------------Config-------------------\n"
          f'epoch:\t\t{opt.epoch}\n'
          f'lr:\t\t{opt.lr}\n'
          f'batchsize:\t{opt.batchsize}\n'
          f'image_size:\t{opt.trainsize}\n'
          f'decay_epoch:\t{opt.decay_epoch}\n'
          f'decay_rate:\t{opt.decay_rate}\n'
          f'checkpoint:\t{opt.load}\n'
          "--------------------------------------------\n")
    print("Start train...")
    import warnings
    warnings.filterwarnings("ignore")
    time_begin = time.time()
    # warm_up_with_multistep_lr = lambda epoch: epoch / 5 if epoch <= 5 else \
    #     0.2 ** len([m for m in [40, 80, 100] if m <= epoch])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    for epoch in range(1, opt.epoch + 1):
        decay = opt.decay_rate ** (epoch // opt.decay_epoch)
        lr = decay * opt.lr

        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        ops.assign(optimizer.learning_rate, Tensor(lr, mindspore.float32))
        # cur_lr = scheduler.get_lr()
        writer.add_scalar('learning-rate', lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path,train_num)
        # test(val_loader, model, epoch, save_path)
        # scheduler.step()
        time_epoch = time.time()
        print(f"Time out:{time_epoch - time_begin:.2f}s\n")
        logging.info(f"Time out:{time_epoch - time_begin:.2f}s\n")
