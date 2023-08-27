"""
@Project: PICR_Net
@File: PGNet_test.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""

import cv2
import numpy as np
import os
from datetime import datetime
from mindspore import dataset,context
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from tqdm import tqdm

from model.build_model_ms import PICR_Net as PGNet
from setting.dataLoader import test_dataset
from setting.options_ms import opt
from setting.utils import create_folder
import time
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# model
print('load model...')
model = PGNet()
mindspore.load_param_into_net(model, mindspore.load_checkpoint(opt.test_model))
model.set_train(False)


test_datasets = ['NJU2K', 'NLPR', 'DUT',  'LFSD', 'STERE1000']
for dataset in test_datasets:
    # load data
    image_root = opt.test_path + dataset + '/RGB/'
    depth_root = opt.test_path + dataset + '/depth/'
    gt_root = opt.test_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)
    save_path = create_folder(opt.smap_save + dataset + '/')

    cost_time = list()
    for i in tqdm(range(test_loader.size), desc=dataset):
        image, depth, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        name = name.split('/')[-1]
        image = mindspore.Tensor(image)
        depth = mindspore.Tensor(depth)
        start_time = time.perf_counter()
        prediction, sides = model(image, depth)
        cost_time.append(time.perf_counter() - start_time)
        pre = F.interpolate(prediction, size=gt.shape, mode='bilinear', align_corners=False)
        pre = pre.sigmoid().asnumpy().squeeze()
        pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
        cv2.imwrite(save_path + name, pre * 255)



    cost_time.pop(0)
    print('Mean running time is: ', np.mean(cost_time))
    print("FPS is: ", test_loader.size / np.sum(cost_time))
print("Test Done!")




