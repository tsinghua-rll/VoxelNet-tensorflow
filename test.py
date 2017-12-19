#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : train.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月18日 星期一 00时18分59秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import glob
import argparse
import os
import time

from model import RPN3D
from config import cfg
from kitti_loader import KittiLoader
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing')

    parser.add_argument('--model-path', type=str, nargs='?',
                        default='', help='using which model')

    parser.add_argument('--output-path', type=str, nargs='?',
                        default='./data/results/data', help='results output dir')

    args = parser.parse_args()

    dataset_dir = './data/object'

    with tf.Graph().as_default():
        with KittiLoader(object_dir=dataset_dir + 'testing', queue_size=100, require_shuffle=False, is_testset=True, batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, use_multi_process_num=8, multi_gpu_sum=cfg.GPU_USE_COUNT) as test_loader:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                        visible_device_list=cfg.GPU_AVAILABLE,
                                        allow_growth=True)
            config = tf.ConfigProto(
                gpu_options=gpu_options,
                device_count={
                    "GPU": cfg.GPU_USE_COUNT,
                    allow_soft_placement=True,
                }
            )
            with tf.Session(config=config) as sess:
                model = RPN3D(
                    cls=cfg.DETECT_OBJ,
                    batch_size=1,
                    tag=args.tag,
                    is_train=False,
                    avail_gpus=cfg.GPU_AVAILABLE.split(',')
                )
                while True:
                    data = test_loader.load()
                    if data is None:
                        print('test done.')
                        break
                    ret = model.predict_step(sess, data)
                    # ret: A, B
                    # A: (N) tag
                    # B: (N, N') (class, x, y, z, h, w, l, rz, score)
                    for tag, result in zip(*ret):
                        of_path = os.path.join(args.output_path, tag + '.txt')
                        with open(of_path, 'w+') as f:
                            for item in result:
                                result[1:8] = lidar_to_rgb_box(
                                    result[1:8][np.newaxis, :])[0]
                                box2d = lidar_box3d_to_camera_box(
                                    result[1:8][np.newaxis, :], cal_projection=False)[0]
                                f.write('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                                    result[0], 0, 0, 0, *box2d, *(result[1:])))
                        print('write out {}'.format(of_path))
