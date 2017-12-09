#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : train.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月09日 星期六 11时32分45秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import glob
import argparse
import os
import time

from config import cfg
from kitti_loader import KittiLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all= '%s,%s,%s,%s' % (mv3d_net.top_view_rpn_name ,mv3d_net.imfeature_net_name,mv3d_net.fusion_net_name, mv3d_net.frontfeature_net_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default=all,  # FIXME
        help='use pre trained weights example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--clear-progress', action='store_true', default=False,
                        help='set continue train flag')

    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=1,
                        help='set continue train flag')

    parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                        help='set learning rate')


    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)

    max_iter = args.max_iter
    weights=[]
    if args.weights != '':
        weights = args.weights.split(',')

    targets=[]
    if args.targets != '':
        targets = args.targets.split(',')

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    if cfg.DATA_SETS_TYPE == 'didi2':

        train_key_list = ['suburu_pulling_up_to_it',
                          'nissan_brief',
                          'cmax_sitting_still',
                          'nissan_pulling_up_to_it',
                          'suburu_sitting_still',
                          'nissan_pulling_to_left',
                          'bmw_sitting_still',
                          'suburu_follows_capture',
                          'nissan_pulling_away',
                          'suburu_pulling_to_left',
                          'bmw_following_long',
                          'nissan_pulling_to_right',
                          'suburu_driving_towards_it',
                          'suburu_following_long',
                          'suburu_not_visible',
                          'suburu_leading_front_left',
                          'nissan_sitting_still',
                          'cmax_following_long',
                          'nissan_following_long',
                          'suburu_driving_away',
                          'suburu_leading_at_distance',
                          'nissan_driving_past_it',
                          'suburu_driving_past_it',
                          'suburu_driving_parallel',
                          ]

        train_key_full_path_list = [os.path.join(cfg.RAW_DATA_SETS_DIR, key) for key in train_key_list]
        train_value_list = [os.listdir(value)[0] for value in train_key_full_path_list]

        train_n_val_dataset = [k + '/' + v for k, v in zip(train_key_list, train_value_list)]

        data_splitter = TrainingValDataSplitter(train_n_val_dataset)


    elif cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test':
        training_dataset = {
            '1': ['6_f', '9_f', '10', '13', '20', '21_f', '15', '19'],
            '2': ['3_f', '6_f', '8_f'],
            '3': ['2_f', '4', '6', '8', '7', '11_f']}

        validation_dataset = {
            '1': ['15']}

    elif cfg.DATA_SETS_TYPE == 'kitti':
      # since 2011_09_26_0009 lacks No. 177,178,179,180 lidar data, deprecated
        training_dataset = {
            '2011_09_26': [
              '0070', '0015', '0052', '0035', '0061', '0002', '0018', '0013', '0032', '0056', '0017', '0011',
              '0001', '0005', '0014', '0020', ' 0059',
              '0019', '0084', '0028', '0051', '0060', '0064', '0027', '0086', '0022', '0023', '0046', '0029', '0087', '0091'
            ]
        }

        validation_dataset = {
            '2011_09_26': [
              '0036', '0057', '0079', '0048', '0039', '0093'
            ]
        }

# '2011_09_26': ['0001', '0002', '0005', '0011', '0013', '0015', '0017', '0018',  '0019', '0020', '0023',
#                        '0027', '0028', '0029', '0035', '0036', '0039', '0046', '0048', '0051', '0052', '0056', '0057', '0059',
#                            '0060', '0061', '0064', '0070', '0079', '0084', '0086', '0091']
#                 #0009 0014 0093 0022 0087 0032


    # for BatchLoading, only tags is essential and it must be type of list[]
    #with BatchLoading(data_splitter.training_bags, data_splitter.training_tags, require_shuffle=True) as training:
        #with BatchLoading(data_splitter.val_bags, data_splitter.val_tags,
    #                      queue_size=1, require_shuffle=True) as validation:
    # with BatchLoading3(tags=training_dataset, require_shuffle=True, use_precal_view=False, queue_size=30, use_multi_process_num=4) as training:
    #   with BatchLoading3(tags=validation_dataset, require_shuffle=False, use_precal_view=False, queue_size=30, use_multi_process_num=1) as validation:
    with KittiLoading(object_dir='/home/maxiaojian/data/kitti/object', queue_size=50, require_shuffle=True, 
         is_testset=False, use_precal_view=False, use_multi_process_num=4, split_file='/home/maxiaojian/workspace/eval-kitti/MV3D/ImageSets/train.txt') as training:
      with KittiLoading(object_dir='/home/maxiaojian/data/kitti/object', queue_size=50, require_shuffle=False, 
           is_testset=False, use_precal_view=False, use_multi_process_num=1, split_file='/home/maxiaojian/workspace/eval-kitti/MV3D/ImageSets/val.txt') as validation:
            train = mv3d.Trainer(train_set=training, validation_set=validation,
                                 pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                                 continue_train = not args.clear_progress, batch_size=args.batch_size, lr=args.lr)
            train(max_iter=max_iter)



if __name__ == '__main__':
    pass	
