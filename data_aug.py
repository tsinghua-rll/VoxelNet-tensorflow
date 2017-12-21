#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : data_aug.py
# Purpose :
# Creation Date : 21-12-2017
# Last Modified : 2017年12月22日 星期五 00时59分25秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import cv2
import os
import multiprocessing as mp
import argparse
import glob

from utils import *

object_dir = './data/object'
output_path = os.path.join(object_dir, 'training_aug')

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--aug-amount', type=int, nargs='?', default=1000)
parser.add_argument('-n', '--num-workers', type=int, nargs='?', default=10)
args = parser.parse_args()


def worker(tag):
    rgb = cv2.resize(cv2.imread(os.path.join(object_dir, 'training',
                                             'image_2', tag + '.png')), (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    lidar = np.fromfile(os.path.join(object_dir, 'training',
                                     'velodyne', tag + '.bin'), dtype=np.float32).reshape(-1, 4)
    label = np.array([line for line in open(os.path.join(
        object_dir, 'training', 'label_2', tag + '.txt'), 'r').readlines()])  # (N')
    cls = np.array([line.split()[0] for line in label])  # (N')
    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls='')[
        0]  # (N', 7) x, y, z, h, w, l, r

    choice = np.random.randint(1, 4)
    if True:
        # if choice == 1:
        # TODO: all in lidar coordinate
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_corner_gt_box3d = center_to_corner_box3d(lidar_center_gt_box3d)
        t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
        t_x = np.random.normal()
        t_y = np.random.normal()
        t_z = np.random.normal()
        for idx in range(len(lidar_corner_gt_box3d)):
            # TODO: precisely gather the point
            box_corner = lidar_corner_gt_box3d[idx]
            minx = np.min(box_corner[:, 0])
            miny = np.min(box_corner[:, 1])
            minz = np.min(box_corner[:, 2])
            maxx = np.max(box_corner[:, 0])
            maxy = np.max(box_corner[:, 1])
            maxz = np.max(box_corner[:, 2])
            bound_x = np.logical_and(
                lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
            bound_y = np.logical_and(
                lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
            bound_z = np.logical_and(
                lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
            bound_box = np.logical_and(
                np.logical_and(bound_x, bound_y), bound_z)
            lidar[bound_box, 0:3] = point_transform(
                lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
        lidar_center_gt_box3d = box_transform(
            lidar_center_gt_box3d, t_x, t_y, t_z, t_rz)
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = '{}_{}_{}'.format(
            tag, choice, np.random.randint(1, args.aug_amount))
    elif choice == 2:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar = lidar * factor
        gt_box3d[:, 0:6] = gt_box3d[:, 0:6] * factor
        newtag = '{}_{}_{:.4f}'.format(tag, choice, factor)
    else:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        gt_lidar_box = camera_to_lidar_box(gt_box3d)
        gt_box3d = lidar_to_camera_box(
            box_transform(gt_lidar_box, 0, 0, 0, angle))
        newtag = '{}_{}_{:.4f}'.format(tag, choice, angle)

    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...])[
        0]  # (N')
    cv2.imwrite(os.path.join(output_path, 'image_2', newtag + '.png'), rgb)
    lidar.reshape(-1).tofile(os.path.join(output_path,
                                          'velodyne', newtag + '.bin'))
    voxel_dict = process_pointcloud(lidar)
    np.savez_compressed(os.path.join(
        output_path, 'voxel' if cfg.DETECT_OBJ == 'Car' else 'voxel_ped', newtag), **voxel_dict)
    with open(os.path.join(output_path, 'label_2', newtag + '.txt'), 'w+') as f:
        for line in label:
            f.write(line)
    print(newtag)


def main():
    fl = glob.glob(os.path.join(object_dir, 'training', 'velodyne', '*.bin'))
    candidate = [f.split('/')[-1].split('.')[0] for f in fl]
    tags = []
    for _ in range(aug_amount):
        tags.append(candidate[np.random.randint(0, len(candidate))])
    print('generate {} tags'.format(len(tags)))
    pool = mp.Pool(args.num_workers)
    pool.map(worker, tags)


if __name__ == '__main__':
    main()
