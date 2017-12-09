#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : config.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月09日 星期六 11时28分41秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]


://github.com/rbgirshick/fast-rcnn/blob/90e75082f087596f28173546cba615d41f0d38fe/lib/fast_rcnn/config.py

"""MV3D config system.
"""

import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict
import math

__C = edict()
# Consumers can get config by:
#    import config as cfg
cfg = __C

# for gpu allocation
__C.GPU_AVAILABLE='1'
__C.GPU_USE_COUNT=1
__C.GPU_MEMORY_FRACTION=0.3

# selected object 
__C.DETECT_OBJ=['Car', 'Van']
__C.DETECT_CLASS_AMOUNT=1

# for data set type
__C.DATA_SETS_TYPE='kitti'

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../.'))
__C.DATA_SETS_DIR=osp.join(__C.ROOT_DIR, , 'data')
__C.CHECKPOINT_DIR=osp.join(__C.ROOT_DIR, 'experiments', 'checkpoint')
__C.LOG_DIR=osp.join(__C.ROOT_DIR, 'experiments', 'log')

# for data preprocess
## sensors
__C.VELODYNE_ANGULAR_RESOLUTION=0.08/180*math.pi 
__C.VELODYNE_VERTICAL_RESOLUTION=0.4/180*math.pi
__C.VELODYNE_HEIGHT=1.73
## front 
__C.FRONT_R_OFFSET=70
__C.FRONT_C_OFFSET=750
__C.FRONT_R_MAX=30
__C.FRONT_R_MIN=-70
__C.FRONT_C_MAX=750
__C.FRONT_C_MIN=-750

__C.FRONT_WIDTH=__C.FRONT_C_MAX-__C.FRONT_C_MIN
__C.FRONT_HEIGHT=__C.FRONT_R_MAX-__C.FRONT_R_MIN
__C.FRONT_CHANNEL=3
# rgb
if __C.DATA_SETS_TYPE == 'kitti':
    __C.IMAGE_WIDTH=1242
    __C.IMAGE_HEIGHT=375
    __C.IMAGE_CHANNEL=3
## top
if __C.DATA_SETS_TYPE == 'kitti':
    __C.TOP_Y_MIN = -30
    __C.TOP_Y_MAX = +30
    __C.TOP_X_MIN = 0
    __C.TOP_X_MAX = 80
    __C.TOP_Z_MIN = -4.2
    __C.TOP_Z_MAX = 0.8

    __C.TOP_X_DIVISION = 0.1
    __C.TOP_Y_DIVISION = 0.1
    __C.TOP_Z_DIVISION = 0.2

    __C.TOP_WIDTH=(__C.TOP_X_MAX-__C.TOP_X_MIN)//__C.TOP_X_DIVISION
    __C.TOP_HEIGHT=(__C.TOP_Y_MAX-__C.TOP_Y_MIN)//__C.TOP_Y_DIVISION
    __C.TOP_CHANNEL=(__C.TOP_Z_MAX-__C.TOP_Z_MIN)//__C.TOP_Z_DIVISION

# for 2d proposal to 3d proposal 
__C.PROPOSAL3D_Z_MIN=-2.3 # -2.52
__C.PROPOSAL3D_Z_MAX=1.5 # -1.02

# for RPN basenet choose
__C.USE_VGG_AS_RPN=0
__C.USE_RESNET_AS_RPN=0
__C.USE_RESNEXT_AS_RPN=0

# for camera and lidar coordination convert
if __C.DATA_SETS_TYPE == 'kitti':
    __C.MATRIX_Mt = ([[  2.34773698e-04,   1.04494074e-02,   9.99945389e-01,  0.00000000e+00],
                  [ -9.99944155e-01,   1.05653536e-02,   1.24365378e-04,  0.00000000e+00],
                  [ -1.05634778e-02,  -9.99889574e-01,   1.04513030e-02,  0.00000000e+00],
                  [  5.93721868e-02,  -7.51087914e-02,  -2.72132796e-01,  1.00000000e+00]])

    __C.MATRIX_Kt = ([[ 721.5377,    0.    ,    0.    ],
                  [   0.    ,  721.5377,    0.    ],
                  [ 609.5593,  172.854 ,    1.    ]])

    __C.MATRIX_T_VELO_2_CAM = ([
            [ 7.533745e-03 , -9.999714e-01 , -6.166020e-04, -4.069766e-03 ], 
            [ 1.480249e-02 , 7.280733e-04 , -9.998902e-01, -7.631618e-02 ],
            [ 9.998621e-01 , 7.523790e-03 , 1.480755e-02, -2.717806e-01 ],
            [ 0, 0, 0, 1]    
    ])
    __C.MATRIX_R_RECT_0 = ([
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]
    ])

# Faster-RCNN/SSD Hyper params
__C.
## for ssd 
__C.SSD_BOXES_WIDTH=[3.9//__C.TOP_X_DIVISION, 1.0//__C.TOP_X_DIVISION]
__C.SSD_BOXES_HEIGHT=[1.6//__C.TOP_X_DIVISION, 0.6//__C.TOP_Y_DIVISION] 
__C.SSD_BOXES_ORINT=[0, 30, 60, 90]
__C.SSD_BOXES_AMOUNT=len(__C.SSD_BOXES_WIDTH*__C.SSD_BOXES_ORINT)
__C.SSD_REGRESS_CORNER=0

## for nms type
__C.USE_GPU_NMS=0
__C.GPUID_FOR_NMS=0
## for rpn nms
__C.ANCHOR_WIDTH_MIN=0
__C.ANCHOR_HEIGHT_MIN=0
__C.RPN_NMS_PRE_TOPK=2000
__C.RPN_NMS_POST_TOPK=2000
__C.RPN_NMS_THRESH=0.7
## for rpn training target choose
## for rcnn nms
## for rcnn training target choose

if __name__ == '__main__':
    print('__C.ROOT_DIR = '+__C.ROOT_DIR)
    print('__C.DATA_SETS_DIR = '+__C.DATA_SETS_DIR)

