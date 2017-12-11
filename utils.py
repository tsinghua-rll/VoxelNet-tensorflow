#!/usr/bin/env python
# -*- cooing:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月11日 星期一 11时12分06秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np 
import shaply.geometry
import shaply.affinity
from numba import jit

from config import cfg 

@jit 
def camera_to_lidar(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), p)
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_T_VELO_2_CAM)), p)
    p = p[0:3]
    return tuple(p)

@jit 
def lidar_to_camear(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.array(cfg.MATRIX_T_VELO_2_CAM), p)
    p = np.matmul(np.array(cfg.MATRIX_R_RECT_0), p)
    p = p[0:3]
    return tuple(p)

@jit 
def camera_to_lidar_point(points):
    # (N, 3) -> (N, 3)
    ret = []
    for p in points:
        x, y, z = p
        p = np.array([x, y, z, 1])
        p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), p)
        p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_T_VELO_2_CAM)), p)
        p = p[0:3]
        ret.append(p)
    return np.array(ret)

@jit 
def lidar_to_camera_point(points):
    # (N, 3) -> (N, 3)
    ret = []
    for p in points:
        x, y, z = p
        p = np.array([x, y, z, 1])
        p = np.matmul(np.array(cfg.MATRIX_T_VELO_2_CAM), p)
        p = np.matmul(np.array(cfg.MATRIX_R_RECT_0), p)
        p = p[0:3]
        ret.append(p)
    return np.array(ret)

@jit 
def camera_to_lidar_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(x, y, z), h, w, l, -ry-np.pi/2
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret)

@jit 
def lidar_to_camera_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(x, y, z), h, w, l, -rz-np.pi/2
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret)

@jit 
def center_to_corner_box(boxes_center):
    # (N, 7) -> (N, 8, 3)
    ret = []
    for box in boxes_center:
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]
        
        h, w, l = size[0],size[1],size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([ \
            [np.cos(yaw), -np.sin(yaw), 0.0], \
            [np.sin(yaw), np.cos(yaw), 0.0], \
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret.append(box3d)

    return np.array(ret)

@jit 
def corner_to_center_box(boxes_corner):
    # (N, 8, 3) -> (N, 7)
    ret = []
    for roi in boxes_corner:
        if cfg.CORNER2CENTER_AVG:  # average version
            roi = np.array(roi)
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            ) / 4
            x, y, z = np.sum(roi, axis=0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            )
            x, y, z = np.sum(roi, axis=0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
        ret.append([x, y, z, h, w, l, ry]) 
    return ret

@jit 
def lidar_box3d_to_camera_box2d(boxes3d):
    # (N, 7) -> (N, 4) x,y,z,h,w,l,rz -> x1,y1,x2,y2
    num = len(boxes3d)
    boxes2d = np.zeros((num, 5), dtype=np.int32)

    # TODO: here maybe some problems, check Mt/Kt
    Mt = np.array(cfg.MATRIX_Mt)
    Kt = np.array(cfg.MATRIX_Kt)

    for n in range(num):
        box3d = boxes3d[n]
        Ps = np.hstack((box3d, np.ones((8, 1))))
        Qs = np.matmul(Ps, Mt)
        Qs = Qs[:, 0:3]
        qs = np.matmul(Qs, Kt)
        zs = qs[:, 2].reshape(8, 1)
        qs = (qs / zs)

        minx = int(np.min(qs[:, 0]))
        maxx = int(np.max(qs[:, 0]))
        miny = int(np.min(qs[:, 1]))
        maxy = int(np.max(qs[:, 1]))
        
        boxes2d[n, 1:5] = minx, miny, maxx, maxy
    
    return boxes2d


@jit 
def lidar_to_bird_view(lidar):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    pass


@jit 
def draw_lidar_box3d_on_image(img, boxse3d, scores, gt_boxes3d=[]):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 8) [x, y, z, h, w, l, r]
    #   scores 
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    return img 


@jit 
def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d=[]):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 8) [x, y, z, h, w, l, r]
    #   scores 
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    return birdview
   

@jit 
def label_to_gt_box3d(labels, cls='Car', coordinate='camera'):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar' 
    # Output:
    #   (N, N', 7)
    boxes3d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    else:
        acc_cls = ['Cyclist']

    for label in labels:
        boxes3d_a_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls:
                box3d = np.array([float(i) for i in ret[-7:]])
                boxes3d_a_label.append(box3d)
        if coordinate == 'lidar':
            boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label))
        boxes3d.append(np.array(boxes3d_a_label))
    return boxes3d 

@jit 
def cal_iou2d(box1, box2):
    # Input:
    #   box1/2: x, y, w, l, r
    # Output:
    #   iou
    x1, y1, w1, l1, r1 = box1
    x2, y2, w2, l2, r2 = box2 
    c1 = shapely.geometry.box(-w1/2.0, -l1/2.0, w1/2.0, l1/2.0)
    c2 = shapely.geometry.box(-w2/2.0, -l2/2.0, w2/2.0, l2/2.0)
    
    c1 = shapely.affinity.rotate(c1, r1, use_radians=True)
    c2 = shapely.affinity.rotate(c2, r2, use_radians=True)

    c1 = shapely.affinity.translate(c1, x1, y1)
    c2 = shapely.affinity.translate(c1, x2, y2)

    intersect = c1.intersection(c2)

    return intersect.area/(c1.area + c2.area - intersect.area)

@jit 
def cal_iou3d(box1, box2):
    # Input:
    #   box1/2: x, y, z, h, w, l, r
    # Output:
    #   iou 
    def cal_z_intersect(cz1, h1, cz2, h2):
        b1z1, b1z2 = cz1 - h1/2, cz1 + h1/2
        b2z1, b2z2 = cz2 - h2/2, cz2 + h2/2
        if b1z1 > b2z2 or b2z1 > b1z2:
            return 0
        elif b2z1 <= b1z1 <= b2z2:
            if b1z2 <= b2z2:
                return h1/h2     
            else:
                return (b2z2-b1z1)/(b1z2-b2z1)
        elif b1z1 < b2z1 < b1z2:
            if b2z2 <= b1z2:
                return h2/h1 
            else:
                return (b1z2-b2z1)/(b2z2-b1z1)

    x1, y1, z1, h1, w1, l1, r1 = box1
    x2, y2, z2, h2, w2, l2, r2 = box2 
    c1 = shapely.geometry.box(-w1/2.0, -l1/2.0, w1/2.0, l1/2.0)
    c2 = shapely.geometry.box(-w2/2.0, -l2/2.0, w2/2.0, l2/2.0)
    
    c1 = shapely.affinity.rotate(c1, r1, use_radians=True)
    c2 = shapely.affinity.rotate(c2, r2, use_radians=True)

    c1 = shapely.affinity.translate(c1, x1, y1)
    c2 = shapely.affinity.translate(c1, x2, y2)

    z_intersect = cal_z_intersect(z1, h1, z2, h2)

    intersect = c1.intersection(c2)

    return intersect.area*z_intersect/(c1.area*h1 + c2.area*h2 - intersect.area*z_intersect)

@jit
def cal_box2d_iou(boxes2d, gt_boxes2d):
    # Inputs:
    #   boxes2d: (N1, 5) x,y,w,l,r
    #   gt_boxed2d: (N2, 5) x,y,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    
    N1, N2 = len(boxes2d), len(gt_boxes2d)
    output = numpy.zeros((N1, N2))
    for idx in range(N1):
        for idy in range(N2):
            output[idx, idy] = cal_iou2d(boxes2d[idx], gt_boxes2d[idy])
    
    return output 

@jit
def cal_anchors(feature_map_shape):
    # Input:
    #   feature_map_shape: [w, l]
    # Output:
    #   anchors: (w*l*2, 7)
    pass 


@jit 
def cal_rpn_target(labels, feature_map_shape, anchors, cls='Car', coordinate='lidar'):
    # Input:
    #   label: (N, N')
    #   feature_map_shape: (w, l)
    #   anchors: (w*l*2, 7)
    # Output:
    #   pos_equal_one (N, w, l, 2)
    #   neg_equal_one (N, w, l, 2)
    #   targets (N, w, l, 14)
    # attention: cal IoU on birdview 
    batch_size = labels.shape[0]
    batch_gt_boxes3d = label_to_gt_box3d(labels, cls=cls, coordinate='lidar')
    pos_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    neg_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    targets = np.zeros((batch_size, *feature_map_shape, 14))

    for idx in range(batch_size):
        iou = cal_box2d_iou(anchors, batch_gt_boxes3d[idx])
        
        # find anchor with highest iou(iou should also > 0)
        id_hightest = np.argmax(iou.T, axis=1)

        # find anchor iou > cfg.XXX_POS_IOU
        id_pos = np.where(iou, )

        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where()
    

@jit 
def delta_to_boxes3d(deltas, feature_map_shape, anchors, coordinate='lidar'):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w*l*2, 7)
    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    pass

if __name__ == '__main__':
    pass	
