#!/usr/bin/env python
# -*- cooing:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月11日 星期一 20时13分49秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np 
import shapely.geometry
import shapely.affinity
from numba import jit

from config import cfg 


@jit 
def lidar_to_bird_view(x, y):
    #using the cfg.INPUT_XXX 
    return (x - cfg.X_MIN) / cfg.VOXEL_X_SIZE, (y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE


@jit 
def camera_to_lidar(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), p)
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_T_VELO_2_CAM)), p)
    p = p[0:3]
    return tuple(p)

@jit 
def lidar_to_camera(x, y, z):
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
def center_to_corner_box2d(boxes_center):
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0,1,4,5,6]] = boxes_center 
    boxes3d_corner = center_to_corner_box3d(boxes3d_center) 
    
    return boxes3d_corner[:, 0:4, 0:2]


@jit
def corner_to_center_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 5)
    N = boxes_corner.shape[0]
    boxes3d_corner = np.zeros((N, 8, 3))
    boxes3d_corner[:, 0:4, 0:2] = boxes_corner 
    boxes3d_corner[:, 4:8, 0:2] = boxes_corner 
    boxes3d_center = corner_to_center_box3d(boxes3d_corner)
    
    return boxes3d_center[:, [0,1,4,5,6]]


@jit 
def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2 
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros(N, 4)
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)
    
    return standup_boxes2d 


@jit 
def center_to_corner_box3d(boxes_center):
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
def corner_to_center_box3d(boxes_corner):
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
def lidar_box3d_to_camera_box(boxes3d, cal_projection=False):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    boxes3d_corner = center_to_corner_box3d(boxes3d_center)
    # TODO: here maybe some problems, check Mt/Kt
    Mt = np.array(cfg.MATRIX_Mt)
    Kt = np.array(cfg.MATRIX_Kt)

    for n in range(num):
        box3d = boxes3d_corner[n]
        Ps = np.hstack((box3d, np.ones((8, 1))))
        Qs = np.matmul(Ps, Mt)
        Qs = Qs[:, 0:3]
        qs = np.matmul(Qs, Kt)
        zs = qs[:, 2].reshape(8, 1)
        qs = (qs / zs)
        
        projections[n] = qs[:, 0:2] 
        minx = int(np.min(qs[:, 0]))
        maxx = int(np.max(qs[:, 0]))
        miny = int(np.min(qs[:, 1]))
        maxy = int(np.max(qs[:, 1]))
        
        boxes2d[n, 1:5] = minx, miny, maxx, maxy
    
    return projections if cal_projection else boxes2d


@jit 
def lidar_to_bird_view_img(lidar):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MIN:
            x, y = (x - cfg.X_MIN) / cfg.VOXEL_X_SIZE, (y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE 
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    birdview = (birdview/divisor*255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview 


@jit 
def draw_lidar_box3d_on_image(img, boxse3d, scores, gt_boxes3d=[], 
        color=(255,255,0), gt_color=(255,0,255), thickness=1):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores 
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True)

    # draw projections 
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
            
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
            
            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    # draw gt projections 
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), gt_color, thickness, cv2.LINE_AA)
            
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), gt_color, thickness, cv2.LINE_AA)
            
            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

    return img 


@jit 
def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d=[], 
        color=(255,255,0), gt_color=(255,0,255), thickness=1):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 8) [x, y, z, h, w, l, r]
    #   scores 
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d) 
    cornet_gt_boxes3d = center_to_corner_box3d(gt_boxes3d)
    # draw gt 
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2])
        x1, y1 = lidar_to_bird_view(*box[1, 0:2])
        x2, y2 = lidar_to_bird_view(*box[2, 0:2])
        x3, y3 = lidar_to_bird_view(*box[3, 0:2])
        
        cv2.line(img, (x0, y0), (x1, y1), gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y2), gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x3, y3), gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (x3, y3), (x1, y1), gt_color, thickness, cv2.LINE_AA)

    # draw detections 
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2])
        x1, y1 = lidar_to_bird_view(*box[1, 0:2])
        x2, y2 = lidar_to_bird_view(*box[2, 0:2])
        x3, y3 = lidar_to_bird_view(*box[3, 0:2])
        
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x3, y3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x3, y3), (x1, y1), color, thickness, cv2.LINE_AA)

    return img 
   

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
def cal_box3d_iou(boxes3d, gt_boxes3d, cal_3d=False):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    
    N1, N2 = len(boxes3d), len(gt_boxes3d)
    output = numpy.zeros((N1, N2))
    for idx in range(N1):
        for idy in range(N2):
            if cal_3d:
                output[idx, idy] = cal_iou3d(boxes3d[idx], gt_boxes3d[idy])
            else:
                output[idx, idy] = cal_iou2d(boxes3d[idx][0,1,4,5,6], gt_boxes3d[idy][0,1,4,5,6])

    return output 


@jit
def cal_box2d_iou(boxes2d, gt_boxes2d):
    # Inputs:
    #   boxes2d: (N1, 5) x,y,w,l,r
    #   gt_boxes2d: (N2, 5) x,y,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    
    N1, N2 = len(boxes2d), len(gt_boxes2d)
    output = numpy.zeros((N1, N2))
    for idx in range(N1):
        for idy in range(N2):
            output[idx, idy] = cal_iou2d(boxes2d[idx], gt_boxes2d[idy])

    return output 


@jit
def cal_anchors():
    # Output:
    #   anchors: (w, l, 2, 7) x y z h w l r
    x = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.INPUT_WIDTH) 
    y = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.INPUT_HEIGHT) 
    cx, cy = np.meshgrid(x, y)
    # all is (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx)*cfg.ANCHOR_Z 
    w = np.ones_like(cx)*cfg.ANCHOR_W
    l = np.ones_like(cx)*cfg.ANCHOR_L
    h = np.ones_like(cx)*cfg.ANCHOR_H 
    r = np.ones_like(cx)
    r[..., 0] = 0  # 0
    r[..., 1] = 90/180*np.pi # 90
    
    # 7*(w,l,2) -> (w, l, 2, 7)
    anchors = np.stack([cx, cy, cz, w, l, h, r], axis=-1)

    return anchors 


@jit 
def cal_rpn_target(labels, feature_map_shape, anchors, cls='Car', coordinate='lidar'):
    # Input:
    #   label: (N, N')
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)
    # Output:
    #   pos_equal_one (N, w, l, 2)
    #   neg_equal_one (N, w, l, 2)
    #   targets (N, w, l, 14)
    # attention: cal IoU on birdview 
    batch_size = labels.shape[0]
    batch_gt_boxes3d = label_to_gt_box3d(labels, cls=cls, coordinate='lidar')
    # defined in eq(1) in 2.2 
    anchors_d = np.sqrt(anchors[:, 4]**2 + anchors[:, 5]**2)
    pos_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    neg_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    targets = np.zeros((batch_size, *feature_map_shape, 14))

    for batch_id in range(batch_size):
        iou = cal_box3d_iou(anchors.reshape(-1, 7), batch_gt_boxes3d[batch_id], cal_3d=False)
        
        # find anchor with highest iou(iou should also > 0)
        id_hightest = np.argmax(iou.T, axis=1)
        id_hightest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_hightest_gt, id_hightest] > 0
        id_hightest, id_hightest_gt = id_hightest[mask], id_hightest_gt[mask]

        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > cfg.RPN_POS_IOU)

        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg, _ = np.where(iou < cfg.RPN_NEG_IOU)
        
        id_pos = np.concatenate([id_pos, id_hightest])
        id_pos_gt = np.concatenate([id_pos_gt, id_hightest_gt])

        # TODO: uniquify the array in a more scientific way 
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()
       
        # cal the target and set the equal one 
        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1

        # ATTENTION: index_z should be np.array 
        targets[batch_id, index_x, index_y, np.array(index_z)*7] = (batch_gt_boxes3d[batch_id][id_pos_gt, 0] - anchors[id_pos][0]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z)*7+1] = (batch_gt_boxes3d[batch_id][id_pos_gt, 1] - anchors[id_pos][1]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z)*7+2] = (batch_gt_boxes3d[batch_id][id_pos_gt, 2] / anchors[id_pos][2]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z)*7+3] = np.log(batch_gt_boxes3d[batch_id][id_pos_gt, 3] / anchors[id_pos][3])
        targets[batch_id, index_x, index_y, np.array(index_z)*7+4] = np.log(batch_gt_boxes3d[batch_id][id_pos_gt, 4] / anchors[id_pos][4])
        targets[batch_id, index_x, index_y, np.array(index_z)*7+5] = np.log(batch_gt_boxes3d[batch_id][id_pos_gt, 5] / anchors[id_pos][5])
        targets[batch_id, index_x, index_y, np.array(index_z)*7+6] = (batch_gt_boxes3d[batch_id][id_pos_gt, 6] - anchors[id_pos][6])


        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, 2))
        neg_equal_one[batch_id, index_x, index_y, index_z] = 1
    
    return pos_equal_one, neg_equal_one, targets 


@jit 
def delta_to_boxes3d(deltas, anchors, coordinate='lidar'):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w*l*2, 7)
    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    deltas = deltas.reshape(deltas.shape[0], -1, 7)
    anchors_d = np.sqrt(anchors[:, 4]**2 + anchors[:, 5]**2)
    boxes3d = np.zeros_like(deltas)
    boxes3d[..., [0, 1, 2]] = deltas[..., [0, 1, 2]]*anchors_d + anchors[..., [0, 1, 2]]
    boxes3d[..., [3, 4, 5]] = np.exp(deltas[..., [3, 4, 5]]) * anchors[..., [3, 4, 5]]
    boxes3d[..., 6] = deltas[..., 6] + anchors[..., 6]
    
    return boxes3d 


if __name__ == '__main__':
    pass	
