#!/usr/bin/env python
# -*- cooing:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月09日 星期六 15时13分27秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np

from config import cfg 

def camera_to_lidar(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), p)
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_T_VELO_2_CAM)), p)
    p = p[0:3]
    return tuple(p)

def lidar_to_camear(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.array(cfg.MATRIX_T_VELO_2_CAM), p)
    p = np.matmul(np.array(cfg.MATRIX_R_RECT_0), p)
    p = p[0:3]
    return tuple(p)

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

def camera_to_lidar_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(x, y, z), h, w, l, -ry-np.pi/2
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret)

def lidar_to_camera_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(x, y, z), h, w, l, -rz-np.pi/2
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret)


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


if __name__ == '__main__':
    pass	
