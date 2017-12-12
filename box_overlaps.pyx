# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

import numpy as np
import shapely.geometry
import shapely.affinity
cimport numpy as np
from cython.parallel import prange, parallel


DTYPE = np.float32
ctypedef float DTYPE_t


def cal_iou2d(
        np.ndarray[DTYPE_t, ndim=1] box1, 
        np.ndarray[DTYPE_t, ndim=1] box2):
    # Input:
    #   box1/2: x, y, w, l, r
    # Output:
    #   iou
    cdef DTYPE_t x1, x2, y1, y2, l1, l2, w1, w2, r1, r2
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


def cal_z_intersect(
        float cz1, 
        float h1, 
        float cz2, 
        float h2):
    
    cdef float b1z1, b1z2, b2z1, b2z2

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


def cal_iou3d(
        np.ndarray[DTYPE_t, ndim=1] box1, 
        np.ndarray[DTYPE_t, ndim=1] box2):
    # Input:
    #   box1/2: x, y, z, h, w, l, r
    # Output:
    #   iou 
    cdef DTYPE_t x1, x2, y1, y2, z1, z2, h1, h2, l1, l2, w1, w2, r1, r2

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


def cal_box3d_iou(
        np.ndarray[float, ndim=2] boxes3d, 
        np.ndarray[float, ndim=2] gt_boxes3d, 
        unsigned int cal_3d=0):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    cdef unsigned int N1 = len(boxes3d)
    cdef unsigned int N2 = len(gt_boxes3d)
    cdef np.ndarray[float, ndim=2] output = np.zeros((N1, N2), dtype=np.float32)

    for idx in range(N1):
        for idy in range(N2):
            if cal_3d:
                output[idx, idy] = float(cal_iou3d(boxes3d[idx], gt_boxes3d[idy]))
            else:
                output[idx, idy] = float(cal_iou2d(boxes3d[idx, [0,1,4,5,6]], gt_boxes3d[idy, [0,1,4,5,6]]))

    return output 


def cal_box2d_iou(
        np.ndarray[float, ndim=2] boxes2d, 
        np.ndarray[float, ndim=2] gt_boxes2d): 
    # Inputs:
    #   boxes2d: (N1, 5) x,y,w,l,r
    #   gt_boxes2d: (N2, 5) x,y,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    
    cdef unsigned int N1 = len(boxes2d)
    cdef unsigned int N2 = len(gt_boxes2d)
    cdef np.ndarray[float, ndim=2] output = np.zeros((N1, N2), dtype=np.float32)
    for idx in range(N1):
        for idy in range(N2):
            output[idx, idy] = cal_iou2d(boxes2d[idx], gt_boxes2d[idy])

    return output 

def func(
    float a):
    c1 = shapely.geometry.box(1/2.0, 2/2.0, 1/2.0, 2/2.0)
    c2 = shapely.geometry.box(1/2.0, 2/2.0, 1/2.0, 2/2.0)


def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def bbox_intersections(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] intersec = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    intersec[n, k] = iw * ih / box_area
    return intersec

# Compute bounding box voting
def box_vote(
        np.ndarray[float, ndim=2] dets_NMS,
        np.ndarray[float, ndim=2] dets_all):
    cdef np.ndarray[float, ndim=2] dets_voted = np.zeros((dets_NMS.shape[0], dets_NMS.shape[1]), dtype=np.float32)
    cdef unsigned int N = dets_NMS.shape[0]
    cdef unsigned int M = dets_all.shape[0]

    cdef np.ndarray[float, ndim=1] det
    cdef np.ndarray[float, ndim=1] acc_box
    cdef float acc_score

    cdef np.ndarray[float, ndim=1] det2
    cdef float bi0, bi1, bit2, bi3
    cdef float iw, ih, ua

    cdef float thresh=0.5

    for i in range(N):
        det = dets_NMS[i, :]
        acc_box = np.zeros((4), dtype=np.float32)
        acc_score = 0.0

        for m in range(M):
            det2 = dets_all[m, :]

            bi0 = max(det[0], det2[0])
            bi1 = max(det[1], det2[1])
            bi2 = min(det[2], det2[2])
            bi3 = min(det[3], det2[3])

            iw = bi2 - bi0 + 1
            ih = bi3 - bi1 + 1

            if not (iw > 0 and ih > 0):
                continue

            ua = (det[2] - det[0] + 1) * (det[3] - det[1] + 1) + (det2[2] - det2[0] + 1) * (det2[3] - det2[1] + 1) - iw * ih
            ov = iw * ih / ua

            if (ov < thresh):
                continue

            acc_box += det2[4] * det2[0:4]
            acc_score += det2[4]

        dets_voted[i][0:4] = acc_box / acc_score
        dets_voted[i][4] = det[4]       # Keep the original score

    return dets_voted
