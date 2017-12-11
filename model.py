#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月11日 星期一 11时12分02秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import sys
import os
import tensorflow as tf
from numba import jit

from config import cfg
from utils import * 
from group_pointcloud import FeatureNet
from rpn import MiddleAndRPN


class GroupPC(object):

    def __init__(object):
        pass


class RPN3D(object):

    def __init__(self,
            cls='Car',
            batch_size=4,
            learning_rate=0.001,
            max_gradient_norm=5.0,
            alpha=1.5,
            beta=1,
            is_train=True):
        # hyper parameters and status
        self.cls = cls 
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha 
        self.beta = beta 

        # build graph
        self.feature = FeatureNet(training=is_train, batch_size=batch_size)
        self.rpn = MiddleAndRPN(input=self.feature.outputs, alpha=self.alpha, beta=self.beta)
        self.delta_output = self.rpn.delta_output 
        self.prob_outpout = self.rpn.prob_output 
     
        # input placeholders
        self.vox_feature = self.feature.feature 
        self.vox_number = self.feature.number 
        self.vox_coordinate = self.feature.coordinate
        self.targets = self.rpn.targets
        self.pos_equal_one = self.rpn.pos_equal_one 
        self.neg_equal_one = self.rpn.neg_equal_one 
        self.rpn_output_shape = self.rpn.output_shape 
        self.anchors = cal_anchors(self.rpn_output_shape)
        # for predict and image summary 
        self.rgb = tf.placeholder(tf.int64, [None, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, 3])
        self.bv = tf.placeholder(tf.int64, [None, cfg.TOP_WIDTH, cfg.TOP_HEIGHT, 3])
        self.boxes2d = tf.placeholder(tf.int64, [None, 4])
        self.boxes2d_scores = tf.placeholder(tf.float32, [None])

        # NMS(2D)
        self.box2d_ind_after_nms = tf.image.non_max_suppression(self.boxes2d, self.boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)

        # loss and optimizer
        self.loss = self.rpn.loss
        self.reg_loss = self.rpn.reg_loss 
        self.cls_loss = self.rpn.cls_loss 
        self.params = tf.trainable_variables()
        opt = tf.AdamOptimizer(self.learing_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradients_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
    
        # summary and saver
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            tf.summary.scalar('train/reg_loss', self.reg_loss),
            tf.summary.scalar('train/cls_loss', self.cls_loss),
            *[tf.summary.histogram(each.name, each) for each in self.params]
        ])

        self.validate_summary = tf.summary.merge([
            tf.summary.scalar('validate/loss', self.loss),
            tf.summary.scalar('validate/reg_loss', self.reg_loss),
            tf.summary.scalar('validate/cls_loss', self.cls_loss)
        ])

        # TODO: bird_view_summary and front_view_summary
        
        self.predict_summary = tf.summary.merge([
            tf.summary.image('predict/bird_view_lidar', self.bv),   
            tf.summary.image('predict/front_view_rgb', self.rgb), 
        ])


    def train_step(self, session, data, train=False, summary=False):
        # input:  
        #     (N) tag 
        #     (N, N') label
        #     vox_feature 
        #     vox_number 
        #     vox_coordinate
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]

        pos_equal_one, neg_equal_one, targets = cal_rpn_target(label, self.rpn_output_shape, self.anchors)
        input_feed = {
            self.vox_feature: vox_feature,
            self.vox_number: vox_number,  
            self.vox_coordinate: vox_coordinate,
            self.targets: targets, 
            self.pos_equal_one: pos_equal_one,
            self.neg_equal_one: neg_equal_one 
        }
        if train:
            output_feed = [self.loss, self.reg_loss, self.cls_loss, self.gradient_norm, self.update]
        else:
            output_feed = [self.loss, self.reg_loss, self.cls_loss]
        if summary:
            output_feed.append(self.train_summary)
        return session.run(output_feed, input_feed)


    def validate_step(self, session, data, summary=False):
        # input:  
        #     (N) tag 
        #     (N, N') label
        #     vox_feature 
        #     vox_number 
        #     vox_coordinate
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]

        pos_equal_one, neg_equal_one, targets = cal_rpn_target(label, self.rpn_output_shape, self.anchors)
        input_feed = {
            self.vox_feature: vox_feature,
            self.vox_number: vox_number,  
            self.vox_coordinate: vox_coordinate,
            self.targets: targets, 
            self.pos_equal_one: pos_equal_one,
            self.neg_equal_one: neg_equal_one 
        }
        output_feed = [self.loss, self.reg_loss, self.cls_loss]
        if summary:
            output_feed.append(self.validate_summary)
        return session.run(output_feed, input_feed)
   
    
    def predict_step(self, session, data, summary=False, with_gt=False):
        # input:  
        #     (N) tag 
        #     (N, N') label
        #     vox_feature 
        #     vox_number 
        #     vox_coordinate
        # output: A, B, C
        #     A: (N) tag
        #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
        #     C; summary(optional) 
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        img = data[5]
        lidar = data[6]

        batch_gt_boxes3d = label_to_gt_box3d(label, cls=self.cls, coordinate='lidar')
        input_feed = {
            self.vox_feature: vox_feature,
            self.vox_number: vox_number,  
            self.vox_coordinate: vox_coordinate,
        }

        output_feed = [self.prob_outpout, self.delta_output]
        probs, deltas = sess.run(output_feed, input_feed)
        boxes3d = delta_to_boxes3d(deltas, self.rpn_output_shape, self.anchors, coordinate='lidar')
        boxes2d = boxes3d[:, :, [0,1,4,5]]
        probs = prob.reshape((self.batch_size, -1))
        # NMS 
        ret_box3d = []
        ret_score = []
        for idx in range(self.batch_size):
            ind = sess.run(self.box2d_ind_after_nms, {
                self.boxes2d: boxes2d,
                self.boxes2d_scores: probs 
            })    
            tmp_box3d = boxes3d[idx, ind, ...]
            tmp_score = probs[idx, ind]
            ind = np.where(tmp_score >= cfg.RPN_SCORE_THRESH)[0]
            ret_box3d.append(tmp_box[ind, ...])
            ret_score.append(tmp_score[ind])

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis], 
                boxes3d, scores[:, np.newaxis]]))

        if summary:
            # only summry 1 in a batch 
            front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0], 
                    batch_gt_boxes3d[0] if with_gt else [])
            bird_view = lidar_to_bird_view(lidar[0])
            bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0], 
                    batch_gt_boxes3d[0] if with_gt else [])
            ret_summary = sess.run(self.predict_summary, {
                self.rgb: front_image[np.newaxis, ...],
                self.bv: bird_view[np.newaxis, ...]
            })
        
        return tag, ret_box3d_score, ret_summary 


if __name__ == '__main__':
    pass	
