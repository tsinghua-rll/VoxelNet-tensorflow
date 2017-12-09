#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : 2017年12月09日 星期六 19时50分00秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import sys
import os
import tensorflow as tf

from config import cfg
from util import * 


class GroupPC(object):

    def __init__(object):
        pass


class RPN3D(object):

    def __init__(self,
            learning_rate=0.001,
            max_gradient_norm=5.0)
        # submodel 
        self.group_pc = GroupPC()
        
        # hyper parameters and status
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        
        # input placeholders

        # build graph

        # loss and optimizer
        self.params = tf.trainable_variables()
        opt = tf.AdamOptimizer(self.learing_rate)
        gradients = tf.gradients(loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradients_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
    
        # summary and saver
        # TODO: add image log
        tf.summary.scalar('loss/step', self.loss)
        for param in self.params:
            tf.summary.histogram(param.name, param)
        self.merged_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def train_step(self, session, data, train=False, summary=False):
        # input:  
        #     (N) tag 
        #     (N, N') label
        #     (N, h, w, 3) rgb
        #     (N, N', 4)  lidar
        input_feed = {}
        if train:
            output_feed = [self.loss, self.accuracy, self.gradient_norm, self.update]
        else:
            output_feed = [self.loss, self.accuracy]
        if summary:
            output_feed.append(self.merged_summary_op)
        return session.run(output_feed, input_feed)
    
    
    def predict_step(self, session, data):
        # input:  
        #     (N) tag 
        #     (N, h, w, 3) rgb
        #     (N, N', 4)  lidar
        # output: A, B
        #     A: (N) tag
        #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
        input_feed = {}
        output_feed = []
        return session.run(output_feed, input_feed)


if __name__ == '__main__':
    pass	
