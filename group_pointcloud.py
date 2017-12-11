#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File Name : rpn.py
# Purpose :
# Creation Date : 10-12-2017
# Last Modified : 2017年12月11日 星期一 12时08分47秒
# Created By : Wei Zhang

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = out_channels / 2
        with tf.name_scope(name):
            self.dense = tf.layers.Dense(self.units, tf.nn.relu, name='dense')
            self.batch_norm = tf.layers.BatchNormalization(name='batch_norm')

    def apply(self, inputs, training):
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        aggregated = tf.reduce_max(pointwise, axis=0, keep_dims=True)

        repeated = tf.tile(aggregated, [tf.shape(pointwise)[0], 1])

        concatenated = tf.concat([pointwise, repeated], axis=1)

        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size):
        super(FeatureNet, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size 
        # [ΣK, 35/45, 7]
        self.feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w)
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        self.vfe1 = VFELayer(32, 'VFE-1')
        self.vfe2 = VFELayer(128, 'VFE-2')
        self.dense = tf.layers.Dense(128, tf.nn.relu, name='dense')
        self.batch_norm = tf.layers.BatchNormalization(name='batch_norm')

        def compute(packed):
            # feature: [35/45, 7], number: scalar
            feature, number = packed
            # Use only non-empty points as input, notice that in the paper,
            # the part of output corresponding to empty points are zeroed
            x = feature[:number]
            x = self.vfe1.apply(x, self.training)
            x = self.vfe2.apply(x, self.training)
            x = self.dense.apply(x)
            x = self.batch_norm.apply(x, self.training)
            return tf.reduce_max(x, axis=0)

        # [ΣK, 128]
        voxelwise = tf.map_fn(
            compute,
            (self.feature, self.number),
            dtype=tf.float32,
            parallel_iterations=32,
            swap_memory=True)

        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []
    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        coordinate = voxel_dict['coordinate_buffer']
        coordinate_list.append(
            np.pad(coordinate, ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

    feature = np.concatenate(feature_list)
    number = np.concatenate(number_list)
    coordinate = np.concatenate(coordinate_list)
    return batch_size, feature, number, coordinate


def run(batch_size, feature, number, coordinate):
    """
    Input:
        batch_size: scalar, the batch size
        feature: [ΣK, T, 7], voxel input feature buffer
        number: [ΣK], number of points in each voxel
        coordinate: [ΣK, 4], voxel coordinate buffer

        A feature tensor feature[i] has number[i] points in it and is located in
        coordinate[i] (a 1-D tensor reprents [batch, d, h, w]) in the output

        Input format is similiar to what's described in section 2.3 of the paper

        Suppose the batch size is 3, the 3 point cloud is loaded as
        1. feature: [K1, T, 7] (K1 is the number of non-empty voxels)
           number: [K1] (number of points in the corresponding voxel)
           coordinate: [K1, 3] (each row is a tensor reprents [d, h, w])
        2. feature: [K2, T, 7]
           number: [K2]
           coordinate: [K2, 3]
        3. feature: [K3, T, 7]
           number: [K3]
           coordinate: [K3, 3]
        Then the corresponding input is
        batch_size: 3
        feature: [K1 + K2 + K3, T, 7]
        number: [K1 + K2 + K3]
        coordinate: [K1 + K2 + K3, 4] (need to append the batch index of the
                                       corresponding voxel in front of each row)
    Output:
        outputs: [batch_size, 10, 400, 352, 128]
    """
    gpu_options = tf.GPUOptions(visible_device_list='0,2,3')
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={'GPU': 3}
    )

    with tf.Session(config=config) as sess:
        model = FeatureNet(training=False, batch_size=batch_size)
        tf.global_variables_initializer().run()
        for i in range(10):
            time_start = time.time()
            feed = {model.feature: feature,
                    model.number: number,
                    model.coordinate: coordinate}
            outputs = sess.run([model.outputs], feed)
            print(outputs[0].shape)
            time_end = time.time()
            print(time_end - time_start)


def main():
    data_dir = './data/object/training/voxel'
    batch_size = 32

    filelist = [f for f in os.listdir(data_dir) if f.endswith('npz')]

    import time 
    voxel_dict_list = []
    for id in range(0, len(filelist), batch_size):
        pre_time = time.time()
        batch_file = [f for f in filelist[id:id+batch_size]]
        voxel_dict_list = []
        for file in batch_file:
            voxel_dict_list.append(np.load(os.path.join(data_dir, file)))

        # example input with batch size 16
        batch_size, feature, number, coordinate = build_input(voxel_dict_list)
        print(time.time() - pre_time)

    run(batch_size, feature, number, coordinate)


if __name__ == '__main__':
    main()

