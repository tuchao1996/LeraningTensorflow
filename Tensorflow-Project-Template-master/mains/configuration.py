#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @desc:
 @author: ZhangTuchao
 @contact: ztchao1996@163.com
 @github: https://github.com/tuchao1996
 @time: 2020-05-25
"""

class Config:
    def __init__(self):
        # MNIST dataset parameters.
        self.data_path = '/home/zju/.keras/datasets/mnist.npz'
        self.num_classes = 10  # total classes (0-9 digits).

        # Training parameters.
        self.learning_rate = 0.001
        self.training_steps = 200
        self.batch_size = 32
        self.display_step = 10
        self.metrics_type = ['acc']

        # Network parameters.
        self.conv1_filters = 32  # number of filters for 1st conv layer.
        self.conv2_filters = 64  # number of filters for 2nd conv layer.
        self.fc1_units = 1024  # number of neurons for 1st fully-connected layer.


if __name__ == '__main__':
    pass