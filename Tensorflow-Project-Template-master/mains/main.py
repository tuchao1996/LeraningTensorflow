#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @desc:
 @author: ZhangTuchao
 @contact: ztchao1996@163.com
 @github: https://github.com/tuchao1996
 @time: 2020-05-25
"""

import sys, os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath('file')))
sys.path.append(base_dir)
print('sys path: {}'.format(sys.path))

from configuration import Config
from data_loader.data_generator import load_data, data_transform, load_tf_data
from models.conv_Net import ConvNet
from trainers.trainer import train_epoch, metrics_func

def main():
    config = Config()
    data_path = config.data_path
    learning_rate = config.learning_rate
    training_steps = config.training_steps
    display_steps = config.display_step
    batch_size = config.batch_size
    metrics_type = config.metrics_type

    (x_train, y_train), (x_test, y_test) = load_data(data_path)
    x_train = data_transform(x_train)
    x_test = data_transform(x_test)
    train_data = load_tf_data(x_train, y_train, batch_size)

    conv_net = ConvNet(config)

    conv_net = train_epoch(train_data, conv_net, learning_rate
                           , training_steps, display_steps, metrics_type)

    pred = conv_net(x_test)
    print("Test Accuracy: {}".format(metrics_func(pred, y_test, metrics_type)))



if __name__ == '__main__':
    main()