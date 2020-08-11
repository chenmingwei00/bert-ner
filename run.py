#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行 BERT NER Server
#@Time    : 2019/1/26 21:00
# @Author  : MaCan (ma_cancan@163.com)
# @File    : run.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
cur_path=os.path.abspath(os.path.dirname(__file__))
roopath=os.path.split(cur_path)[0]
sys.path.append(roopath)


def train_ner():
    from NEZHA_TensorFlow.NEZHA_MODEL.utils.train_helper import get_args_parser
    from NEZHA_TensorFlow.NEZHA_MODEL.model_train.nezha_lstm_ner import train

    args = get_args_parser()
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    train(args=args)


if __name__ == '__main__':
    """
    如果想训练，那么直接 指定参数跑，如果想启动服务，那么注释掉train,打开server即可
    """
    train_ner()
    #start_server()