# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import ModelUtils
import argparse    
import random

def Draw(feat, labels):
    
    plt.figure(figsize=(16,9))
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    for i in range(10):
        mask = labels==i
        x = feat[mask,0].flatten()
        y = feat[mask,1].flatten()
        plt.plot(x, y, '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.grid()
    plt.show()
    plt.close()
    
    

def TrainModel(model, session, mnist, cfg):

    mean = np.mean(mnist.train.images, axis=0)
    for iteration in range(5000):
        batch = mnist.train.next_batch(128)
        feed_dict = {model.pl_images:batch[0]- mean, model.pl_labels:batch[1]}
        acc, _, center = session.run([model.accuracy, model.op_train, model.loss_center], feed_dict=feed_dict)
        
        if iteration % 200==0:
            print('[batch:{}] acc:{} center:{}'.format(iteration, acc, center))
        
    
    feed_tr = {model.pl_images:mnist.train.images[:10000]-mean}
    feat_tr = session.run(model.deep_features, feed_dict=feed_tr)
    labels_tr = mnist.train.labels[:10000]
    Draw(feat_tr, labels_tr)
    
    
    feed_te = {model.pl_images:mnist.test.images[:10000]-mean}
    feat_te = session.run(model.deep_features, feed_dict=feed_te)
    labels_te = mnist.test.labels[:10000]
    Draw(feat_te, labels_te)


def Run(cfg, mnist):
    g = tf.Graph()
    with g.as_default():
        np.random.seed(seed=cfg.seed)
        random.seed(cfg.seed)
        tf.set_random_seed(cfg.seed)
        model = ModelUtils.MyModel(cfg)
        session = tf.InteractiveSession()
        session.run(model.op_init)
        TrainModel(model, session, mnist, cfg)
        session.close()
    return 

def GetCfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='center')
    #parser.add_argument('--loss', type=str, default='softmax')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lmbda', type=float, default=0.5)
    
    parser.add_argument('--data_dir', type=str, default='F:\\FV_TMP\\Data\\MNIST_data')
    parser.add_argument('--seed', type=int, default=231)
    return parser.parse_args([])

if __name__ == '__main__':
    
    cfg = GetCfg()
    mnist = input_data.read_data_sets(cfg.data_dir , one_hot=False, seed=231)
    Run(cfg, mnist)



