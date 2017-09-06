# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.keras import layers
from lazy_property import cached_property
from tensorflow.contrib import slim

class MyModel():
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.regularizer = slim.l2_regularizer(self.cfg.reg)
        self.accuracy
        self.op_train
        
    
    @cached_property
    def deep_features(self):
        batch_norm_params = {
                'decay': 0.995, # Decay for the moving averages.
                'epsilon': 0.001, # epsilon to prevent 0s in variance.
                'updates_collections': None, # force in-place updates of mean and variance estimates
                'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ], # Moving averages ends up in the trainable variables collection
                }
        with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], kernel_size=2):
                #with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=self.regularizer, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                    features = inference(self.pl_images)
        return features
            
    @cached_property
    def prediction(self):
        prelu = layers.PReLU()
        x = prelu(self.deep_features)
        prediction = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')
        return prediction
    
    @cached_property
    def pl_images(self):
        return tf.placeholder(tf.float32, shape=[None, 28*28, ])
        
    @cached_property
    def pl_labels(self):
        return tf.placeholder(tf.int32, shape=[None])
       
    @cached_property
    def global_step(self):
        with tf.name_scope('global'):
            step = tf.Variable(1, name='global_step', dtype=tf.int32, trainable=False)
        return step
   
    # cross entropy loss
    @cached_property
    def loss_softmax(self):
        # cross_entropys = tf.nn.softmax_cross_entropy_with_logits(labels=self.pl_labels, logits=self.prediction)
        cross_entropys = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pl_labels, logits=self.prediction)
        loss_cross_entropy = tf.reduce_mean(cross_entropys)
        return loss_cross_entropy
    
    @cached_property
    def centers(self):
        initializer=tf.constant_initializer(0)
        n_features = self.deep_features.get_shape()[1]
        n_classes = self.prediction.get_shape()[1]
        centers = tf.get_variable('centers', [n_classes, n_features], dtype=tf.float32, initializer=initializer, trainable=False)
        return centers
    
    @cached_property
    def loss_center(self):
        print('hahaha')
        alpha = self.cfg.alpha
        centers_batch = tf.gather(self.centers, self.pl_labels)             #(Batch, n_features)
        numerator = centers_batch - self.deep_features                      #(Batch, n_features)
        
        _, idx, count = tf.unique_with_counts(self.pl_labels)
        denominator = tf.gather(count, idx)                                 #(Batch)
        denominator = tf.cast(denominator, tf.float32)                      #(Batch)
        denominator = tf.reshape(denominator, [-1, 1]) 

                    
        diff = tf.divide(numerator, denominator) * alpha                    #(Batch, n_features)
        self.centers = tf.scatter_sub(self.centers, self.pl_labels, diff)   #(n_classes, n_features)
        square = tf.square(self.deep_features - centers_batch)
        loss_batch = tf.reduce_sum(square, axis=1)
        loss = tf.reduce_mean(loss_batch)
        result = tf.scalar_mul(self.cfg.lmbda, loss)
        return result
    
    # total loss
    @cached_property
    def loss(self):
        loss = None
        if self.cfg.loss == 'softmax':
            loss = self.loss_softmax
        elif self.cfg.loss == 'center':
            center = tf.scalar_mul(self.cfg.lmbda, self.loss_center)
            loss = tf.add_n([self.loss_softmax, center])
        return loss
        
    @cached_property
    def op_train(self):
        optimizer = tf.train.AdamOptimizer(0.001)
        op = optimizer.minimize(self.loss, global_step=self.global_step)
        return op
   
    @cached_property
    def accuracy(self):
        correct = tf.nn.in_top_k(self.prediction, self.pl_labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy
        
    @cached_property
    def op_init(self):
        init = tf.global_variables_initializer()
        return init
    
   
def inference(imgs):
    x = tf.reshape(imgs, [-1, 28, 28, 1])
    x = slim.conv2d(x, num_outputs=32, scope='conv1_1')
    x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
    x = slim.max_pool2d(x, scope='pool1')
     
    x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
    x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
    x = slim.max_pool2d(x, scope='pool2')
            
    x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
    x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
    x = slim.max_pool2d(x, scope='pool3')
            
    x = slim.flatten(x, scope='flatten')
           
    x = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')
    return x
     

