from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

try:
    from .lazy_property import cached_property
except ImportError:
    from lazy_property import cached_property
    
class AbstractModel():
    
    def __init__(self, cfg):
        self.cfg = cfg
        
    def prediction(self):
        assert False, 'must be implemented by subclass'
        
    def deep_features(self):
        assert False, 'must be implemented by subclass'
        
    def loss(self):    #total loss
        assert False, 'must be implemented by subclass'
        
    @cached_property
    def saver(self):
        self.op_init
        return tf.train.Saver(max_to_keep=5)
    
    @cached_property
    def epoch(self):
        with tf.name_scope('epoch'):
            return tf.Variable(0, dtype=tf.int32, trainable=False, name='epoch')

    @cached_property
    def pl_epoch(self):
        with tf.name_scope('epoch'):
            return tf.placeholder(tf.int32, shape=(), name='input')
    
    @cached_property
    def op_assign_epoch(self):
        with tf.name_scope('epoch'):
            return tf.assign(self.epoch, self.pl_epoch, name='assign')
    
    @cached_property
    def pl_images(self):
        with tf.name_scope('images'):
            sz = self.cfg.image_size
            return tf.placeholder(tf.float32, shape=[None, sz, sz, 3], name='images')
        
    @cached_property
    def pl_labels(self):
        with tf.name_scope('labels'):
            return tf.placeholder(tf.int32, shape=[None], name='labels')
    
    @cached_property
    def pl_phase(self):
        # training phase if set to true
        # testing phase if set to false
        # this placeholder determines different behavior for batch normalization and dropout
        with tf.name_scope('phase'):
            return tf.placeholder(tf.bool, name='phase')
        
    @cached_property
    def pl_lr(self):
        # to achieve adaptive learning rate
        # a place holder is needed to hold dynamic learning rate in run time
        with tf.name_scope('lr'):
            return tf.placeholder(tf.float32, name='lr')
        
    @cached_property
    def global_step(self):
        with tf.name_scope('global_step'):
            return tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    @cached_property
    def op_train(self):
        with tf.name_scope('train', values=[self.loss, self.pl_lr]):
            optimizer = None
            if self.cfg.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.pl_lr)
            if self.cfg.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.pl_lr)
            assert optimizer is not None, 'wrong optimizer option, please check the config file'
            op = optimizer.minimize(self.loss, global_step=self.global_step)
        return op
   
    @cached_property
    def accuracy(self):
        with tf.name_scope('accuracy', values=[self.prediction, self.pl_labels]):
            correct = tf.nn.in_top_k(self.prediction, self.pl_labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy
        
    @cached_property
    def op_summary(self):
        with tf.name_scope('MySummary', values=[self.loss]):
            tf.summary.scalar('scalar_loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            # because you have several summaries, 
            # we should merge them all into one op to make it easier to manage
            summary_op = tf.summary.merge_all()
        return summary_op
    
    @cached_property
    def writer(self):
        '''
        Create the summary writer after graph definition 
        and before running your session
        '''
        path = 'logs'
        import os
        import shutil
        if os.path.exists(path):
            shutil.rmtree(path)
        writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
        return writer
        
    @cached_property
    def op_init(self):
        self.op_train
        self.accuracy
        self.op_assign_epoch
        self.op_summary
        #self.writer
        init = tf.global_variables_initializer()
        return init
    
class Inception_Res_V1(AbstractModel):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
    @cached_property
    # normalized deep features
    def embeddings(self):
        with tf.name_scope('embeddings', values=[self.deep_features]):
            return tf.nn.l2_normalize(self.deep_features, 1, 1e-10, name='embeddings')
    
    @cached_property
    def deep_features(self):
        
        batch_norm_params = {
            'decay': 0.995, # Decay for the moving averages.
            'epsilon': 0.001, # epsilon to prevent 0s in variance.
            'updates_collections': None, # force in-place updates of mean and variance estimates
            'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ], # Moving averages ends up in the trainable variables collection
        }
        regularizer = slim.l2_regularizer(self.cfg.reg)
        
        with tf.variable_scope('InceptionResnetV1', values=[self.pl_images, self.pl_phase]):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                        weights_regularizer=regularizer,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.pl_phase):
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                        net, end_points = self.inference(inputs=self.pl_images)

        self.end_points = end_points
        return net
    
    @cached_property
    def prediction(self):
        regularizer = slim.l2_regularizer(self.cfg.reg*10)
        with tf.name_scope('prediction', values=[self.deep_features]):
            net = tf.nn.relu(self.deep_features, name='relu')
            net = slim.dropout(net, 0.8, scope='Dropout')
            net = slim.fully_connected(inputs=net, num_outputs=self.cfg.n_classes, activation_fn=None, weights_regularizer=regularizer, scope='fully_connected')
        return net
        
    # softmax loss
    @cached_property
    def loss_softmax(self):
        with tf.name_scope('Softmax'):
            cross_entropys = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pl_labels, logits=self.prediction)
            loss_cross_entropy = tf.reduce_mean(cross_entropys)
        return loss_cross_entropy
    
    # regularization loss
    @cached_property
    def loss_reg(self):
        with tf.name_scope('Regularization'):
            self.loss_softmax
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg = tf.reduce_sum(regularization_losses)
        return reg
        
    @cached_property
    def centers(self):
        with tf.name_scope('centers'):
            initializer=tf.constant_initializer(0)
            n_features = self.deep_features.get_shape()[1]
            n_classes = self.prediction.get_shape()[1]
            centers = tf.get_variable('centers', [n_classes, n_features], dtype=tf.float32, initializer=initializer, trainable=False)
        return centers
    
    # center loss
    @cached_property
    def loss_center(self):
        with tf.name_scope('CenterLoss'):
            if self.cfg.loss_method == 'softmax':
                return tf.Variable(0, dtype=tf.float32, trainable=False)
        
            alpha = self.cfg.alpha
            # self.pl_labels            (Batch)
            # self.deep_features        (Batch, n_features)
            # self.centers              (n_classes, n_features)
            centers_batch = tf.gather(self.centers, self.pl_labels)             #(Batch, n_features)
            numerator = centers_batch - self.deep_features                      #(Batch, n_features)
            # idx                       (Batch)
            # count                     (?)
            _, idx, count = tf.unique_with_counts(self.pl_labels)
            denominator = tf.gather(count, idx)                                 #(Batch)
            denominator = tf.cast(denominator, tf.float32)                      #(Batch)
            denominator = tf.reshape(denominator, [-1, 1])                      #(Batch, 1)
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
        with tf.name_scope('Loss', values=[self.prediction, self.pl_labels, self.deep_features]):
            loss = tf.add_n([self.loss_softmax, self.loss_reg, self.loss_center])
        return loss

    # Inception-Renset-A
    # Builds the 35x35 resnet block.
    @staticmethod
    def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None):
        with tf.variable_scope(scope, 'Block35', [net]):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
            mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net
    
    # Inception-Renset-B
    # Builds the 17x17 resnet block.
    @staticmethod
    def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None):
        with tf.variable_scope(scope, 'Block17', [net]):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7], scope='Conv2d_0b_1x7')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1], scope='Conv2d_0c_7x1')
            mixed = tf.concat([tower_conv, tower_conv1_2], 3)
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net
    
    # Inception-Resnet-C
    # Builds the 8x8 resnet block.
    @staticmethod
    def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None):
        with tf.variable_scope(scope, 'Block8', [net]):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3], scope='Conv2d_0b_1x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1], scope='Conv2d_0c_3x1')
            mixed = tf.concat([tower_conv, tower_conv1_2], 3)
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net
      
    @staticmethod
    def reduction_a(net):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, 3, scope='Conv2d_0b_3x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        return net
    
    @staticmethod
    def reduction_b(net):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_3'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
        return net
    
    @staticmethod
    def inference(inputs):
        
        end_points = {}
        # 149 x 149 x 32
        net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        end_points['Conv2d_1a_3x3'] = net
        # 147 x 147 x 32
        net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
        end_points['Conv2d_2a_3x3'] = net
        # 147 x 147 x 64
        net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
        end_points['Conv2d_2b_3x3'] = net
        # 73 x 73 x 64
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
        end_points['MaxPool_3a_3x3'] = net
        # 73 x 73 x 80
        net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')
        end_points['Conv2d_3b_1x1'] = net
        # 71 x 71 x 192
        net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')
        end_points['Conv2d_4a_3x3'] = net
        # 35 x 35 x 256
        net = slim.conv2d(net, 256, 3, stride=2, padding='VALID', scope='Conv2d_4b_3x3')
        end_points['Conv2d_4b_3x3'] = net
                
        # 5 x Inception-resnet-A
        net = slim.repeat(net, 5, Inception_Res_V1.block35, scale=0.17)
        end_points['Mixed_5a'] = net
        
        # Reduction-A
        with tf.variable_scope('Mixed_6a'):
            net = Inception_Res_V1.reduction_a(net)
        end_points['Mixed_6a'] = net
                
        # 10 x Inception-Resnet-B
        net = slim.repeat(net, 10, Inception_Res_V1.block17, scale=0.10)
        end_points['Mixed_6b'] = net
                
        # Reduction-B
        with tf.variable_scope('Mixed_7a'):
            net = Inception_Res_V1.reduction_b(net)
        end_points['Mixed_7a'] = net
                
        # 5 x Inception-Resnet-C
        net = slim.repeat(net, 5, Inception_Res_V1.block8, scale=0.20)
        end_points['Mixed_8a'] = net
                
        net = Inception_Res_V1.block8(net, activation_fn=None)
        end_points['Mixed_8b'] = net
   
        with tf.variable_scope('Logits'):
            end_points['PrePool'] = net
            #pylint: disable=no-member
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
            net = slim.flatten(net)
          
            # net = slim.dropout(net, 0.8, scope='Dropout')

            end_points['PreLogitsFlatten'] = net
                
        net = slim.fully_connected(net, 128, activation_fn=None, scope='Bottleneck')
        return net, end_points
        
if __name__=='__main__':
    from fv.UtilsConfig import Cfg
    cfg = Cfg().model
    cfg.n_classes = 100
    cfg.loss_method = 'softmax'
    
    with tf.Graph().as_default():
        session = tf.Session()
        model = Inception_Res_V1(cfg)
        session.run(model.op_init)
        model.writer.flush()
        model.writer.close()
    

       
        
