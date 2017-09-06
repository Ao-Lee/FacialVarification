from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

try:
    from .lazy_property import cached_property
    from . import inception_resnet_v2 as network
except ImportError:
    from lazy_property import cached_property
    import inception_resnet_v2 as network
    
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
            sz = self.image_size
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
        self.image_size = 299
        
    @cached_property
    # normalized deep features
    def embeddings(self):
        with tf.name_scope('embeddings', values=[self.deep_features]):
            return tf.nn.l2_normalize(self.deep_features, 1, 1e-10, name='embeddings')
    
    @staticmethod
    def inference(inputs, weight_decay):
        num_classes = None
        with slim.arg_scope(network.inception_resnet_v2_arg_scope(weight_decay=weight_decay)):
            net, endpoints = network.inception_resnet_v2(inputs, num_classes)
        return net, endpoints
        
    @staticmethod
    def GetArgScope(weight_decay):
        batch_norm_params = {'decay': 0.9997, 'epsilon': 0.001, 'fused': None}
        w_regularizer = slim.l2_regularizer(weight_decay)
        b_regularizer = slim.l2_regularizer(weight_decay)
        with slim.arg_scope([slim.fully_connected], 
                            weights_regularizer=w_regularizer, 
                            biases_regularizer=b_regularizer,
                            activation_fn=None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params
                            ) as scope:
            return scope
        
    @cached_property
    def deep_features(self):
        net, endpoints = self.inference(self.pl_images, self.cfg.reg)
        self.endpoints = endpoints
        with tf.name_scope('deep_features', values=[net]):
            with slim.arg_scope(self.GetArgScope(self.cfg.reg)):
                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, scope='FC1')
        return net
        
    @cached_property
    def prediction(self):
        regularizer = slim.l2_regularizer(self.cfg.reg*10)
        with tf.name_scope('prediction', values=[self.deep_features]):
            net = tf.nn.relu(self.deep_features, name='relu')
            net = slim.dropout(net, 0.7, scope='Dropout')
            net = slim.fully_connected(inputs=net, num_outputs=self.cfg.n_classes, activation_fn=None, weights_regularizer=regularizer, scope='FC2')
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
    

       
        
