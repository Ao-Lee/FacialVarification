# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.keras import layers

collection_name = 'regularization'

class _Missing(object):

    def __repr__(self):
        return 'no value'

    def __reduce__(self):
        return '_missing'

_missing = _Missing()

class cached_property(object):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::
        class Foo(object):
            @cached_property
            def foo(self):
                # calculate something important here
                return 42
    The class has to have a `__dict__` in order for this property to
    work.
    """

    # implementation detail: this property is implemented as non-data
    # descriptor.  non-data descriptors are only invoked if there is
    # no entry with the same name in the instance's __dict__.
    # this allows us to completely get rid of the access function call
    # overhead.  If one choses to invoke __get__ by hand the property
    # will still work as expected because the lookup logic is replicated
    # in __get__ for manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value

class AbstractModel():
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.accuracy
        self.op_train
        self.saver = tf.train.Saver()
    
    def prediction(self):
        assert False, 'must be implemented by subclass' 
        
    def deep_features(self):
        assert False, 'must be implemented by subclass' 
    
    @cached_property
    def pl_images(self):
        return tf.placeholder(tf.float32, shape=[None, 28*28])
        
    @cached_property
    def pl_labels(self):
        return tf.placeholder(tf.int32, shape=[None])
    
    @cached_property
    def pl_dropout_rate(self):
        # dropout rate
        # rate=0.1 would drop out 10% of input units
        # set rate to 0.5 for traning
        # set rate to 0 for testing
        return tf.placeholder(tf.float32)
        
    @cached_property
    def pl_phase(self):
        # training phase if set to true
        # testing phase if set to false
        # this placeholder determines different behavior for batch normalization and dropout
        return tf.placeholder(tf.bool)
        
    @cached_property
    def pl_lr(self):
        # to achieve adaptive learning rate
        # a place holder is needed to hold dynamic learning rate in run time
        return tf.placeholder(tf.float32)
        
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
    
    # regularization loss
    @cached_property
    def loss_reg(self):
        self.loss_softmax
        weights = tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection(collection_name)])
        #print(len(tf.get_collection(collection_name)))
        loss_reg = self.cfg.reg * tf.reduce_sum(weights)
        return loss_reg
    
    @cached_property
    def centers(self):
        initializer=tf.constant_initializer(0)
        n_features = self.deep_features.get_shape()[1]
        n_classes = self.prediction.get_shape()[1]
        centers = tf.get_variable('centers', [n_classes, n_features], dtype=tf.float32, initializer=initializer, trainable=False)
        return centers
    
    @cached_property
    def loss_center(self):
        alpha = self.cfg.alpha
        # self.pl_labels            (Batch)
        # self.deep_features        (Batch, n_features)
        # self.centers              (n_classes, n_features)
        centers_batch = tf.gather(self.centers, self.pl_labels)                  #(Batch, n_features)
        numerator = centers_batch - self.deep_features                      #(Batch, n_features)
        # idx                       (Batch)
        # count                     (?)
        _, idx, count = tf.unique_with_counts(self.pl_labels)
        denominator = tf.gather(count, idx)                                 #(Batch)
        denominator = tf.cast(denominator, tf.float32)                      #(Batch)
        denominator = tf.reshape(denominator, [-1, 1])                      #(Batch, 1)
        diff = tf.divide(numerator, denominator) * (1 - alpha)              #(Batch, n_features)
        self.centers = tf.scatter_sub(self.centers, self.pl_labels, diff)             #(n_classes, n_features)
        loss_center = tf.reduce_mean(tf.square(self.deep_features - centers_batch))
        return loss_center
    
    # total loss
    @cached_property
    def loss(self):
        loss = None
        if self.cfg.loss == 'softmax':
            loss = tf.add_n([self.loss_softmax, self.loss_reg])
        if self.cfg.loss == 'center':
            center = tf.scalar_mul(self.cfg.lmbda, self.loss_center)
            loss = tf.add_n([self.loss_softmax, self.loss_reg, center])
        return loss
        
    @cached_property
    def op_train(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.pl_lr)
        op = optimizer.minimize(self.loss, global_step=self.global_step)
        return op
   
    @cached_property
    def accuracy(self):
        #correct = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.pl_labels,1))
        correct = tf.nn.in_top_k(self.prediction, self.pl_labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy
        
    @cached_property
    def op_init(self):
        init = tf.global_variables_initializer()
        return init
    
    def Conv(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
    def Pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
    def BN(self, x):
        if self.cfg.use_bn:
            return tf.layers.batch_normalization(x, training=self.pl_phase)
        else:
            return x
            

    def PReLU(self):
        return layers.PReLU()

    def WeightVariable(self, shape, name):
        if self.cfg.distribution == 'uniform':
            uniform = True
        if self.cfg.distribution == 'normal':
            uniform = False
        init = tf.contrib.layers.xavier_initializer(uniform=uniform, dtype=tf.float32, seed=self.cfg.seed)
        result = tf.get_variable(name, shape=shape, initializer=init)
        tf.add_to_collection(collection_name, result)
        return result
        
    def BiasVariable(self, shape, name):
        return tf.Variable(tf.constant(0, shape=shape, dtype=tf.float32), name=name)
    
class SmallModel(AbstractModel):
    @cached_property
    def deep_features(self):
        #(None, 28*28) -> (None, 28, 28, 1)
        with tf.name_scope('reshape'):
            x_image = tf.reshape(self.pl_images, [-1, 28, 28, 1])
        
        #(None, 28, 28, 1) -> (None, 14, 14, 32)
        with tf.name_scope('conv1'):
            W_conv1 = self.WeightVariable([5, 5, 1, 32], 'W_conv1')
            b_conv1 = self.BiasVariable([32], 'b_conv1')
            relu_conv1 = self.PReLU()
            h_conv1 = relu_conv1(self.BN(self.Conv(x_image, W_conv1) + b_conv1))
            h_pool1 = self.Pool(h_conv1)
        
        #(None, 14, 14, 32) -> (None, 7, 7, 64)
        with tf.name_scope('conv2'):
            W_conv2 = self.WeightVariable([5, 5, 32, 64], 'W_conv2')
            b_conv2 = self.BiasVariable([64], 'b_conv2')
            relu_conv2 = self.PReLU()
            h_conv2 = relu_conv2(self.BN(self.Conv(h_pool1, W_conv2) + b_conv2))
            h_pool2 = self.Pool(h_conv2)

        #(None, 7, 7, 64) -> (None, 7*7*64) 
        with tf.name_scope('flatten'):
            flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        
        #(None, 7*7*64) -> (None, 1024)
        with tf.name_scope('fc1'):
            W_fc1 = self.WeightVariable([7 * 7 * 64, 1024], 'W_fc1')
            b_fc1 = self.BiasVariable([1024], 'b_fc1')
            relu_fc1 = self.PReLU()
            deep_features = relu_fc1(tf.matmul(flat, W_fc1) + b_fc1)
            #h_fc1_drop = tf.layers.dropout(h_fc1, self.pl_dropout_rate, training=self.pl_phase)
        return deep_features
    
    @cached_property
    def prediction(self):
        #(None, 1024) -> (None, n_classes)
        with tf.name_scope('fc2'):
            W_fc2 = self.WeightVariable([1024, self.cfg.n_classes], 'W_fc2')
            b_fc2 = self.BiasVariable([self.cfg.n_classes], 'b_fc2')
            predicts = tf.matmul(self.deep_features, W_fc2) + b_fc2
        return predicts
    
class BigModel(AbstractModel):  
    @cached_property
    def deep_features(self):
        with tf.name_scope('stage0.reshape'):
            #input      images_pl       (None, 28 * 28)
            #output     x_image         (None, 28, 28, 1)
            x_image = tf.reshape(self.pl_images, [-1, 28, 28, 1])
            
        with tf.name_scope('stage1.conv1'):
            #input      x_image         (None, 28, 28, 1)
            #output     h_stage1_conv1  (None, 28, 28, 32)
            W_stage1_conv1 = self.WeightVariable([5, 5, 1, 32], 'W_stage1_conv1')
            b_stage1_conv1 = self.BiasVariable([32], 'b_stage1_conv1')
            relu_stage1_cov1 = self.PReLU()
            h_stage1_conv1 = relu_stage1_cov1(self.BN(self.Conv(x_image, W_stage1_conv1) + b_stage1_conv1))
            
        with tf.name_scope('stage1.conv2'):
            #input      h_stage1_conv1  (None, 28, 28, 32)
            #output     h_stage1_conv2  (None, 28, 28, 32)
            W_stage1_conv2 = self.WeightVariable([5, 5, 32, 32], 'W_stage1_conv2')
            b_stage1_conv2 = self.BiasVariable([32], 'b_stage1_conv2')
            relu_stage1_cov2 = self.PReLU()
            h_stage1_conv2 = relu_stage1_cov2(self.BN(self.Conv(h_stage1_conv1, W_stage1_conv2) + b_stage1_conv2))
        
        with tf.name_scope('stage1.pool'):
            #input      h_stage1_conv2   (None, 28, 28, 32)
            #output     h_stage1_pool    (None, 14, 14, 32)
            h_stage1_pool = self.Pool(h_stage1_conv2)
            
        with tf.name_scope('stage2.conv1'):
            #input      h_stage1_pool   (None, 14, 14, 32)
            #output     h_stage2_conv1  (None, 14, 14, 64)
            W_stage2_conv1 = self.WeightVariable([5, 5, 32, 64], 'W_stage2_conv1')
            b_stage2_conv1 = self.BiasVariable([64], 'b_stage2_conv1')
            relu_stage2_cov1 = self.PReLU()
            h_stage2_conv1 = relu_stage2_cov1(self.BN(self.Conv(h_stage1_pool, W_stage2_conv1) + b_stage2_conv1))
            
        with tf.name_scope('stage2.conv2'):
            #input      h_stage2_conv1  (None, 14, 14, 64)
            #output     h_stage2_conv2  (None, 14, 14, 64)
            W_stage2_conv2 = self.WeightVariable([5, 5, 64, 64], 'W_stage2_conv2')
            b_stage2_conv2 = self.BiasVariable([64], 'b_stage2_conv2')
            relu_stage2_cov2 = self.PReLU()
            h_stage2_conv2 = relu_stage2_cov2(self.BN(self.Conv(h_stage2_conv1, W_stage2_conv2) + b_stage2_conv2))
        
        with tf.name_scope('stage2.pool'):
            #input      h_stage2_conv2  (None, 14, 14, 64)
            #output     h_stage2_pool   (None, 7, 7, 64)
            h_stage2_pool = self.Pool(h_stage2_conv2)
            
        with tf.name_scope('stage3.conv'):
            #input      h_stage2_pool   (None, 7, 7, 64)
            #output     h_stage3_conv   (None, 7, 7, 128)
            W_stage3_conv = self.WeightVariable([5, 5, 64, 128], 'W_stage3_conv')
            b_stage3_conv = self.BiasVariable([128], 'b_stage3_conv')
            relu_stage3_cov = self.PReLU()
            h_stage3_conv = relu_stage3_cov(self.BN(self.Conv(h_stage2_pool, W_stage3_conv) + b_stage3_conv))
            
        with tf.name_scope('stage3.pool'):
            #input      h_stage3_conv   (None, 7, 7, 128)
            #output     h_stage3_pool   (None, 4, 4, 128)
            h_stage3_pool = self.Pool(h_stage3_conv)
            
        with tf.name_scope('stage4.flatten'):
            #input      h_stage3_pool   (None, 4, 4, 128)
            #output     h_stage4_flat   (None, 4 * 4 * 128)
            h_stage4_flat = tf.reshape(h_stage3_pool, [-1, 4 * 4 * 128])
            
        with tf.name_scope('stage4.fc1'):
            #input      h_stage4_flat   (None, 4 * 4 * 128)
            #output     h_stage4_fc1    (None, 2)
            W_stage4_fc1 = self.WeightVariable([4 * 4 * 128, 1024], 'W_stage4_fc1')
            b_stage4_fc1 = self.BiasVariable([1024], 'b_stage4_fc1')
            relu_stage4_fc1 = self.PReLU()
            deep_features = relu_stage4_fc1(tf.matmul(h_stage4_flat, W_stage4_fc1) + b_stage4_fc1)
        return deep_features
    
    @cached_property
    def prediction(self):
        with tf.name_scope('stage4.fc2'):
            #input      h_stage4_fc1    (None, 2)
            #output     prediction      (None, 10)
            W_stage4_fc2 = self.WeightVariable([1024, 10], 'W_stage4_fc2')
            b_stage4_fc2 = self.BiasVariable([10], 'b_stage4_fc2')
            prediction = tf.matmul(self.deep_features, W_stage4_fc2) + b_stage4_fc2
        return prediction
    
def _Test():
    import ConfigUtils
    cfg = ConfigUtils.ModelCfg()
    model = BigModel(cfg)
    session = tf.InteractiveSession()
    session.run(model.op_init)
    
if __name__=='__main__':
    _Test()

