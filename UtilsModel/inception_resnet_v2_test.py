from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
import inception_resnet_v2 as network



class InceptionTest(tf.test.TestCase):

    def testBuildLogits(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, endpoints = network(inputs, num_classes)
            self.assertTrue('AuxLogits' in endpoints)
            auxlogits = endpoints['AuxLogits']
            self.assertTrue(
                    auxlogits.op.name.startswith('InceptionResnetV2/AuxLogits'))
            self.assertListEqual(auxlogits.get_shape().as_list(),
                                                     [batch_size, num_classes])
            self.assertTrue(logits.op.name.startswith('InceptionResnetV2/Logits'))
            self.assertListEqual(logits.get_shape().as_list(),
                                                     [batch_size, num_classes])

    def testBuildWithoutAuxLogits(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, endpoints = network(inputs, num_classes,
                                                                                                                create_aux_logits=False)
            self.assertTrue('AuxLogits' not in endpoints)
            self.assertTrue(logits.op.name.startswith('InceptionResnetV2/Logits'))
            self.assertListEqual(logits.get_shape().as_list(),
                                                     [batch_size, num_classes])
            
    def testBuildNoClasses(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = None
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            net, endpoints = network(inputs, num_classes)
            self.assertTrue('AuxLogits' not in endpoints)
            self.assertTrue('Logits' not in endpoints)
            self.assertTrue(
                    net.op.name.startswith('InceptionResnetV2/Logits/AvgPool'))
            self.assertListEqual(net.get_shape().as_list(), [batch_size, 1, 1, 1536])

    
def testBuildNoClasses():
    batch_size = 5
    height, width = 299, 299
    num_classes = None
    inputs = tf.random_uniform((batch_size, height, width, 3),name='MyInputs')
    with slim.arg_scope(network.inception_resnet_v2_arg_scope()):
        net, endpoints = network.inception_resnet_v2(inputs, num_classes)
    return
        
          
def GetWriter():
    path = 'logs'
    import os
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
    return writer
                
if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.Session().as_default():
            testBuildNoClasses()
            writer = GetWriter()
            writer.flush()
            writer.close()
    # tf.test.main()