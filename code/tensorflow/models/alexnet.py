import tensorflow as tf
import numpy as np

from model_session import ModelSession
from wrappers import *

class AlexNet(ModelSession):
    """ AlexNet Model
    Including modifications based off of the oral cavity cancer histopath model:
    """
    
    IMAGE_SIZE = 28
    IMAGE_CHANNELS = 1
    CLASSES = 10
    
    @staticmethod
    def create_graph(batch_size=128):
        ''' Create the graph for the VAE.
        In this case we create an encoder and decoder graph, with references to each.
        The "train" scope contains the entropy, optimizer, loss, accuracy measurements.
        Also, trying to use the Layer API for definition.
        '''
        def weight_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.01))
        
        def bias_variable(shape, name=None):
            return tf.Variable(tf.constant(0.0, shape=None), name=name)
        
        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        iteration = tf.Variable(initial_value=0, trainable=False, name='iteration')
        
        # Define the architecture here
        conv1_num, conv1_size = 32, 5
        pool1_size, pool1_stride = 2, 2
        conv2_num, conv2_size = 64, 5
        pool2_size, pool2_stride = 2, 2
        fc1_size, fc1_dropout = 64, 0.5
        fc2_size, fc2_dropout = 64, 0.5
        
        with tf.variable_scope('parameters'):
            # Define input
            x = tf.placeholder(tf.float32, shape=[None, AlexNet.IMAGE_SIZE, AlexNet.IMAGE_SIZE, AlexNet.IMAGE_CHANNELS], name='x')
            y = tf.placeholder(tf.float32, shape=[None, AlexNet.CLASSES], name='y')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(
                inputs=x,
                filters=conv1_num,
                kernel_size=conv1_size,
                padding='same',
                activation=tf.nn.relu
            )
            
            batch1 = tf.layers.batch_normalization(
                inputs=conv1
            )
            
            pool1 = tf.layers.max_pooling2d(
                inputs=batch1,
                pool_size=pool1_size,
                strides=pool1_stride
            )
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', conv1)
            tf.summary.histogram('bias', batch1)
            
        with tf.variable_scope('conv2'):
            
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=conv2_num,
                kernel_size=conv2_size,
                padding='same',
                activation=tf.nn.relu
            )
            
            batch2 = tf.layers.batch_normalization(
                inputs=conv2
            )
            
            pool2 = tf.layers.max_pooling2d(
                inputs=batch2,
                pool_size=pool2_size,
                strides=pool2_stride
            )
            
            print(tf.size(pool2))
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', conv2)
            tf.summary.histogram('bias', batch2)
            
            # Make flat
            pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
            
        with tf.variable_scope('fc1'):
            
            fc1 = tf.layers.dense(
                inputs=pool2_flat,
                units=fc1_size,
                activation=tf.nn.relu
            )
            
            dropout1 = tf.layers.dropout(
                inputs=fc1,
                rate=fc1_dropout
            )
            
            ## Add summaries for TensorBoard
            #tf.summary.histogram('weights', enc_w_latent1)
            #tf.summary.histogram('biases', enc_b_latent1)
            
        with tf.variable_scope('fc2'):
            
            fc2 = tf.layers.dense(
                inputs=dropout1,
                units=fc2_size,
                activation=tf.nn.relu
            )
            
            dropout2 = tf.layers.dropout(
                inputs=fc2,
                rate=fc2_dropout
            )
            
        with tf.variable_scope('output'):
            logits = tf.layers.dense(
                inputs=dropout2,
                units=AlexNet.CLASSES
            )
            
        with tf.variable_scope('train'):
            pred = tf.nn.softmax(logits=logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y,1)), tf.float32), name='accuracy')
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=iteration, name='train_step')
            
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            #tf.summary.scalar('cost', cost)
            
    def __str__(self):
        return 'AlexNet Model (iteration {})'.format(self.session.run(self.iteration))
    
    def train(self, x, y, learning_rate):
        ''' Train model based on mini-batch of input data '''
        return self.session.run([self.train_step, self.iteration], 
                                  feed_dict={self.x: x,
                                             self.y: y,
                                             self.learning_rate: learning_rate})
    
    def test(self, x, y):
        ''' Returns the reconstruction cost (not REALLY testing) '''
        return self.session.run([self.loss], 
                                feed_dict={self.x: x, 
                                           self.y: y})
    
    @property
    def iteration(self):
        return self._tensor('iteration:0')
   
    # Parameters
    @property
    def x(self):
        return self._tensor('parameters/x:0')
    
    @property
    def y(self):
        return self._tensor('parameters/y:0')
    
    @property
    def learning_rate(self):
        return self._tensor('parameters/learning_rate:0')
    
    # Training
    @property
    def train_step(self):
        return self._tensor('train/train_step:0')
    
    @property
    def loss(self):
        return self._tensor('train/loss:0')
    
    @property
    def accuracy(self):
        return self._tensor('train/accuracy:0')
    
    def _tensor(self, name):
        return self.session.graph.get_tensor_by_name(name)
