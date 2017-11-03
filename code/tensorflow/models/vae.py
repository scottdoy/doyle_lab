import tensorflow as tf
import numpy as np

# ModelSession is the superclass that is used to set up the session manager and
# control things getting saved to / loaded from checkpoint files.
from model_session import ModelSession
from wrappers import *

class VAEModelSession(ModelSession):
    """ Variational Autoencoder with sklearn-like interface.
    Modified to fit into the model-based architecture we're using for TF models.
    Also using ModelSession, which should allow for easy saving / restoring (we'll see...)
    ModelSession: https://github.com/wpm/tf_model_session/blob/master/mnist_model.py
    """
    
    # Commenting xavier initialization for now
    #def xavier_init(fan_in, fan_out, constant=1):
    #    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    #    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    #    return tf.random_uniform((fan_in, fan_out),
    #                            minval=low, maxval=high,
    #                            dtype=tf.float32)
    
    IMAGE_SIZE = 28
    CLASSES = 10
    
    @staticmethod
    def create_graph(encoder_nodes=500, decoder_nodes=500, latent_dims=20):
        ''' Create the graph for the VAE.
        In this case we create an encoder and decoder graph, with references to each.
        The "train" scope contains the entropy, optimizer, loss, accuracy measurements.
        Also, trying to use the Layer API for definition.
        '''
        def weight_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        
        def bias_variable(shape, name=None):
            return tf.Variable(tf.constant(0.1, shape=None), name=name)
        
        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        iteration = tf.Variable(initial_value=0, trainable=False, name='iteration')
        
        with tf.variable_scope('parameters'):
            # Define input
            x = tf.placeholder(tf.float32, shape=[None, VAEModelSession.IMAGE_SIZE * VAEModelSession.IMAGE_SIZE], name='x')
            y = tf.placeholder(tf.float32, shape=[None, VAEModelSession.CLASSES], name='y')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            
        with tf.variable_scope('encoder_1'):
            # Define ops for first layer of the encoder
            x_image = x
            enc_w_layer1 = weight_variable([VAEModelSession.IMAGE_SIZE*VAEModelSession.IMAGE_SIZE, encoder_nodes])
            enc_b_layer1 = bias_variable([encoder_nodes], 'bias')
            enc_h_layer1 = tf.nn.softplus(tf.add(tf.matmul(x_image, enc_w_layer1), enc_b_layer1))
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', enc_w_layer1)
            tf.summary.histogram('biases', enc_b_layer1)
            tf.summary.histogram('activations', enc_h_layer1)
            
        with tf.variable_scope('encoder_2'):
            # Define ops for second layer of the encoder
            enc_w_layer2 = weight_variable([encoder_nodes, encoder_nodes])
            enc_b_layer2 = bias_variable([encoder_nodes], 'bias')
            enc_h_layer2 = tf.nn.softplus(tf.add(tf.matmul(enc_h_layer1, enc_w_layer2), enc_b_layer2))
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', enc_w_layer2)
            tf.summary.histogram('biases', enc_b_layer2)
            tf.summary.histogram('activations', enc_h_layer2)
            
        with tf.variable_scope('encoder_latent'):
            # Define latent output of encoder
            enc_w_latent1 = weight_variable([encoder_nodes, latent_dims])
            enc_b_latent1 = bias_variable([latent_dims], 'bias1')
            z_mean = tf.add(tf.matmul(enc_h_layer2, enc_w_latent1), enc_b_latent1, name='z_mean')
            
            enc_w_latent2 = weight_variable([encoder_nodes, latent_dims])
            enc_b_latent2 = bias_variable([latent_dims], 'bias2')
            z_log_sigma_sq = tf.add(tf.matmul(enc_h_layer2, enc_w_latent2), enc_b_latent2)
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', enc_w_latent1)
            tf.summary.histogram('biases', enc_b_latent1)
           
        with tf.variable_scope('sampler'):
            # Draw one sample z from the Gaussian distribution
            # First size argument: (batch_size, latent_dims)
            eps = tf.random_normal((64, latent_dims), 0, 1, dtype=tf.float32)

            # Calculate z = mu + sigma*epsilon
            z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps), name='z')

        with tf.variable_scope('decoder_1'):
            # Use the decoder / generator to determine mean of
            # Bernoulli distribution of reconstructed output
            dec_w_layer1 = weight_variable([latent_dims, decoder_nodes])
            dec_b_layer1 = bias_variable([decoder_nodes], 'bias')
            dec_h_layer1 = tf.nn.softplus(tf.add(tf.matmul(z, dec_w_layer1), dec_b_layer1))
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', dec_w_layer1)
            tf.summary.histogram('biases', dec_b_layer1)
            tf.summary.histogram('activations', dec_h_layer1)
        
        with tf.variable_scope('decoder_2'):
            dec_w_layer2 = weight_variable([decoder_nodes, decoder_nodes])
            dec_b_layer2 = bias_variable([decoder_nodes], 'bias')
            dec_h_layer2 = tf.nn.softplus(tf.add(tf.matmul(dec_h_layer1, dec_w_layer2), dec_b_layer2))
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', dec_w_layer2)
            tf.summary.histogram('biases', dec_b_layer2)
            tf.summary.histogram('activations', dec_h_layer2)
             
        with tf.variable_scope('decoder_reconstr'):
            # Calculate the output reconstruction (same dimensionality as the input)
            dec_w_reconstr = weight_variable([decoder_nodes, VAEModelSession.IMAGE_SIZE * VAEModelSession.IMAGE_SIZE])
            dec_b_reconstr = bias_variable([VAEModelSession.IMAGE_SIZE * VAEModelSession.IMAGE_SIZE], 'bias')
            
            # Add summaries for TensorBoard
            tf.summary.histogram('weights', dec_w_reconstr)
            tf.summary.histogram('biases', dec_b_reconstr)
            
            x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(dec_h_layer2, dec_w_reconstr), dec_b_reconstr), name='x_reconstr_mean')
            
        with tf.variable_scope('train'):
            # Calculate the optimizer and error function
            # Cost = reconstruction loss + latent loss
            reconstr_loss = -tf.reduce_sum(x_image * tf.log(1e-5 + x_reconstr_mean)
                                        + (1-x_image) * tf.log(1e-5 + 1 - x_reconstr_mean),
                                           1, name='reconstr_loss')

            latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                            - tf.square(z_mean)
                                               - tf.exp(z_log_sigma_sq), 1, name='latent_loss')

            cost = tf.reduce_mean(reconstr_loss + latent_loss, name='cost')
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=iteration, name='train_step')
            
            #tf.summary.scalar('reconstr_loss', reconstr_loss)
            #tf.summary.scalar('latent_loss', latent_loss)
            tf.summary.scalar('cost', cost)
            
    def __str__(self):
        return 'MNIST VAE Model(iteration {})'.format(self.session.run(self.iteration))
    
    def train(self, x, y, learning_rate):
        ''' Train model based on mini-batch of input data '''
        return self.session.run([self.train_step, self.iteration], 
                                  feed_dict={self.x: x,
                                             self.y: y,
                                             self.learning_rate: learning_rate})
    
    def test(self, x, y):
        ''' Returns the reconstruction cost (not REALLY testing) '''
        return self.session.run([self.cost, self.reconstr_loss, self.latent_loss], feed_dict={self.x: x, self.y: y})
    
    def transform(self, x):
        ''' Map data onto the latent space. '''
        return self.session.run(self.z_mean, feed_dict={self.x: x})
    
    def generate(self, z_mu=None):
        ''' Generate data by sampling from latent space.
        If `z_mu` is `None`, simply generate data from a normally-distributed random value.
        If `z_mu` has a value, then return the mean reconstruction for that point in latent space.
        '''
        if z_mu is None:
            z_mu = np.random.normal(size=self.latent_dims)
        return self.session.run(self.x_reconstr_mean, 
                                feed_dict={self.z: z_mu})
    
    def reconstruct(self, x):
        ''' Use VAE to reconstruct given data. '''
        print('Beginning reconstruction')
        return self.session.run(self.x_reconstr_mean,
                                feed_dict={self.x: x})
   
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
    
    # Encoder
    @property
    def enc_b_layer1(self):
        return self._tensor('encoder_1/bias:0')
    
    @property
    def enc_b_layer2(self):
        return self._tensor('encoder_2/bias:0')
    
    @property
    def enc_b_latent1(self):
        return self._tensor('encoder_latent/bias1:0')
    
    @property
    def enc_b_latent2(self):
        return self._tensor('encoder_latent/bias2:0')
    
    @property
    def z_mean(self):
        return self._tensor('encoder_latent/z_mean:0')
    
    @property
    def z(self):
        return self._tensor('sampler/z:0')
    
    # Decoder
    @property
    def dec_b_layer1(self):
        return self._tensor('decoder_1/bias:0')
    
    @property
    def dec_b_layer2(self):
        return self._tensor('decoder_2/bias:0')
    
    @property
    def dec_b_reconstr(self):
        return self._tensor('decoder_reconstr/bias:0')
    
    # Training
    @property
    def train_step(self):
        return self._tensor('train/train_step:0')
    
    @property
    def cost(self):
        return self._tensor('train/cost:0')
    
    @property
    def x_reconstr_mean(self):
        return self._tensor('decoder_reconstr/x_reconstr_mean:0')
    
    @property
    def reconstr_loss(self):
        return self._tensor('train/reconstr_loss:0')
    
    @property
    def latent_loss(self):
        return self._tensor('train/latent_loss:0')
    
    def _tensor(self, name):
        return self.session.graph.get_tensor_by_name(name)
