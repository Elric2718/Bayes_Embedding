import tensorflow as tf
from tensorflow.contrib import learn
import os
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.keras import layers
import time

class BayesNetwork(object):
    """
    """

    def __init__(self,
                     n_batch = 100,\
                     n_prior = None,\
                     n_hidden = 500,\
                     n_obs = None,\
                     learning_rate = 0.001,\
                     lambda1 = 1,\
                     lambda2 = 1,\
                     n_epoch = 100,\
                     seed = 0,\
                     checkpoint_path = 'checkpoints/model.ckpt'):
        """
        Initialize the object.
        
        Args:
            n_batch: mini-batch size.
            n_prior: demensionality of the variable of prior info.
            n_hidden: number of hidden neurons.
            n_obs: demensionality of the observed variable.
            learning_rate: the initial learning rate of the optimization algorithm.
            n_epoch: number of epochs to train the autoencoder.
            seed: a random seed to create reproducible results.
            checkpoint_path: path to the optimized model parameters.
        """
        self.epsilon = 1e-8
        self.n_batch = n_batch
        self.n_prior = n_prior
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_epoch = n_epoch
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        

        self.learning_curve = {'train': [], 'val': []}                 

    def _create_graph(self):
        """
        Return the computational graph.
        """

        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.seed)
            self.loss = self._create_model()
            self.optimizer = self._create_optimizer(self.loss, self.learning_rate)
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            graph.finalize()
        return graph

    def _restore_model(self, session):
        """
        Restores the model from the model's checkpoint path.
        
        Args:
            session: current session which should hold the graph to which the model parameters are assigned.
        """

        self.saver.restore(session, self.checkpoint_path)
        
    def _create_encoder(self, z):
        """
        """
        
        h = layers.Dense(self.n_hidden, activation='relu')(z)
        mu1 = layers.Dense(self.n_prior)(h)
        log_sigma1_squared = layers.Dense(self.n_prior)(h)
        
        mu2 = layers.Dense(self.n_prior)(h)
        log_sigma2_squared = layers.Dense(self.n_prior)(h)
        
        mu3 = layers.Dense(self.n_obs)(h)
        log_sigma3_squared = layers.Dense(self.n_obs)(h)
        
        mu4 = layers.Dense(self.n_obs)(h)        
        log_sigma4_squared = layers.Dense(self.n_obs)(h)

        return mu1, log_sigma1_squared, mu2, log_sigma2_squared, mu3, log_sigma3_squared, mu4, log_sigma4_squared

    def _create_encoder_v2(self, z):
        """
        """

        h = layers.Dense(self.n_hidden, activation='relu')(z)
        mu1 = layers.Dense(self.n_prior)(h)
        log_sigma1_squared = layers.Dense(self.n_prior)(h)

        mu2 = layers.Dense(self.n_prior)(h)
        log_sigma2_squared = layers.Dense(self.n_prior)(h)

        return mu1, log_sigma1_squared, mu2, log_sigma2_squared
        
    def _create_decoder(self, h1, h2, sigma_squared_obs1, sigma_squared_obs2):
        """
        """
        h1 = tf.nn.l2_normalize(h1, axis = 1)
        f1 = layers.Dense(self.n_hidden, activation = 'relu')(h1)
        f1 = layers.Dense(self.n_obs)(f1)
        f1 = tf.nn.l2_normalize(f1, axis = 1)

        h2 = tf.nn.l2_normalize(h2, axis = 1)
        f2 = layers.Dense(self.n_hidden, activation = 'relu')(h2)
        f2 = layers.Dense(self.n_obs)(f2)
        f2 = tf.nn.l2_normalize(f2, axis = 1)

        z1_minius_z2 = tf.random_normal([self.n_obs], mean = f1 - f2, stddev = tf.sqrt(sigma_squared_obs1 + sigma_squared_obs2))
        
        return z1_minius_z2, h1, h2, f1, f2


    def _compute_prior_parameter(self, h1, h2, z1, z2):
        """
        """
        mu1_prior = tf.zeros(self.n_prior)
        log_sigma1_squared_prior = tf.log(tf.tile(tf.expand_dims(self.lambda1 * tf.nn.moments(h1, [0])[1] + self.epsilon, 0), [self.n_batch, 1]))
        mu2_prior = tf.zeros(self.n_prior)
        log_sigma2_squared_prior = tf.log(tf.tile(tf.expand_dims(self.lambda1 * tf.nn.moments(h2, [0])[1] + self.epsilon, 0), [self.n_batch, 1]))
        
        z1_aug = tf.square(z1 - tf.reduce_mean(z1, 0)) * self.n_batch * 1./(self.n_batch - 1)
        mu3_prior = tf.tile(tf.expand_dims(tf.nn.moments(tf.log(z1_aug + self.epsilon), [0])[0], 0), [self.n_batch, 1])
        log_sigma3_squared_prior = tf.log(tf.tile(tf.expand_dims(self.lambda2 * tf.nn.moments(tf.log(z1_aug + self.epsilon), [0])[1], 0), [self.n_batch, 1]) + self.epsilon)


        z2_aug = tf.square(z2 - tf.reduce_mean(z2, 0)) * self.n_batch * 1./(self.n_batch - 1)
        mu4_prior = tf.tile(tf.expand_dims(tf.nn.moments(tf.log(z2_aug + self.epsilon), [0])[0], 0), [self.n_batch, 1])
        log_sigma4_squared_prior = tf.log(tf.tile(tf.expand_dims(self.lambda2 * tf.nn.moments(tf.log(z2_aug + self.epsilon), [0])[1], 0), [self.n_batch, 1]) + self.epsilon)
        
        
        return mu1_prior, log_sigma1_squared_prior, mu2_prior, log_sigma2_squared_prior, mu3_prior, log_sigma3_squared_prior, mu4_prior, log_sigma4_squared_prior

    def _create_model(self):
        """
        """

        self.z1 = tf.placeholder(tf.float32, [None, self.n_obs])
        self.z2 = tf.placeholder(tf.float32, [None, self.n_obs])
        self.h1 = tf.placeholder(tf.float32, [None, self.n_prior])
        self.h2 = tf.placeholder(tf.float32, [None, self.n_prior])

        # _create_encoder
        #self.mu1, self.log_sigma1_squared, self.mu2, self.log_sigma2_squared, self.mu3, self.log_sigma3_squared, self.mu4, self.log_sigma4_squared = self._create_encoder(tf.concat([self.z1, self.z2], axis = 1))
        # _create_encoder_v2
        self.mu1, self.log_sigma1_squared, self.mu3, self.log_sigma3_squared = self._create_encoder_v2(self.z1)
        self.mu2, self.log_sigma2_squared, self.mu4, self.log_sigma4_squared = self._create_encoder_v2(self.z2)
        
        self.mu1_prior, self.log_sigma1_squared_prior, self.mu2_prior, self.log_sigma2_squared_prior, self.mu3_prior, self.log_sigma3_squared_prior, self.mu4_prior, self.log_sigma4_squared_prior = self._compute_prior_parameter(self.h1, self.h2, self.z1, self.z2)

        self.delta1 = tf.random_normal([self.n_prior], mean = self.mu1, stddev = tf.sqrt(tf.exp(self.log_sigma1_squared)))
        self.delta2 = tf.random_normal([self.n_prior], mean = self.mu2, stddev = tf.sqrt(tf.exp(self.log_sigma2_squared)))

        self.sigma_squared_obs1 = tf.exp(tf.random_normal([self.n_obs], mean = self.mu3, stddev = tf.sqrt(tf.exp(self.log_sigma3_squared))))
        self.sigma_squared_obs2 = tf.exp(tf.random_normal([self.n_obs], mean = self.mu4, stddev = tf.sqrt(tf.exp(self.log_sigma4_squared))))

        self.z1_minius_z2, self.new_h1, self.new_h2, self.f1, self.f2 = self._create_decoder(self.h1 + self.delta1, self.h2 + self.delta2, self.sigma_squared_obs1, self.sigma_squared_obs2)


        # regularization term
        self.regularizer =  0.5 * tf.reduce_sum(1 + self.log_sigma1_squared - self.log_sigma1_squared_prior - (tf.exp(self.log_sigma1_squared) + tf.square(self.mu1 - self.mu1_prior))/(tf.exp(self.log_sigma1_squared_prior) + self.epsilon), 1) 
        self.regularizer += 0.5 * tf.reduce_sum(1 + self.log_sigma2_squared - self.log_sigma2_squared_prior - (tf.exp(self.log_sigma2_squared) + tf.square(self.mu2 - self.mu2_prior))/(tf.exp(self.log_sigma2_squared_prior) + self.epsilon), 1)
        self.regularizer += 0.5 * tf.reduce_sum(1 + self.log_sigma3_squared - self.log_sigma3_squared_prior - (tf.exp(self.log_sigma3_squared) + tf.square(self.mu3 - self.mu3_prior))/(tf.exp(self.log_sigma3_squared_prior) + self.epsilon), 1)
        self.regularizer += 0.5 * tf.reduce_sum(1 + self.log_sigma4_squared - self.log_sigma4_squared_prior - (tf.exp(self.log_sigma4_squared) + tf.square(self.mu4 - self.mu4_prior))/(tf.exp(self.log_sigma4_squared_prior) + self.epsilon), 1)

        # reconstruction term
        self.recon_error = tf.reduce_sum(-tf.log(self.sigma_squared_obs1 + self.sigma_squared_obs2 + self.epsilon)/2. - tf.square(self.z1 - self.z2 - self.z1_minius_z2)/2./(self.sigma_squared_obs1 + self.sigma_squared_obs2 + self.epsilon), 1)
 
        # loss
        loss = -tf.reduce_mean(self.regularizer + self.recon_error)

        return loss        
        

    def _create_optimizer(self, loss, learning_rate):
        """
        """

        step = tf.Variable(0, trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = step)
        return optimizer
        
    def _compute_loss(self, session, batch, optimize = True):
        """
        """

        ops = [self.loss, self.optimizer] if optimize else [self.loss]
        loss = session.run(ops, {self.h1: batch["h1"], self.h2: batch["h2"], self.z1: batch["z1"], self.z2: batch["z2"]})[0]

        return loss
        
    def fit(self, train, validation):
        """
        """

        
        self.graph = self._create_graph() # Create the computational graph
        self.sess = tf.Session(graph=self.graph)
        #coord = tf.train.Coordinator() # Let tensorflow handle multiple threads.
        #threads = tf.train.start_queue_runners(self.sess, coord) 
        self.sess.run(self.initializer) # Initialize the network.
        num_steps_in_epoch = train.num_examples // self.n_batch * train.Decare_rounds
        n_step = num_steps_in_epoch * self.n_epoch 

        start = time.time()
        np.random.seed(self.seed) # Fix the seed used for shuffling the input data

        try:
            self.learning_curve['train'].clear()
            self.learning_curve['val'].clear()
            loss_train = 0.
            loss_val = 0.
            for step in range(n_step):
                loss_train += self._compute_loss(self.sess, batch = train.next_batch(self.n_batch), optimize=True)
                loss_val += self._compute_loss(self.sess, batch = validation.next_batch(int(self.n_batch)), optimize=False)

                if (step + 1) % num_steps_in_epoch == 0:
                    train_error = self.n_batch / train.num_examples * loss_train
                    val_error = self.n_batch / validation.num_examples * loss_val
                    # Return the negative error to allow monitoring for the ELBO.
                    self.learning_curve['train'] += [train_error]
                    self.learning_curve['val'] += [val_error]
                    loss_train = 0.
                    loss_val = 0.
                    print('epoch: {:2d}, step: {:5d}, training error: {:03.4f}, '
                          'validation error: {:03.4f}, time elapsed: {:4.0f} s'
                          .format(train.epochs_completed, step, train_error, val_error, time.time() - start))
        except KeyboardInterrupt:
            print('ending training')
        finally:
            # If interrupted or stopped, store the progress of the model.
            self.saver.save(self.sess, self.checkpoint_path)
            self.sess.close()
            #coord.request_stop()
            #coord.join(threads)
            print('finished training')
        return self                  

    def encode(self, z):
        """
        """

        with tf.Session(graph=self.graph) as session:
            self._restore_model(session)
            mu1 = session.run([self.mu1], {self.z1: z})
        return np.array(mu1)

    def decode(self, mu, h):
        """
        """

        with tf.Session(graph=self.graph) as session:
            self._restore_model(session)
            f1, new_h1 = session.run([self.f1, self.new_h1], {self.h1: h, self.delta1: mu})
            
        return np.array(f1), np.array(new_h1)

    
