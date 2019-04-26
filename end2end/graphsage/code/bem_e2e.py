""" This module implements the Bayes Embedding method.

"""

import tensorflow as tf
from tensorflow.contrib import learn
import os
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.keras import layers
import time
tfd = tf.contrib.distributions

m_func = lambda x: -1 + tf.log(1 + tf.exp(x))
h_func = lambda x: tf.tanh(x)
h_func_prime = lambda x: 1 - tf.tanh(x) ** 2
eps = 1e-7

# planar flow
def planar_flow(z0, z_dim, length=8, reuse=False):
    z_prev = z0
    for k in range(length):
        with tf.variable_scope('planar_flow_layer_%d' % k, reuse=reuse):
            u = tf.get_variable('u', dtype=tf.float32, shape=(1, z_dim))
            w = tf.get_variable('w', dtype=tf.float32, shape=(1, z_dim))
            b = tf.get_variable('b', dtype=tf.float32, shape=())
            u_hat = (m_func(tf.tensordot(w, u, 2)) - tf.tensordot(w, u, 2)) * (w / tf.reduce_sum(tf.square(w))) + u
            z_prev = z_prev + u_hat * h_func(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b)
    zK = z_prev
    return zK

# normalization flow to model multi-modal distributions
def SLDJ(z0, z_dim, length=8, reuse=False):
    z_prev = z0
    sum_log_det_jacob = 0.
    for k in range(length):
        with tf.variable_scope('planar_flow_layer_%d' % k, reuse=reuse):
            u = tf.get_variable('u', dtype=tf.float32, shape=(1, z_dim))
            w = tf.get_variable('w', dtype=tf.float32, shape=(1, z_dim))
            b = tf.get_variable('b', dtype=tf.float32, shape=())
            u_hat = (m_func(tf.tensordot(w, u, 2)) - tf.tensordot(w, u, 2)) * (w / tf.reduce_sum(tf.square(w))) + u
            affine = h_func_prime(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b) * w
            sum_log_det_jacob += tf.log(eps + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))
            z_prev = z_prev + u_hat * h_func(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b)
    return sum_log_det_jacob

class BayesNetwork(object):
    """
    """

    def __init__(self,
                 n_batch = 100,\
                 n_w = None,\
                 n_hidden = 500,\
                 n_z = None,\
                 w1_inputs = None,\
                 w2_inputs = None,\
                 z1_inputs = None,\
                 z2_inputs = None,\
                 mapping_index = None,\
                 learning_rate = 0.001,\
                 lambda1 = 1,\
                 lambda2 = 1,\
                 #n_epoch = 100,\
                 nf_K = 8,\
                 pair_or_single = "pair"):#,\
                 #seed = 0,\
                 #checkpoint_path = 'checkpoints/model.ckpt',\
                 #log_file = None):
        """
        Initialize the object.
        
        Args:
            n_batch: mini-batch size.
            n_w: demensionality of the variable of prior info.
            n_hidden: number of hidden neurons.
            n_z: demensionality of the observed variable.
            learning_rate: the initial learning rate of the optimization algorithm.
            n_epoch: number of epochs to train the autoencoder.
            seed: a random seed to create reproducible results.
            checkpoint_path: path to the optimized model parameters.
        """
        self.epsilon = 1e-8
        self.n_batch = n_batch
        self.n_w = n_w
        self.n_hidden = n_hidden
        self.n_z = n_z
        self.w1_inputs = w1_inputs
        self.w2_inputs = w2_inputs
        self.z1_inputs = z1_inputs
        self.z2_inputs = z2_inputs
        self.mapping_index = mapping_index
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        #self.n_epoch = n_epoch
        self.nf_K = nf_K
        self.pair_or_single = pair_or_single
        #self.seed = seed
        #self.checkpoint_path = checkpoint_path
        #self.log_file = log_file
        
        #self.learning_curve = {'train': [], 'val': []}                 

    # def _create_graph(self):
    #     """
    #     Return the computational graph.
    #     """

    #     with tf.Graph().as_default() as graph:
    #         tf.set_random_seed(self.seed)
    #         self.loss, self.regu_term1, self.regu_term2, self.regu_term3, self.regu_term4 = self._create_model()
    #         self.optimizer = self._create_optimizer(self.loss, self.learning_rate)
    #         self.initializer = tf.global_variables_initializer()
    #         self.saver = tf.train.Saver()
    #         graph.finalize()
    #     return graph

    # def _restore_model(self, session):
    #     """
    #     Restores the model from the model's checkpoint path.
        
    #     Args:
    #         session: current session which should hold the graph to which the model parameters are assigned.
    #     """

    #     self.saver.restore(session, self.checkpoint_path)
        
    # def _create_encoder(self, input_dat):
    #     """
    #     not used
    #     """
        
    #     h = layers.Dense(self.n_hidden, activation='relu')(input_dat)
    #     mu_delta1 = layers.Dense(self.n_w)(h)
    #     log_sigma_squared_delta1 = layers.Dense(self.n_w)(h)
        
    #     mu_delta2 = layers.Dense(self.n_w)(h)
    #     log_sigma_squared_delta2 = layers.Dense(self.n_w)(h)
        
    #     mu_s1 = layers.Dense(self.n_z)(h)
    #     log_sigma_squared_s1 = layers.Dense(self.n_z)(h)
        
    #     mu_s2 = layers.Dense(self.n_z)(h)        
    #     log_sigma_squared_s2 = layers.Dense(self.n_z)(h)

    #     return mu_delta1, log_sigma_squared_delta1, mu_delta2, log_sigma_squared_delta2, mu_s1, log_sigma_squared_s1, mu_s2, log_sigma_squared_s2

    def _create_encoder_v2(self, input_dat):
        """
        encoder, inference model
        """

        h = layers.Dense(self.n_hidden, activation='relu')(input_dat)
        mu_delta = layers.Dense(self.n_w)(h)
        log_sigma_squared_delta = layers.Dense(self.n_w)(h)

        mu_s = layers.Dense(self.n_z)(h)
        log_sigma_squared_s = layers.Dense(self.n_z)(h)

        return mu_delta, log_sigma_squared_delta, mu_s, log_sigma_squared_s
        
    def _create_decoder(self, w1, w2, sigma_squared_s1, sigma_squared_s2):
        """
        decoder, generative model 
        """
        w1 = tf.nn.l2_normalize(w1, axis = 1)
        f1 = layers.Dense(self.n_hidden, activation = 'relu')(w1)
        f1 = layers.Dense(self.n_z)(f1)
        f1 = tf.nn.l2_normalize(f1, axis = 1)

        if self.pair_or_single == "pair":
            w2 = tf.nn.l2_normalize(w2, axis = 1)
            f2 = layers.Dense(self.n_hidden, activation = 'relu')(w2)
            f2 = layers.Dense(self.n_z)(f2)
            f2 = tf.nn.l2_normalize(f2, axis = 1)

            z1_minius_z2 = tf.random_normal([self.n_z], mean = f1 - f2, stddev = tf.sqrt(sigma_squared_s1 + sigma_squared_s2))
            return z1_minius_z2, w1, w2, f1, f2
        else:

            generated_z1 = tf.random_normal([self.n_z], mean = f1, stddev = tf.sqrt(sigma_squared_s1))
            return generated_z1, w1, f1
        

    def _compute_prior_parameter(self, w1, w2, z1, z2):
        """
        compute parameters for prior distributions
        """
        mu_delta1_prior = tf.zeros(self.n_w)
        log_sigma_squared_delta1_prior = tf.log(tf.tile(tf.expand_dims(self.lambda1 * tf.nn.moments(w1, [0])[1] + self.epsilon, 0), [self.n_batch, 1]))
        if self.pair_or_single == "pair":
            mu_delta2_prior = tf.zeros(self.n_w)
            log_sigma_squared_delta2_prior = tf.log(tf.tile(tf.expand_dims(self.lambda1 * tf.nn.moments(w2, [0])[1] + self.epsilon, 0), [self.n_batch, 1]))

            
        z1_aug = tf.square(z1 - tf.reduce_mean(z1, 0)) * tf.cast(self.n_batch, tf.float32) * 1./(tf.cast(self.n_batch, tf.float32) - 1.)
        mu_s1_prior = tf.tile(tf.expand_dims(tf.nn.moments(tf.log(z1_aug + self.epsilon), [0])[0], 0), [self.n_batch, 1])
        log_sigma_squared_s1_prior = tf.log(tf.tile(tf.expand_dims(self.lambda2 * tf.nn.moments(tf.log(z1_aug + self.epsilon), [0])[1], 0), [self.n_batch, 1]) + self.epsilon)
        if self.pair_or_single == "pair":
            z2_aug = tf.square(z2 - tf.reduce_mean(z2, 0)) * tf.cast(self.n_batch, tf.float32) * 1./(tf.cast(self.n_batch, tf.float32) - 1.)
            mu_s2_prior = tf.tile(tf.expand_dims(tf.nn.moments(tf.log(z2_aug + self.epsilon), [0])[0], 0), [self.n_batch, 1])
            log_sigma_squared_s2_prior = tf.log(tf.tile(tf.expand_dims(self.lambda2 * tf.nn.moments(tf.log(z2_aug + self.epsilon), [0])[1], 0), [self.n_batch, 1]) + self.epsilon)
        
        if self.pair_or_single == "pair":
            return mu_delta1_prior, log_sigma_squared_delta1_prior, mu_delta2_prior, log_sigma_squared_delta2_prior, mu_s1_prior, log_sigma_squared_s1_prior, mu_s2_prior, log_sigma_squared_s2_prior
        else:
            return mu_delta1_prior, log_sigma_squared_delta1_prior, mu_s1_prior, log_sigma_squared_s1_prior

    def _create_model(self):
        """
        create the completed model. First use encode model to get the parameters
        for the approximated posterior distributions, then use the decode model to 
        sample the latent variables, finally compute the loss.

        (mu_delta1, sigma_squared1), (mu_s1, sigma_squared3) represent the means and 
        varainces for the delta_i and s_i; while (mu_delta2, sigma_squared2), 
        (mu_s2, sigma_squared4) represent the means and varainces for the delta_j 
        and s_j represent the means and variances for the delta_j and s_j. Here, 
        (i, j) correspond to the imput two instances, i.e., (z_1, h_1) and 
        (z_2, h_2).
        """

        # self.z1 = tf.placeholder(tf.float32, [None, self.n_z])
        # self.z2 = tf.placeholder(tf.float32, [None, self.n_z])
        # self.w1 = tf.placeholder(tf.float32, [None, self.n_w])
        # self.w2 = tf.placeholder(tf.float32, [None, self.n_w])

        self.w1 = tf.gather(self.w1_inputs, tf.transpose(self.mapping_index)[0])
        self.w2 = tf.gather(self.w2_inputs, tf.transpose(self.mapping_index)[1])
        self.z1 = tf.gather(self.z1_inputs, tf.transpose(self.mapping_index)[0])
        self.z2 = tf.gather(self.z2_inputs, tf.transpose(self.mapping_index)[1])

        # _create_encoder
        #self.mu_delta1, self.log_sigma_squared_delta1, self.mu_delta2, self.log_sigma_squared_delta2, self.mu_s1, self.log_sigma_squared_s1, self.mu_s2, self.log_sigma_squared_s2 = self._create_encoder(tf.concat([self.z1, self.z2], axis = 1))
        # _create_encoder_v2
        # conditional on only z
        # self.mu_delta1, self.log_sigma_squared_delta1, self.mu_s1, self.log_sigma_squared_s1 = self._create_encoder_v2(self.z1)
        # self.mu_delta2, self.log_sigma_squared_delta2, self.mu_s2, self.log_sigma_squared_s2 = self._create_encoder_v2(self.z2)
        # conditional on both z and h

        self.mu_delta1, self.log_sigma_squared_delta1, self.mu_s1, self.log_sigma_squared_s1 = self._create_encoder_v2(tf.concat([self.z1, self.w1], axis = 1))
        if self.pair_or_single == "pair":
            self.mu_delta2, self.log_sigma_squared_delta2, self.mu_s2, self.log_sigma_squared_s2 = self._create_encoder_v2(tf.concat([self.z2, self.w2], axis = 1))
       
        if self.pair_or_single == "pair":
            self.mu_delta1_prior, self.log_sigma_squared_delta1_prior, self.mu_delta2_prior, self.log_sigma_squared_delta2_prior, self.mu_s1_prior, self.log_sigma_squared_s1_prior, self.mu_s2_prior, self.log_sigma_squared_s2_prior = self._compute_prior_parameter(self.w1, self.w2, self.z1, self.z2)
        else:
            self.mu_delta1_prior, self.log_sigma_squared_delta1_prior, self.mu_s1_prior, self.log_sigma_squared_s1_prior = self._compute_prior_parameter(self.w1, None, self.z1, None)

        self.delta1 = tf.random_normal([self.n_w], mean = self.mu_delta1, stddev = tf.sqrt(tf.exp(self.log_sigma_squared_delta1)))
        if self.pair_or_single == "pair":
            self.delta2 = tf.random_normal([self.n_w], mean = self.mu_delta2, stddev = tf.sqrt(tf.exp(self.log_sigma_squared_delta2)))
        
        if self.nf_K > 0:
            self.delta1_K = planar_flow(self.delta1, z_dim = self.n_w, length = self.nf_K)
            if self.pair_or_single == "pair":
                self.delta2_K = planar_flow(self.delta2, z_dim = self.n_w, length = self.nf_K, reuse = True)
        else:
            self.delta1_K = self.delta1
            if self.pair_or_single == "pair":
                self.delta2_K = self.delta2

        self.sigma_squared_s1 = tf.exp(tf.random_normal([self.n_z], mean = self.mu_s1, stddev = tf.sqrt(tf.exp(self.log_sigma_squared_s1))))
        if self.pair_or_single == "pair":
            self.sigma_squared_s2 = tf.exp(tf.random_normal([self.n_z], mean = self.mu_s2, stddev = tf.sqrt(tf.exp(self.log_sigma_squared_s2))))
            
        if self.pair_or_single == "pair":
            self.z1_minius_z2, self.new_w1, self.new_w2, self.f1, self.f2 = self._create_decoder(self.w1 + self.delta1_K, self.w2 + self.delta2_K, self.sigma_squared_s1, self.sigma_squared_s2)
        else:
            self.generated_z1, self.new_w1, self.f1 = self._create_decoder(self.w1 + self.delta1_K, None, self.sigma_squared_s1, None)

        #print(self.log_sigma_squared_s1, self.log_sigma_squared_s1_prior, self.mu_s1, self.mu_s1_prior)
        # regularization term
        if self.nf_K > 0:
            regular_term1 = tfd.MultivariateNormalDiag(loc = self.mu_delta1_prior, scale_diag = tf.exp(self.log_sigma_squared_delta1_prior/2.)).log_prob(self.delta1_K) - tfd.MultivariateNormalDiag(loc = self.mu_delta1, scale_diag = tf.exp(self.log_sigma_squared_delta1/2.)).log_prob(self.delta1) + SLDJ(self.delta1, z_dim = self.n_w, length = self.nf_K, reuse = True)
            if self.pair_or_single == "pair":
                regular_term2 = tfd.MultivariateNormalDiag(loc = self.mu_delta2_prior, scale_diag = tf.exp(self.log_sigma_squared_delta2_prior/2.)).log_prob(self.delta2_K) - tfd.MultivariateNormalDiag(loc = self.mu_delta2, scale_diag = tf.exp(self.log_sigma_squared_delta2/2.)).log_prob(self.delta2) + SLDJ(self.delta2, z_dim = self.n_w, length = self.nf_K, reuse = True)
            else:
                regular_term2 = tf.constant(0.0)
        else:
            regular_term1 =  0.5 * tf.reduce_sum(1 + self.log_sigma_squared_delta1 - self.log_sigma_squared_delta1_prior - (tf.exp(self.log_sigma_squared_delta1) + tf.square(self.mu_delta1 - self.mu_delta1_prior))/(tf.exp(self.log_sigma_squared_delta1_prior) + self.epsilon), 1)
            if self.pair_or_single == "pair":
                regular_term2 = 0.5 * tf.reduce_sum(1 + self.log_sigma_squared_delta2 - self.log_sigma_squared_delta2_prior - (tf.exp(self.log_sigma_squared_delta2) + tf.square(self.mu_delta2 - self.mu_delta2_prior))/(tf.exp(self.log_sigma_squared_delta2_prior) + self.epsilon), 1)
            else:
                regular_term2 = tf.constant(0.0)
            
        regular_term3 = 0.5 * tf.reduce_sum(1 + self.log_sigma_squared_s1 - self.log_sigma_squared_s1_prior - (tf.exp(self.log_sigma_squared_s1) + tf.square(self.mu_s1 - self.mu_s1_prior))/(tf.exp(self.log_sigma_squared_s1_prior) + self.epsilon), 1)
        if self.pair_or_single == "pair":
            regular_term4 = 0.5 * tf.reduce_sum(1 + self.log_sigma_squared_s2 - self.log_sigma_squared_s2_prior - (tf.exp(self.log_sigma_squared_s2) + tf.square(self.mu_s2  - self.mu_s2_prior))/(tf.exp(self.log_sigma_squared_s2_prior) + self.epsilon), 1)
        else:
            regular_term4 = tf.constant(0.0)

        self.regularizer = regular_term1 + regular_term2 + regular_term3 + regular_term4
        # reconstruction term
        if self.pair_or_single == "pair":
            self.recon_error = tf.reduce_sum(-tf.log(self.sigma_squared_s1 + self.sigma_squared_s2 + self.epsilon)/2. - tf.square(self.z1 - self.z2 - self.z1_minius_z2)/2./(self.sigma_squared_s1 + self.sigma_squared_s2 + self.epsilon), 1)
        else:
            self.recon_error = tf.reduce_sum(-tf.log(self.sigma_squared_s1 + self.epsilon)/2. - tf.square(self.z1 - self.generated_z1)/2./(self.sigma_squared_s1 + self.epsilon), 1)
 
        # loss
        loss = -tf.reduce_mean(self.regularizer + self.recon_error)

        return loss, -tf.reduce_mean(regular_term1), -tf.reduce_mean(regular_term2), -tf.reduce_mean(regular_term3), -tf.reduce_mean(regular_term4)
        

    def _create_optimizer(self, loss, learning_rate):
        """
        """

        step = tf.Variable(0, trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = step)
        return optimizer
        
    def _compute_loss(self, session, batch, optimize = True):
        """
        """

        ops = [self.loss, self.regu_term1, self.regu_term2, self.regu_term3, self.regu_term4, self.optimizer] if optimize else [self.loss, self.regu_term1, self.regu_term2, self.regu_term3, self.regu_term4]
        loss, regu_term1, regu_term2, regu_term3, regu_term4 = session.run(ops, {self.w1: batch["w1"], self.w2: batch["w2"], self.z1: batch["z1"], self.z2: batch["z2"]})[0:5]

        return loss, regu_term1, regu_term2, regu_term3, regu_term4
        
    # def fit(self, train, validation):
    #     """
    #     """

        
    #     self.graph = self._create_graph() # Create the computational graph
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     self.sess = tf.Session(graph=self.graph, config = config)
    #     #coord = tf.train.Coordinator() # Let tensorflow handle multiple threads.
    #     #threads = tf.train.start_queue_runners(self.sess, coord) 
    #     self.sess.run(self.initializer) # Initialize the network.
    #     num_steps_in_epoch = train.num_examples // self.n_batch * train.Decare_rounds
    #     n_step = num_steps_in_epoch * self.n_epoch 

    #     start = time.time()
    #     np.random.seed(self.seed) # Fix the seed used for shuffling the input data

    #     try:
    #         self.learning_curve['train'].clear()
    #         self.learning_curve['val'].clear()
    #         loss_train = 0.
    #         loss_train_regu1 = 0.
    #         loss_train_regu2 = 0.
    #         loss_train_regu3 = 0.
    #         loss_train_regu4 = 0.
    #         loss_val = 0.
    #         loss_val_regu1 = 0.
    #         loss_val_regu2 = 0.
    #         loss_val_regu3 = 0.
    #         loss_val_regu4 = 0.


    #         for step in range(n_step):
    #             loc_loss_train, loc_loss_train_regu1, loc_loss_train_regu2, loc_loss_train_regu3, loc_loss_train_regu4 = self._compute_loss(self.sess, batch = train.next_batch(self.n_batch), optimize=True)
    #             loc_loss_val, loc_loss_val_regu1, loc_loss_val_regu2, loc_loss_val_regu3, loc_loss_val_regu4 = self._compute_loss(self.sess, batch = validation.next_batch(int(self.n_batch)), optimize=False)

    #             loss_train += loc_loss_train
    #             loss_train_regu1 += loc_loss_train_regu1
    #             loss_train_regu2 += loc_loss_train_regu2
    #             loss_train_regu3 += loc_loss_train_regu3
    #             loss_train_regu4 += loc_loss_train_regu4
    #             loss_val += loc_loss_val
    #             loss_val_regu1 += loc_loss_val_regu1
    #             loss_val_regu2 += loc_loss_val_regu2
    #             loss_val_regu3 += loc_loss_val_regu3
    #             loss_val_regu4 += loc_loss_val_regu4
                

    #             if (step + 1) % num_steps_in_epoch == 0:
    #                 train_error = self.n_batch / train.num_examples * loss_train
    #                 val_error = self.n_batch / validation.num_examples * loss_val
    #                 train_error_regu1 = self.n_batch / train.num_examples * loss_train_regu1
    #                 train_error_regu2 = self.n_batch / train.num_examples * loss_train_regu2
    #                 train_error_regu3 = self.n_batch / train.num_examples * loss_train_regu3
    #                 train_error_regu4 = self.n_batch / train.num_examples * loss_train_regu4
    #                 val_error_regu1 = self.n_batch / validation.num_examples * loss_val_regu1
    #                 val_error_regu2 = self.n_batch / validation.num_examples * loss_val_regu2
    #                 val_error_regu3 = self.n_batch / validation.num_examples * loss_val_regu3
    #                 val_error_regu4 = self.n_batch / validation.num_examples * loss_val_regu4                                                                                
    #                 # Return the negative error to allow monitoring for the ELBO.
    #                 self.learning_curve['train'] += [train_error]
    #                 self.learning_curve['val'] += [val_error]
    #                 loss_train = 0.
    #                 loss_train_regu1 = 0.
    #                 loss_train_regu2 = 0.
    #                 loss_train_regu3 = 0.
    #                 loss_train_regu4 = 0.
    #                 loss_val = 0.
    #                 loss_val_regu1 = 0.
    #                 loss_val_regu2 = 0.
    #                 loss_val_regu3 = 0.
    #                 loss_val_regu4 = 0.

    #                 store_msg = '---------------------------------\n---------------------------------\n '\
    #                   'epoch: {epoch_id}\n step: {steps} \n training error: {train_error}\n validation error: {val_error}\n '\
    #                   'train error (regu1): {train_error_regu1}\n train error (regu2): {train_error_regu2}\n '\
    #                   'train error (regu3): {train_error_regu3}\n train error (regu4): {train_error_regu4}\n '\
    #                   'val error (regu1): {val_error_regu1}\n val error (regu2): {val_error_regu2}\n '\
    #                   'val error (regu3): {val_error_regu3}\n val error (regu4): {val_error_regu4}\n '\
    #                   'time elapsed: {time_elapsed} s\n'.format(epoch_id = train.epochs_completed,\
    #                                                             steps = step,\
    #                                                             train_error = train_error,\
    #                                                             val_error =  val_error,\
    #                                                             train_error_regu1 = train_error_regu1,\
    #                                                             train_error_regu2 = train_error_regu2,\
    #                                                             train_error_regu3 = train_error_regu3,\
    #                                                             train_error_regu4 = train_error_regu4,\
    #                                                             val_error_regu1 = val_error_regu1,\
    #                                                             val_error_regu2 = val_error_regu2,\
    #                                                             val_error_regu3 = val_error_regu3,\
    #                                                             val_error_regu4 = val_error_regu4,\
    #                                                             time_elapsed = time.time() - start)
    #                 print(store_msg)
    #                 if ((step + 1)/num_steps_in_epoch <= 5\
    #                    or (step + 1)/num_steps_in_epoch >= self.n_epoch - 5)\
    #                    and os.path.isfile(self.log_file):
    #                     write_file=open(self.log_file, "a+")
    #                     write_file.write(store_msg)
    #                     write_file.close()
                    
    #     except KeyboardInterrupt:
    #         print('ending training')
    #     finally:
    #         # If interrupted or stopped, store the progress of the model.
    #         self.saver.save(self.sess, self.checkpoint_path)
    #         self.sess.close()
    #         #coord.request_stop()
    #         #coord.join(threads)
    #         print('finished training')
    #     return self                  

    # def encode(self, z, h):
    #     """
    #     """
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     with tf.Session(graph=self.graph, config = config) as session:
    #         self._restore_model(session)
    #         mu_delta1 = session.run([self.mu_delta1], {self.z1: z, self.w1: h})
    #     return np.array(mu_delta1)

    # def decode(self, mu, h):
    #     """
    #     """
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     with tf.Session(graph=self.graph, config = config) as session:
    #         self._restore_model(session)
    #         f1, new_w1 = session.run([self.f1, self.new_w1], {self.w1: h, self.delta1: mu})
            
    #     return np.array(f1), np.array(new_w1)
   
