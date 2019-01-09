""" This module implements a classifier/regressor used to evaluate the embeddings on the classification/regression tasks.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import time as time
import logz
import inspect
import types


def measurement(predictions, signals, supervision_type = "classification"):
    """
    """
    if supervision_type == "classification":
        results = np.mean(predictions - signals == 0)
    else:
        results = np.mean(np.square(predictions - signals)) * 0.5
    return results

def build_mlp(input_placeholder,\
                  output_size,\
                  scope,\
                  n_layers = 2,\
                  size = 64, \
                  activation = tf.tanh,\
                  output_activation = None,\
                  stddev = 0.02\
                  ):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 

        Hint: use tf.layers.dense    
    """
    with tf.variable_scope(scope):

        net = tf.layers.dense(inputs = input_placeholder,\
                                  units = size,\
                                  activation = activation,\
                                  #kernel_initializer = tf.random_normal_initializer(stddev=stddev),\
                                  name='l1')



        for i in range(1, n_layers):
            net = tf.layers.dense(inputs = net,\
                                  units=size,\
                                  activation=activation,\
                                  #kernel_initializer=tf.random_normal_initializer(stddev=stddev),\
                                  name = 'l' + str(i+1))
        net = tf.layers.dense(inputs = net,\
                              units = output_size,\
                              activation = output_activation,\
                              #kernel_initializer = tf.random_normal_initializer(stddev=stddev),\
                              name='output')
        return net

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(Supervisor.__init__)[0]
    params = {k: locals_[k] if k in locals_ and not isinstance(locals_[k], types.FunctionType) and k is not "self" else None for k in args}
    logz.save_params(params)
    
class Supervisor(object):
    def __init__(self,\
                     n_batch = 100,\
                     n_feat = None,\
                     n_signal = None,\
                     n_layers = 2,\
                     layer_size = 128,\
                     activation = 0.001,\
                     output_activation = None,\
                     nn_init_stddev = 0.02,\
                     learning_rate = 0.001,\
                     n_epoch = 100,\
                     seed = 0,\
                     supervision_type = "classification",\
                     checkpoint_path = None,\
                     logdir = None):
        """
        Initialize the object.
        
        Args:
            n_batch: mini-batch size.             
            n_feat: number of features
            n_singal: number of labels for classification or 1 for regression
            n_layers: number of layers
            layer_size: number of hidden states per layer
            learning_rate: the initial learning rate of the optimization algorithm.
            n_epoch: number of epochs to train the autoencoder.
            seed: a random seed to create reproducible results.
            checkpoint_path: path to the optimized model parameters.
        """

        assert None not in [n_feat, n_signal, logdir]
        self.n_batch = n_batch
        self.n_feat = n_feat
        self.n_signal = n_signal
        
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.activation = activation
        self.output_activation = output_activation
        self.nn_init_stddev = nn_init_stddev
        
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.seed = seed
        self.supervision_type = supervision_type
        self.checkpoint_path = checkpoint_path

        self.learning_curve = {'train': [], 'val': []}

        setup_logger(logdir, locals())

    def define_placeholders(self):
        """
        """
        
        features = tf.placeholder(shape = [None, self.n_feat], name = "features", dtype = tf.float32)
        if self.supervision_type == "classification":
            signals = tf.placeholder(shape=[None], name = "labels", dtype = tf.int32)
        elif self.supervision_type == "regression":
            signals = tf.placeholder(shape=[None], name = "values", dtype = tf.float32)

        return features, signals

    def policy_forward_pass(self, ob_ph):
        """
        """
        
        output = build_mlp(input_placeholder = self.features,\
                           output_size = self.n_signal,\
                           scope = self.supervision_type,\
                           n_layers = self.n_layers,\
                           size = self.layer_size,\
                           activation = self.activation,\
                           output_activation = self.output_activation,\
                           stddev = self.nn_init_stddev)
        return tf.squeeze(output)

    def get_log_prob(self, logits, labels):
        """
        """
        
        logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits + tf.Variable(1e-4))
        return logprob    


    def create_optimizer(self, loss, learning_rate):
        """
        """

        step = tf.Variable(0, trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = step)
        return optimizer

    def compute_loss(self, session, batch, optimize):
        """
        """

        ops = [self.loss, self.optimizer] if optimize else [self.loss]
        loss =  session.run(ops, {self.features: batch["features"], self.signals: batch["signals"]})

        return loss[0]
            

    def build_computation_graph(self):
        """
        """
        with tf.Graph().as_default() as graph:
            self.features, self.signals = self.define_placeholders()
            self.pred_results = self.policy_forward_pass(self.features)

            if self.supervision_type == "classification":                           
                self.logprob = self.get_log_prob(self.pred_results, self.signals)
                self.loss = tf.reduce_mean(self.logprob)
            elif self.supervision_type == "regression":
                self.loss = tf.reduce_mean(tf.square(self.pred_results - self.signals)) * 0.5

            self.optimizer = self.create_optimizer(self.loss, self.learning_rate)

            self.initializer = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

            graph.finalize()
        return graph
        
        
        
    def fit(self, dataset):
        """
        """      

        self.graph = self.build_computation_graph()
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.initializer)

        num_steps_in_epoch = dataset.num_train // self.n_batch
        n_step = num_steps_in_epoch * self.n_epoch

        start = time.time()
        np.random.seed(self.seed)

        try:
            self.learning_curve['train'].clear()
            self.learning_curve['val'].clear()
            loss_train = 0.
            
            for step in range(n_step):
                local_batch = dataset.next_batch(self.n_batch)
                loss_train += self.compute_loss(self.sess, batch = local_batch, optimize=True)

                
                
                if (step + 1) % num_steps_in_epoch == 0:
                    train_error = self.n_batch / dataset.num_train * loss_train
                    val_error = self.compute_loss(self.sess, batch = dataset.testdata(), optimize = False)
                    # Return the negative error to allow monitoring for the ELBO.
                    self.learning_curve['train'] += [train_error]
                    self.learning_curve['val'] += [val_error]
                    loss_train = 0.

                    logz.log_tabular("Time", time.time() - start)
                    logz.log_tabular("Fold", dataset.test_fold)
                    logz.log_tabular("Epoch", dataset.epochs_completed)
                    logz.log_tabular("BatchStep", step)
                    logz.log_tabular("TrainError", train_error)
                    logz.log_tabular("ValError", val_error)
                    logz.dump_tabular()
                    # print('epoch: {:2d}, step: {:5d}, training error: {:03.4f}, '
                    #       'validation error: {:03.4f}, time elapsed: {:4.0f} s'
                    #       .format(dataset.epochs_completed, step, train_error, val_error, time.time() - start))
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
                
        
        
    def _restore_model(self, session):
        """
        Restores the model from the model's checkpoint path.
        
        Args:
            session: current session which should hold the graph to which the model parameters are assigned.
        """

        self.saver.restore(session, self.checkpoint_path)

    def predict(self, features):
        """
        """

        with tf.Session(graph = self.graph) as session:
            self._restore_model(session)
            pred_results = session.run(self.pred_results, {self.features: features})
            if self.supervision_type == "classification":
                predictions = np.argmax(pred_results, axis = 1)
            elif self.supervision_type == "regression":
                predictions = pred_results
        return predictions

    
