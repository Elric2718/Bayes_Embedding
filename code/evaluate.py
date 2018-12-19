import numpy as np
import pandas as pd
import tensorflow as tf
import time as time
import math


def measurement(predictions, labels):
    """
    """

    accuracy = np.mean(predictions - labels == 0)
    return accuracy

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

    
class Classifier(object):
    def __init__(self,\
                     n_batch = 100,\
                     n_feat = None,\
                     n_label = None,\
                     network_dict = None,\
                     learning_rate = 0.001,\
                     n_epoch = 100,\
                     seed = 0,\
                     checkpoint_path = 'checkpoints/classification_model.ckpt'):
        """
        Initialize the object.
        
        Args:
            n_batch: mini-batch size.             
            n_feat: number of features
            n_label: number of labels
            n_layers: number of layers
            layer_size: number of hidden states per layer
            learning_rate: the initial learning rate of the optimization algorithm.
            n_epoch: number of epochs to train the autoencoder.
            seed: a random seed to create reproducible results.
            checkpoint_path: path to the optimized model parameters.
        """

        assert None not in [n_feat, n_label, network_dict]
        self.n_batch = n_batch
        self.n_feat = n_feat
        self.n_label = n_label
        
        self.n_layers = network_dict["n_layers"]
        self.layer_size = network_dict["layer_size"]
        self.activation = network_dict["activation"]
        self.output_activation = network_dict["output_activation"]
        self.nn_init_stddev = network_dict["nn_init_stddev"]
        
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.seed = seed
        self.checkpoint_path = checkpoint_path

        self.learning_curve = {'train': [], 'val': []}
      

    def define_placeholders(self):
        """
        """
        
        features = tf.placeholder(shape = [None, self.n_feat], name = "features", dtype = tf.float32)
        labels = tf.placeholder(shape=[None], name = "labels", dtype = tf.int32)

        return features, labels

    def policy_forward_pass(self, ob_ph):
        """
        """
        
        logits = build_mlp(input_placeholder = self.features,\
                           output_size = self.n_label,\
                           scope = "classification",\
                           n_layers = self.n_layers,\
                           size = self.layer_size,\
                           activation = self.activation,\
                           output_activation = self.output_activation,\
                           stddev = self.nn_init_stddev)
        return logits

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
        loss =  session.run(ops, {self.features: batch["features"], self.labels: batch["labels"]})

        return loss[0]
            

    def build_computation_graph(self):
        """
        """
        with tf.Graph().as_default() as graph:
            self.features, self.labels = self.define_placeholders()

            self.logits = self.policy_forward_pass(self.features)
            
            self.logprob = self.get_log_prob(self.logits, self.labels)
            
            self.loss = tf.reduce_mean(self.logprob)

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

                    print('epoch: {:2d}, step: {:5d}, training error: {:03.4f}, '
                          'validation error: {:03.4f}, time elapsed: {:4.0f} s'
                          .format(dataset.epochs_completed, step, train_error, val_error, time.time() - start))
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
            logits = session.run(self.logits, {self.features: features})
            predictions = np.argmax(logits, axis = 1)
        return predictions

    
