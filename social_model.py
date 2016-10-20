'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 17th October 2016
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell


class SocialModel():

    def __init__(self, args, infer=False):
        '''
        Initialisation function for the class SocialModel
        params:
        args : Contains arguments required for the Model creation
        '''

        if infer:
            args.batch_size = 1
            args.seq_length = 1

        # Store the arguments
        self.args = args
        self.infer = infer

        cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)

        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        self.input_data = tf.placeholder(tf.float32, [1, 2])
        self.target_data = tf.placeholder(tf.float32, [1, 2])

        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        self.output_size = 5  # 2 mu, 2 sigma and 1 corr

        self.initial_state = cell.zero_state(batch_size=1, dtype=tf.float32)

        self.social_tensor = tf.placeholder(tf.float32, [1, args.grid_size*args.grid_size*args.rnn_size])

        with tf.variable_scope("coordinate_embedding"):
            embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_b = tf.get_variable("embedding_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        with tf.variable_scope("tensor_embedding"):
            embedding_t_w = tf.get_variable("embedding_t_w", [args.grid_size*args.grid_size*args.rnn_size, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        with tf.variable_scope("output_layer"):
            output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.1))

        embedded_input = tf.nn.relu(tf.add(tf.matmul(self.input_data, embedding_w), embedding_b))
        embedded_tensor = tf.nn.relu(tf.add(tf.matmul(self.social_tensor, embedding_t_w), embedding_t_b))

        complete_input = tf.concat(1, [embedded_input, embedded_tensor])
        complete_input = tf.split(1, 1, complete_input)

        output, last_state = tf.nn.rnn(cell, complete_input, initial_state=self.initial_state)
        # output, last_state = tf.nn.seq2seq.rnn_decoder(complete_input, self.initial_state, cell)
        # TODO : Needed?
        output = tf.reshape(tf.concat(1, output), [-1, args.rnn_size])
        self.output = output

        # Apply the linear layer
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        self.final_state = last_state

        [x_data, y_data] = tf.split(1, 2, self.target_data)

        def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
            # eq 3 in the paper
            # and eq 24 & 25 in Graves (2013)
            normx = tf.sub(x, mux)
            normy = tf.sub(y, muy)
            sxsy = tf.mul(sx, sy)
            z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.mul(rho, tf.mul(normx, normy)), sxsy)
            negRho = 1 - tf.square(rho)
            result = tf.exp(tf.div(-z, 2*negRho))
            denom = 2 * np.pi * tf.mul(sxsy, tf.sqrt(negRho))
            result = tf.div(result, denom)
            self.result = result
            return result

        # Important difference between loss func of Social LSTM and Graves (2013)
        # is that it is evaluated over all time steps in the latter whereas it is
        # done from t_obs+1 to t_pred in the former
        def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
            result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

            epsilon = 1e-20  # For numerical stability purposes
            # TODO: (resolve) I don't think we need this as we don't have the inner
            # summation
            # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
            result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

            # TODO: For now, implementing loss func over all time-steps
            return tf.reduce_sum(result1)

        def get_coef(output):
            # eq 20 -> 22 of Graves (2013)
            # TODO : (resolve) Does Social LSTM paper do this as well?
            # the paper says otherwise but this is essential as we cannot
            # have negative standard deviation and correlation needs to be between
            # -1 and 1

            z = output
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z)

            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            z_corr = tf.tanh(z_corr)

            return [z_mux, z_muy, z_sx, z_sy, z_corr]

        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(output)

        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr

        # Compute the loss function
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

        self.cost = lossfunc

        # Get trainable_variables
        tvars = tf.trainable_variables()

        # TODO: (resolve) We are clipping the gradients as is usually done in LSTM
        # implementations. Social LSTM paper doesn't mention about this at all
        self.gradients = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # NOTE: Using RMSprop as suggested by Social LSTM instead of Adam as Graves(2013) does
        # optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # Train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
