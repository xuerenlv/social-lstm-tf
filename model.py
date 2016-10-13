'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
'''


import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell


# TODO: For now just implementing vanilla LSTM without the social layer
class Model():

    def __init__(self, args):
        '''
        Initialisation function for the class Model.
        Params:
        args: Contains arguments required for the Model creation
        '''
        self.args = args

        # args.rnn_size contains the dimension of the hidden state of the LSTM
        cell = rnn_cell.BASICLSTMCell(args.rnn_size)
        
        # IMPROV: For now, let's use a single layer of LSTM
        # IMRPOV: Dropout layer can be added here
        self.cell = cell

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])
        self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])

        self.lr = tf.Variable(args.learning_rate, name="learning_rate")

        # Initial cell state of the LSTM (initialised with zeros)
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # Embedding 
        with tf.variable_scope("coordinate_embedding"):
            # The spatial embedding layer
            # Embed the 2D coordinates into embedding_size dimensions
            # TODO: For now assume embedding_size = rnn_size
            embedding = tf.get_variable("embedding", [2, args.embedding_size])

        # Output linear layer
        with tf.variable_scope("output_layer"):
            # args.output_size contains the number of outputs of the RNN
            output_w = tf.get_variable("output_w", [args.rnn_size, args.output_size])
            output_b = tf.get_variable("output_b", [args.output_size])

        # Embed inputs
        inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope="rnnlm")
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])

        # Apply the linear layer
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        self.final_state = last_state

        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(self.target_data, [-1, 2])
        [x_data, y_data] = tf.split(1, 2, flat_target_data)

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
            return result

        # Important difference between loss func of Social LSTM and Graves (2013)
        # is that it is evaluated over all time steps in the latter whereas it is
        # done from t_obs+1 to t_pred in the former
        def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data, args):
            result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

            epsilon = 1e-20 # For numerical stability purposes
            # TODO: (resolve) I don't think we need this as we don't have the inner
            # summation
            # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
            result1 = tf.log(tf.maximum(result0, epsilon)) # Numerical stability

            # TODO: For now, implementing loss func over all time-steps
            return tf.reduce_sum(result1)

        def get_coef(output):
            # eq 20 -> 22 of Graves (2013)
            # TODO : (resolve) Does Social LSTM paper do this as well?
            # the paper says otherwise but this is essential as we cannot
            # have negative standard deviation and correlation needs to be between
            # -1 and 1

            z = output
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z[:,1:])

            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            z_corr = tf.tanh(z_corr)

            return [z_mux, z_muy, z_sx, z_sy, z_corr]

        
