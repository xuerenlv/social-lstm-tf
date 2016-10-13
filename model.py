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
        
        # TODO: (improve) For now, let's use a single layer of LSTM
        # TODO: (improve) Dropout layer can be added here
        self.cell = cell

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])
        self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])

        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Initial cell state of the LSTM (initialised with zeros)
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # Embedding 
        with tf.variable_scope("coordinate_embedding"):
            # The spatial embedding layer
            # Embed the 2D coordinates into embedding_size dimensions
            # TODO: (improve) For now assume embedding_size = rnn_size
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
            result1 = tf.log(tf.maximum(result0, epsilon))  # Numerical stability

            # TODO: For now, implementing loss func over all time-steps
            return tf.reduce_sum(result1)

        def get_coef(output):
            # eq 20 -> 22 of Graves (2013)
            # TODO : (resolve) Does Social LSTM paper do this as well?
            # the paper says otherwise but this is essential as we cannot
            # have negative standard deviation and correlation needs to be between
            # -1 and 1

            z = output
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z[:, 1:])

            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            z_corr = tf.tanh(z_corr)

            return [z_mux, z_muy, z_sx, z_sy, z_corr]

        # Extract the coef from the output of the linear layer
        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(output)

        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr

        # Compute the loss function
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data, args)

        # Compute the cost
        self.cost = lossfunc / (args.batch_size * args.seq_length)

        # Get trainable_variables
        tvars = tf.trainable_variables()

        # TODO: (resolve) We are clipping the gradients as is usually done in LSTM
        # implementations. Social LSTM paper doesn't mention about this at all
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)

        # NOTE: Using RMSprop as suggested by Social LSTM instead of Adam as Graves(2013) does
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # Train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, traj, num=10):
        '''
        Given an initial trajectory (as a list of tuples of points), predict the future trajectory
        until a few timesteps
        Params:
        sess: Current session of Tensorflow
        traj: List of past trajectory points (as tuples)
        num: Number of time-steps into the future to be predicted
        '''
        def sample_gaussian_2d(mux, muy, sx, sy, rho):
            # Extract mean
            mean = [mux, muy]
            # Extract covariance matrix
            cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
            # Sample a point from the multivariate normal distribution
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        # Initial state with zeros
        state = sess.run(self.cell.zero_state(1, tf.float32))

        # Iterate over all the positions seen in the trajectory
        for pos in traj[:-1]:
            # Create the input data tensor
            data = np.zeros((1, 1, 2), dtype=np.float32)
            data[0, 0, 0] = pos[0]  # x
            data[0, 0, 1] = pos[1]  # y

            # Create the feed dict
            feed = {self.input_data: data, self.initial_state: state}
            # Get the final state after processing the current position
            [state] = sess.run([self.final_state], feed)

        ret = traj
        # Last position in the observed trajectory
        last_pos = traj[-1]

        # Construct the input data tensor for the last point
        prev_data = np.zeros((1, 1, 2), dtype=np.float32)
        prev_data[0, 0, 0] = last_pos[0]  # x
        prev_data[0, 0, 1] = last_pos[1]  # y

        for t in range(num):
            # Create the feed dict
            feed = {self.input_data: prev_data, self.initial_state: state}

            # Get the final state and also the coef of the distribution of the next point
            [o_mux, o_muy, o_sx, o_sy, o_corr, state] = sess.run([self.mux, self.muy, self.sx, self.sy, self.corr, self.final_state], feed)

            # Sample the next point from the distribution
            next_x, next_y = sample_gaussian_2d(o_mux[0], o_muy[0], o_sx[0], o_sy[0], o_corr[0])
            # Append the new point to the trajectory
            ret.append((next_x, next_y))

            # Set the current sampled position as the last observed position
            prev_data[0, 0, 0] = next_x
            prev_data[0, 0, 1] = next_y

        return ret
