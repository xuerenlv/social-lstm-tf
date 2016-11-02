'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 10th October 2016
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell


# The Vanilla LSTM model
class Model():

    def __init__(self, args, infer=False):
        '''
        Initialisation function for the class Model.
        Params:
        args: Contains arguments required for the Model creation
        '''

        # If sampling new trajectories, then infer mode
        if infer:
            # Infer one position at a time
            args.batch_size = 1
            args.seq_length = 1

        # Store the arguments
        self.args = args

        # Initialize a BasicLSTMCell recurrent unit
        # args.rnn_size contains the dimension of the hidden state of the LSTM
        cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)

        # Multi-layer RNN construction, if more than one layer
        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        # TODO: (improve) Dropout layer can be added here
        # Store the recurrent unit
        self.cell = cell

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])
        # target data contains sequences of (x,y) points as well
        self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])

        # Learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Initial cell state of the LSTM (initialised with zeros)
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # Output size is the set of parameters (mu, sigma, corr)
        output_size = 5  # 2 mu, 2 sigma and 1 corr

        # Embedding for the spatial coordinates
        with tf.variable_scope("coordinate_embedding"):
            #  The spatial embedding using a ReLU layer
            #  Embed the 2D coordinates into embedding_size dimensions
            #  TODO: (improve) For now assume embedding_size = rnn_size
            embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size])
            embedding_b = tf.get_variable("embedding_b", [args.embedding_size])

        # Output linear layer
        with tf.variable_scope("rnnlm"):
            output_w = tf.get_variable("output_w", [args.rnn_size, output_size], initializer=tf.truncated_normal_initializer(stddev=0.01), trainable=True)
            output_b = tf.get_variable("output_b", [output_size], initializer=tf.constant_initializer(0.01), trainable=True)

        # Split inputs according to sequences.
        inputs = tf.split(1, args.seq_length, self.input_data)
        # Get a list of 2D tensors. Each of size numPoints x 2
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # Embed the input spatial points into the embedding space
        embedded_inputs = []
        for x in inputs:
            # Each x is a 2D tensor of size numPoints x 2
            # Embedding layer
            embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
            embedded_inputs.append(embedded_x)

        # Feed the embedded input data, the initial state of the LSTM cell, the recurrent unit to the seq2seq decoder
        outputs, last_state = tf.nn.seq2seq.rnn_decoder(embedded_inputs, self.initial_state, cell, loop_function=None, scope="rnnlm")

        # Concatenate the outputs from the RNN decoder and reshape it to ?xargs.rnn_size
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])

        # Apply the output linear layer
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        # Store the final LSTM cell state after the input data has been feeded
        self.final_state = last_state

        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(self.target_data, [-1, 2])
        # Extract the x-coordinates and y-coordinates from the target data
        [x_data, y_data] = tf.split(1, 2, flat_target_data)

        def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
            '''
            Function that implements the PDF of a 2D normal distribution
            params:
            x : input x points
            y : input y points
            mux : mean of the distribution in x
            muy : mean of the distribution in y
            sx : std dev of the distribution in x
            sy : std dev of the distribution in y
            rho : Correlation factor of the distribution
            '''
            # eq 3 in the paper
            # and eq 24 & 25 in Graves (2013)
            # Calculate (x - mux) and (y-muy)
            normx = tf.sub(x, mux)
            normy = tf.sub(y, muy)
            # Calculate sx*sy
            sxsy = tf.mul(sx, sy)
            # Calculate the exponential factor
            z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.mul(rho, tf.mul(normx, normy)), sxsy)
            negRho = 1 - tf.square(rho)
            # Numerator
            result = tf.exp(tf.div(-z, 2*negRho))
            # Normalization constant
            denom = 2 * np.pi * tf.mul(sxsy, tf.sqrt(negRho))
            # Final PDF calculation
            result = tf.div(result, denom)
            self.result = result
            return result

        # Important difference between loss func of Social LSTM and Graves (2013)
        # is that it is evaluated over all time steps in the latter whereas it is
        # done from t_obs+1 to t_pred in the former
        def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
            '''
            Function to calculate given a 2D distribution over x and y, and target data
            of observed x and y points
            params:
            z_mux : mean of the distribution in x
            z_muy : mean of the distribution in y
            z_sx : std dev of the distribution in x
            z_sy : std dev of the distribution in y
            z_rho : Correlation factor of the distribution
            x_data : target x points
            y_data : target y points
            '''
            step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

            # Calculate the PDF of the data w.r.t to the distribution
            result0_1 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_2 = tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_3 = tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_4 = tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)

            result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4), tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
            result0 = tf.mul(tf.mul(result0, step), step)

            # For numerical stability purposes
            epsilon = 1e-20

            # TODO: (resolve) I don't think we need this as we don't have the inner
            # summation
            # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
            # Apply the log operation
            result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

            # TODO: For now, implementing loss func over all time-steps
            # Sum up all log probabilities for each data point
            return tf.reduce_sum(result1)

        def get_coef(output):
            # eq 20 -> 22 of Graves (2013)
            # TODO : (resolve) Does Social LSTM paper do this as well?
            # the paper says otherwise but this is essential as we cannot
            # have negative standard deviation and correlation needs to be between
            # -1 and 1

            z = output
            # Split the output into 5 parts corresponding to means, std devs and corr
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z)

            # The output must be exponentiated for the std devs
            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            # Tanh applied to keep it in the range [-1, 1]
            z_corr = tf.tanh(z_corr)

            return [z_mux, z_muy, z_sx, z_sy, z_corr]

        # Extract the coef from the output of the linear layer
        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(output)
        # Store the output from the model
        self.output = output

        # Store the predicted outputs
        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr

        # Compute the loss function
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

        # Compute the cost
        self.cost = tf.div(lossfunc, (args.batch_size * args.seq_length))

        # Get trainable_variables
        tvars = tf.trainable_variables()

        # TODO: (resolve) We are clipping the gradients as is usually done in LSTM
        # implementations. Social LSTM paper doesn't mention about this at all
        # Calculate gradients of the cost w.r.t all the trainable variables
        self.gradients = tf.gradients(self.cost, tvars)
        # Clip the gradients if they are larger than the value given in args
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # NOTE: Using RMSprop as suggested by Social LSTM instead of Adam as Graves(2013) does
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # initialize the optimizer with teh given learning rate
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # Train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, traj, num=10):
        '''
        Given an initial trajectory (as a list of tuples of points), predict the future trajectory
        until a few timesteps
        Params:
        sess: Current session of Tensorflow
        traj: List of past trajectory points
        num: Number of time-steps into the future to be predicted
        '''
        def sample_gaussian_2d(mux, muy, sx, sy, rho):
            '''
            Function to sample a point from a given 2D normal distribution
            params:
            mux : mean of the distribution in x
            muy : mean of the distribution in y
            sx : std dev of the distribution in x
            sy : std dev of the distribution in y
            rho : Correlation factor of the distribution
            '''
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
            next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])
            # Append the new point to the trajectory
            ret = np.vstack((ret, [next_x, next_y]))

            # Set the current sampled position as the last observed position
            prev_data[0, 0, 0] = next_x
            prev_data[0, 0, 1] = next_y

        return ret
