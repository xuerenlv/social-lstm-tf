'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 17th October 2016
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell


# The social LSTM Model
class SocialModel():

    def __init__(self, args, infer=False):
        '''
        Initialisation function for the class SocialModel
        params:
        args : Contains arguments required for the Model creation
        '''

        # If sampling new trajectories, then infer mode
        if infer:
            # Sample one position at a time
            args.batch_size = 1
            args.seq_length = 1

        # Store the arguments
        self.args = args
        self.infer = infer

        # Construct the basicLSTMCell recurrent unit with a dimension given by args.rnn_size
        cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)

        # Construct multiple layers of the recurrent unit given by cell
        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        # placeholders for the input data and the target data
        # The input data is going to be (x, y) position of a single pedestrian at the current time-step
        self.input_data = tf.placeholder(tf.float32, [1, 2])
        # The target data is going to be (x, y) position of a single pedestrian at the next time-step
        self.target_data = tf.placeholder(tf.float32, [1, 2])

        # Variable to hold the value of the learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Output dimension of the model
        self.output_size = 5  # 2 mu, 2 sigma and 1 corr

        # Initial cell state of the LSTM. Since we are passing one pedestrian at a time, the batch_size is 1
        self.initial_state = cell.zero_state(batch_size=1, dtype=tf.float32)

        # Placeholder for the input social tensor. This is going to be a tensor of size 1 x (m*m*D)
        # m : grid size and D : hidden state dimension
        self.social_tensor = tf.placeholder(tf.float32, [1, args.grid_size*args.grid_size*args.rnn_size])

        # Define variables for the spatial coordinates embedding layer
        with tf.variable_scope("coordinate_embedding"):
            embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_b = tf.get_variable("embedding_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        # Define variables for the social tensor embedding layer
        with tf.variable_scope("tensor_embedding"):
            embedding_t_w = tf.get_variable("embedding_t_w", [args.grid_size*args.grid_size*args.rnn_size, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.1))

        # Embed the spatial coordinates (x, y) of the current pedestrian
        embedded_input = tf.nn.relu(tf.add(tf.matmul(self.input_data, embedding_w), embedding_b))
        # Embed the social tensor of the current pedestrian
        embedded_tensor = tf.nn.relu(tf.add(tf.matmul(self.social_tensor, embedding_t_w), embedding_t_b))

        # Concatenate the embeddings of the spatial coordinates and the social tensor
        complete_input = tf.concat(1, [embedded_input, embedded_tensor])
        # Convert the input to a list of input (rnn requires inputs as a list of tensors)
        complete_input = tf.split(1, 1, complete_input)

        # Execute one step of the RNN with cell as the recurrent unit and the initial state as defined above
        output, last_state = tf.nn.rnn(cell, complete_input, initial_state=self.initial_state)
        # output, last_state = tf.nn.seq2seq.rnn_decoder(complete_input, self.initial_state, cell)

        # Reshape the output to be of size ?xargs.rnn_size
        output = tf.reshape(tf.concat(1, output), [-1, args.rnn_size])

        # NOTE: You can add a nonlinear ReLU layer here to match the implementation from the paper
        # Save this output (as this will be used to construct social tensor for other pedestrians in the next time-step)
        self.output = output

        # Apply the linear layer
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        # Store the final LSTM cell state
        self.final_state = last_state

        # Extract the x coordinates and the y coordinates of the target data
        [x_data, y_data] = tf.split(1, 2, self.target_data)

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
            # Calculate the PDF of the data w.r.t to the distribution
            result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

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

        # Store the predicted outputs
        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr

        # Compute the loss function
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

        # The cost is just the loss as only one size pedestrian is considered
        self.cost = lossfunc

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
        # initialize the optimizer with the given learning rate
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # Train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
