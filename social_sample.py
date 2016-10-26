import numpy as np
import tensorflow as tf

import os
import pickle
import argparse

from social_utils import SocialDataLoader
from social_model import SocialModel
from social_train import getSocialGrid, getSocialTensor


def get_mean_error(pred_x, true_y):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    pred_x : a numpy matrix of size numPedsx3 with columns being ped, x, y
    true_y : a numpy matrix of size numPedsDiffx3 with columns same as above
    NOTE: that the pedestrians present in pred_x *may* not be in true_y (or)
    the pedestrians in true_y *may* not be in pred_x. Discount these peds
    while calculating the mean error
    '''
    # Get the list of pedestrians for which predictions are made
    pred_peds_list = pred_x[:, 0].tolist()

    # Variable to maintain the error
    error = 0
    # Counter
    counter = 0
    # For each pedestrian for whom predictions are made
    for ped in pred_peds_list:
        # Predicted positions for the current pedestrian
        pred_pos = pred_x[pred_x[:, 0] == ped, :]
        # True positions for the current pedestrian
        true_pos = true_y[true_y[:, 0] == ped, :]

        # If there is no ped in true_pos
        if true_pos.size == 0:
            # Ped not present in true_y
            continue

        # Else ped is present in true_y
        # Compute error as euclidean norm
        error += np.linalg.norm(pred_pos[:, [1, 2]] - true_pos[:, [1, 2]])
        # Increment counter
        counter += 1

    # Return mean error
    if counter != 0:
        return error / counter
    else:
        return error


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    '''
    Function to sample from a 2D gaussian
    params:
    mux : Mean of the distribution in X
    muy : Mean of the distribution in Y
    sx : Std dev of the distribution in X
    sy : Std dev of the distribution in Y
    rho : Correlation factor of the distribution
    '''
    # Construct mean and covariance matrix of the 2D distribution
    mean = [mux, muy]
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]

    # Sample from the 2D distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def main():
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')

    # Parse the parameters
    sample_args = parser.parse_args()

    # Define the path for the config file for saved args
    with open(os.path.join('save', 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Create a SocialModel object with the saved_args and infer set to true
    model = SocialModel(saved_args, True)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize a saver
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state('save')
    print ('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Create a SocialDataLoader object with batch_size 1 and seq_length equal to observed_length + pred_length
    data_loader = SocialDataLoader(1, sample_args.pred_length + sample_args.obs_length)

    # Reset all pointers of the data_loader
    data_loader.reset_batch_pointer()

    # Variable to maintain total error
    total_error = 0
    # For each batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        lstm_state = sess.run(model.initial_state)
        x, y, d = data_loader.next_batch()

        # Batch size is 1
        x_batch, y_batch, d_batch = x[0], y[0], d[0]

        # Get list of pedIDs in the current sequence of frames
        pedIDs = np.unique(x_batch[:, :, 0])
        pedIDs = pedIDs[pedIDs != 0]

        # Initialize dictionaries to store hidden states and cell states of all the pedestrians
        states = {}
        lstm_states = {}

        # For each ped in the sequence of frames
        for ped in pedIDs:
            # Initialize the hidden state of the ped to zeros
            states[ped] = np.zeros((1, saved_args.rnn_size))
            # Initialize the cell state of the ped to initial LSTM cell state
            lstm_states[ped] = sess.run(model.initial_state)

        # For each observed frame in the sequence of frames
        for seq in range(sample_args.obs_length-1):
            # For each observed frame
            x_batch_seq, y_batch_seq,  d_batch_seq = x_batch[seq, :, :], y_batch[seq, :, :], d_batch

            # Extract only the data of pedestrians in current frame
            x_batch_seq = x_batch_seq[x_batch_seq[:, 0] != 0, :]
            y_batch_seq = y_batch_seq[y_batch_seq[:, 0] != 0, :]

            # Get social grids of all the peds
            grid_batch_seq = getSocialGrid(x_batch_seq, d_batch_seq, saved_args)

            # Get list of pedIDs of the peds in the current frame
            peds_batch_seq = x_batch_seq[:, 0].tolist()

            # For each pedestrian in the observed frame
            for ped in peds_batch_seq:

                # Check if the current ped is in the target data
                if np.all(y_batch_seq[:, 0] != ped):
                    # If not in the target data, then continue
                    continue

                # Get the source data and target data of the current ped
                x_ped_batch_seq = x_batch_seq[x_batch_seq[:, 0] == ped, [1, 2]]
                y_ped_batch_seq = y_batch_seq[y_batch_seq[:, 0] == ped, [1, 2]]
                # Get the social grid of the current ped
                grid_ped_batch_seq = grid_batch_seq[ped]

                # NOTE: need to add a non-linear ReLU layer before computing the tensor
                # Compute the social tensor of the ped given his grid and states of other peds
                social_tensor = getSocialTensor(grid_ped_batch_seq, states, saved_args)

                # reshape input data
                x_ped_batch_seq = np.reshape(x_ped_batch_seq, (1, 2))
                y_ped_batch_seq = np.reshape(y_ped_batch_seq, (1, 2))

                # reshape tensor data
                social_tensor = np.reshape(social_tensor, (1, saved_args.grid_size*saved_args.grid_size*saved_args.rnn_size))

                # Feed the source, initial LSTM cell state and the social tensor to the model
                feed = {model.input_data: x_ped_batch_seq, model.initial_state: lstm_states[ped], model.social_tensor: social_tensor}
                # feed = {model.input_data: x_ped_batch_seq, model.initial_state: lstm_state, model.social_tensor: social_tensor}

                # Fetch the output and the final LSTM cell state
                states[ped], lstm_states[ped] = sess.run([model.output, model.final_state], feed)
                # states[ped], lstm_state = sess.run([model.output, model.final_state], feed)

        # Store last positions of all the observed pedestrians
        last_obs_frame = sample_args.obs_length-1
        # Get the source, target and dataset data regarding the last observed frame
        x_batch_seq, y_batch_seq, d_batch_seq = x_batch[last_obs_frame, :, :], y_batch[last_obs_frame, :, :], d_batch

        # For each frame to be predicted
        for seq in range(sample_args.obs_length, sample_args.pred_length + sample_args.obs_length):
            # Extract only the data of pedestrians in current frame
            x_batch_seq = x_batch_seq[x_batch_seq[:, 0] != 0, :]
            y_batch_seq = y_batch_seq[y_batch_seq[:, 0] != 0, :]

            # Compute their social grids
            grid_batch_seq = getSocialGrid(x_batch_seq, d_batch_seq, saved_args)

            # Get the list of pedIDs of the peds in the current frame
            peds_batch_seq = x_batch_seq[:, 0].tolist()

            # Create a copy of the data for the next frame
            x_batch_next_seq = np.copy(x_batch_seq)

            # For each pedestrian to be predicted
            for ped in peds_batch_seq:

                # Get data regarding the current ped
                x_ped_batch_seq = x_batch_seq[x_batch_seq[:, 0] == ped, [1, 2]]
                # Get his social grid
                grid_ped_batch_seq = grid_batch_seq[ped]

                # NOTE: need to add a non-linear ReLU layer before computing the tensor
                # Compute his social tensor
                social_tensor = getSocialTensor(grid_ped_batch_seq, states, saved_args)

                # reshape input data
                x_ped_batch_seq = np.reshape(x_ped_batch_seq, (1, 2))
                y_ped_batch_seq = np.reshape(y_ped_batch_seq, (1, 2))

                # reshape tensor data
                social_tensor = np.reshape(social_tensor, (1, saved_args.grid_size*saved_args.grid_size*saved_args.rnn_size))

                # Feed the model the source data, initial LSTM cell state and the social tensor
                feed = {model.input_data: x_ped_batch_seq, model.initial_state: lstm_states[ped], model.social_tensor: social_tensor}
                # feed = {model.input_data: x_ped_batch_seq, model.initial_state: lstm_state, model.social_tensor: social_tensor}

                # Fetch the output, final LSTM state, (mu, sigma, corr)
                states[ped], lstm_states[ped], o_mux, o_muy, o_sx, o_sy, o_corr = sess.run([model.output, model.final_state, model.mux, model.muy, model.sx, model.sy, model.corr], feed)
                # states[ped], lstm_state, o_mux, o_muy, o_sx, o_sy, o_corr = sess.run([model.output, model.final_state, model.mux, model.muy, model.sx, model.sy, model.corr], feed)

                # Sample the next position for the current ped given the parameters of the 2D distribution
                next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])

                # Replace the position with the predicted position for the current ped
                x_batch_next_seq[x_batch_next_seq[:, 0] == ped, 1] = next_x
                x_batch_next_seq[x_batch_next_seq[:, 0] == ped, 2] = next_y

            # Compute the mean error between the predictions and the target data
            total_error += get_mean_error(x_batch_next_seq, y_batch_seq)
            # Compute the source, target data for the next frame
            x_batch_seq = x_batch_next_seq
            y_batch_seq = y_batch[seq, :, :]

        print "Processed batch number : ", b, " of the dataset ", d_batch

    # Compute the mean error over all batches
    print "Total mean error of the model is ", total_error/data_loader.num_batches

if __name__ == '__main__':
    main()
