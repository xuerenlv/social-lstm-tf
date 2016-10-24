import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from social_model import SocialModel
from social_utils import SocialDataLoader


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=10,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=50,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=2,
                        help='Grid size of the social grid')
    args = parser.parse_args()
    train(args)


def getSocialGrid(x, d, args):
    '''
    Function that computes the social grid for all peds in the current frame of dataset d
    params:
    x : numpy matrix of size numPeds x 3
    d : dataset index
    '''
    # pedsWithGrid is a dictionary mapping pedIDs to their social grids in the current frame
    pedsWithGrid = {}
    # List of pedestrians
    pedsList = x[:, 0].tolist()

    # bounds to calculate the normalized neighborhood size
    if d == 0:
        # ETH univ dataset
        width = 640.
        height = 480.
    else:
        # Other datasets
        width = 720.
        height = 576.

    # Bounds for the current dataset
    width_bound = args.neighborhood_size/width
    height_bound = args.neighborhood_size/height

    # For each pedestrian in the current frame
    for ped in pedsList:
        # Extract data of all the "other" pedestrians in the current frame
        pedsInFrameBut = x[x[:, 0] != ped, :]

        # Extract the (x, y) of the current pedestrian
        current_x = x[x[:, 0] == ped, 1]
        current_y = x[x[:, 0] == ped, 2]

        # Get lower and upper bounds for the grid
        width_low = current_x - width_bound/2.
        height_low = current_y - height_bound/2.
        width_high = current_x + width_bound/2.
        height_high = current_y + height_bound/2.

        # Get the pedestrians who are in the surrounding
        pedsInFrameSurr = pedsInFrameBut[pedsInFrameBut[:, 1] <= width_high, :]
        pedsInFrameSurr = pedsInFrameSurr[pedsInFrameSurr[:, 1] >= width_low, :]
        pedsInFrameSurr = pedsInFrameSurr[pedsInFrameSurr[:, 2] <= height_high, :]
        pedsInFrameSurr = pedsInFrameSurr[pedsInFrameSurr[:, 2] >= height_low, :]

        # Calculate the grid cells in which these pedestrians are present
        cell_x = np.floor(((pedsInFrameSurr[:, 1] - width_low)/(width_bound)) * args.grid_size)
        cell_y = np.floor(((pedsInFrameSurr[:, 2] - height_low)/(height_bound)) * args.grid_size)

        # initialize an empty grid
        grid = [[] for i in range(args.grid_size * args.grid_size)]
        for m in range(args.grid_size):
            for n in range(args.grid_size):
                # peds in (m, n) grid cell
                ind = np.all([cell_x == m, cell_y == n], axis=0)
                # Get the pedIDs of all these pedestrians
                pedsInMN = pedsInFrameSurr[ind, 0]
                # Store the list of pedIDs in the grid cell
                grid[m + n*args.grid_size] = map(int, pedsInMN.tolist())
        # Save the list of lists in the dictionary
        pedsWithGrid[ped] = grid

    return pedsWithGrid


def getSocialTensor(grid, states, args):
    '''
    Function to get social tensor from hidden states and the social grid
    params:
    grid : Social grid as a list of lists
    states : dictionary containing the hidden states of all peds
    '''
    # Initialise a tensor of size (mxmxD) with zeros
    tensor = np.zeros((args.grid_size, args.grid_size, args.rnn_size))

    # For each grid cell in the social grid
    for m in range(args.grid_size):
        for n in range(args.grid_size):
            # get the list of peds in the current cell
            listOfPeds = grid[m + n*args.grid_size]
            # Initialize the hiddenStateSum of size (1xD) with zeros
            hiddenStateSum = np.zeros((1, args.rnn_size))
            # For each ped in the grid cell
            for x in listOfPeds:
                # Sum up their hidden states (or outputs from the previous time-steps)
                hiddenStateSum += states[x]
            # Store the hiddenStateSum in the tensor
            tensor[m, n, :] = hiddenStateSum
    return tensor


def train(args):
    # Create a socialDataloader object to get batches of size batch_size with a sequence of frames of length seq_length
    data_loader = SocialDataLoader(args.batch_size, args.seq_length, forcePreProcess=True)

    print "Number of mini-batches per epoch is", data_loader.num_batches

    # Save the arguments in the social_config file
    with open(os.path.join('save', 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a SocialModel object with the arguments
    model = SocialModel(args)

    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Initialize all variables in the graph
        sess.run(tf.initialize_all_variables())
        # Initialize a saver that saves all the variables in the graph
        saver = tf.train.Saver(tf.all_variables())

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate value for this epoch
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # Reset the data pointers in the data_loader
            data_loader.reset_batch_pointer()

            # For each batch
            for b in range(data_loader.num_batches):
                # Tic
                start = time.time()

                # Get the source, target and dataset data for the next batch
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                x, y, d = data_loader.next_batch()

                # Variable to store the loss for this batch
                loss_batch = 0
                # Counter
                counter = 0

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch, d_batch = x[batch], y[batch], d[batch]

                    # Get the list of pedIDs in this sequence
                    pedIDs = np.unique(x_batch[:, :, 0])
                    # Remove all pedIDs with value of zero (non-existing peds)
                    pedIDs = pedIDs[pedIDs != 0]

                    # Create data structures to maintain all their hidden states
                    # states maintains their hidden states (or outputs) after each frame
                    # lstm_states maintains their cell states after each frame
                    # Both are dictionaries mapping pedIDs to corresponding states
                    states = {}
                    lstm_states = {}
                    # For each pedID in the current sequence of frames
                    for ped in pedIDs:
                        # Initialize a tensor of size 1 x D with zeros as the initial hidden state
                        states[ped] = np.zeros((1, args.rnn_size))
                        # Initialise the LSTM corresponding to each ped. Store the initial state in the data structure
                        lstm_states[ped] = sess.run(model.initial_state)

                    # For each frame in the sequence
                    for seq in range(data_loader.seq_length):
                        # Get the source, target data for all the pedestrians in the current frame
                        x_batch_seq, y_batch_seq,  d_batch_seq = x_batch[seq, :, :], y_batch[seq, :, :], d_batch

                        # Extract only the data of pedestrians in current frame (remove non-existing peds)
                        x_batch_seq = x_batch_seq[x_batch_seq[:, 0] != 0, :]
                        y_batch_seq = y_batch_seq[y_batch_seq[:, 0] != 0, :]

                        # Compute their social grids
                        grid_batch_seq = getSocialGrid(x_batch_seq, d_batch_seq, args)

                        # Get the list of peds (their pedIDs) in the current frame
                        peds_batch_seq = x_batch_seq[:, 0].tolist()

                        # For each ped in the current frame
                        for ped in peds_batch_seq:

                            # If the current ped is not present in the target data, continue
                            if np.all(y_batch_seq[:, 0] != ped):
                                continue

                            # Extract the (x, y) position of the current ped in the current frame
                            x_ped_batch_seq = x_batch_seq[x_batch_seq[:, 0] == ped, [1, 2]]
                            y_ped_batch_seq = y_batch_seq[y_batch_seq[:, 0] == ped, [1, 2]]
                            # Extract the grid of the current ped in the current frame
                            grid_ped_batch_seq = grid_batch_seq[ped]

                            # NOTE: need to add a non-linear ReLU layer before computing the tensor
                            # Compute the social tensor given his grid
                            social_tensor = getSocialTensor(grid_ped_batch_seq, states, args)

                            # reshape input data
                            x_ped_batch_seq = np.reshape(x_ped_batch_seq, (1, 2))
                            y_ped_batch_seq = np.reshape(y_ped_batch_seq, (1, 2))

                            # reshape tensor data
                            social_tensor = np.reshape(social_tensor, (1, args.grid_size*args.grid_size*args.rnn_size))

                            # Feed the source, target, the LSTM cell state and the social tensor to the model
                            feed = {model.input_data: x_ped_batch_seq, model.target_data: y_ped_batch_seq, model.initial_state: lstm_states[ped], model.social_tensor: social_tensor}
                            # Fetch the cost for this point, output of the LSTM, the final cell state and the train operator
                            train_loss, states[ped], lstm_states[ped], _ = sess.run([model.cost, model.output, model.final_state, model.train_op], feed)

                            # Increment the batch loss with the loss incurred
                            loss_batch += train_loss
                            # Increment the counter
                            counter += 1

                # Toc
                end = time.time()

                # Calculate the mean batch loss
                if counter != 0:
                    loss_batch = loss_batch / counter
                else:
                    # No ped existed for more than one frame
                    print "Never trained. Peds existed only for one frame"
                    loss_batch = 0

                # Print epoch, batch, loss and time taken
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        loss_batch, end - start))

                # Save the model if the current epoch and batch number match the frequency
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'social_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
