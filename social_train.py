import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from social_model import SocialModel
from social_utils import SocialDataLoader

import ipdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=10,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # NOTE: Added a new argument that represents the embeding size
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--neighborhood_size', type=int, default=50,
                        help='Neighborhood size to be considered for social grid')
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
    
    pedsWithGrid = {}
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
    
    width_bound = args.neighborhood_size/width
    height_bound = args.neighborhood_size/height

    for ped in pedsList:
        pedsInFrameBut = x[x[:, 0]!=ped, :]
        current_x = x[x[:, 0]==ped, 1]
        current_y = x[x[:, 0]==ped, 2]

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
    
        cell_x = np.floor(((pedsInFrameSurr[:, 1] - width_low)/(width_bound)) * args.grid_size)
        cell_y = np.floor(((pedsInFrameSurr[:, 2] - height_low)/(height_bound)) * args.grid_size)

        grid = [[] for i in range(args.grid_size * args.grid_size)]
        for m in range(args.grid_size):
            for n in range(args.grid_size):
                # peds in (m, n) grid cell
                ind = np.all([cell_x == m, cell_y == n], axis=0)
                pedsInMN = pedsInFrameSurr[ind, 0]
                grid[m + n*args.grid_size] = map(int, pedsInMN.tolist())
        pedsWithGrid[ped] = grid
        
    return pedsWithGrid

def getSocialTensor(grid, states, args):
    '''
    Function to get social tensor from hidden states and the social grid
    params:
    grid : Social grid as a list of lists
    states : dictionary containing the hidden states of all peds
    '''
    tensor = np.zeros((args.grid_size, args.grid_size, args.rnn_size))
    
    for m in range(args.grid_size):
        for n in range(args.grid_size):
            listOfPeds = grid[m + n*args.grid_size]
            hiddenStateSum = np.zeros((1, args.rnn_size))
            for x in listOfPeds:
                hiddenStateSum += states[x]
            tensor[m, n, :] = hiddenStateSum
    return tensor

def train(args):
    data_loader = SocialDataLoader(args.batch_size, args.seq_length, args.neighborhood_size, args.grid_size, forcePreProcess=True)

    print "Number of mini-batches per epoch is", data_loader.num_batches

    with open(os.path.join('save', 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = SocialModel(args)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            # lstm_state = sess.run(model.initial_state)

            for b in range(data_loader.num_batches):
                start = time.time()
                x, y, d = data_loader.next_batch()
                
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # xp, yp are lists containing lists of length seq_length with each element containing the number of peds in that frame
                # d is the dataset index from which this batch is generated (used to differentiate between datasets)
                
                loss_batch = 0
                counter = 0
                
                for batch in range(data_loader.batch_size):
                    # In each batch of seq_length frames
                    x_batch, y_batch, d_batch = x[batch], y[batch], d[batch]

                    pedIDs = np.unique(x_batch[:, :, 0])
                    pedIDs = pedIDs[pedIDs != 0]  # Get all the unique pedIDs in the current batch

                    # Create data structures to maintain all their hidden states
                    states = {}
                    lstm_states = {}
                    for ped in pedIDs:
                        # Initialise the LSTM corresponding to each ped. Store the initial state in the data structure
                        states[ped] = np.zeros((1, args.rnn_size))
                        lstm_states[ped] = sess.run(model.initial_state)
                        
                    for seq in range(data_loader.seq_length):
                        # In each frame
                        x_batch_seq, y_batch_seq,  d_batch_seq = x_batch[seq, :, :], y_batch[seq, :, :], d_batch

                        # Extract only the data of pedestrians in current frame
                        x_batch_seq = x_batch_seq[x_batch_seq[:, 0]!=0, :]
                        y_batch_seq = y_batch_seq[y_batch_seq[:, 0]!=0, :]

                        
                        grid_batch_seq = getSocialGrid(x_batch_seq, d_batch_seq, args)

                        peds_batch_seq = x_batch_seq[:, 0].tolist()

                        for ped in peds_batch_seq:
                            
                            if np.all(y_batch_seq[:, 0]!=ped):
                                continue
                            
                            x_ped_batch_seq = x_batch_seq[x_batch_seq[:,0]==ped, [1, 2]]
                            y_ped_batch_seq = y_batch_seq[y_batch_seq[:,0]==ped, [1, 2]]
                            grid_ped_batch_seq = grid_batch_seq[ped]

                            # NOTE: need to add a non-linear ReLU layer before computing the tensor
                            social_tensor = getSocialTensor(grid_ped_batch_seq, states, args)                                                    

                            # reshape input data
                            x_ped_batch_seq = np.reshape(x_ped_batch_seq, (1, 2))
                            y_ped_batch_seq = np.reshape(y_ped_batch_seq, (1, 2))

                            # reshape tensor data
                            social_tensor = np.reshape(social_tensor, (1, args.grid_size*args.grid_size*args.rnn_size))
                            
                            feed = {model.input_data: x_ped_batch_seq, model.target_data: y_ped_batch_seq, model.initial_state: lstm_states[ped], model.social_tensor: social_tensor}
                            train_loss, states[ped], lstm_states[ped], _ = sess.run([model.cost, model.output, model.final_state, model.train_op], feed)
                            
                            loss_batch += train_loss
                            counter += 1
                            
                end = time.time()
                
                # loss_batch = loss_batch/(data_loader.batch_size * data_loader.seq_length)
                if counter != 0:
                    loss_batch = loss_batch / counter
                else:
                    print "Never trained. Peds existed only for one frame"
                    loss_batch = 0

                
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        loss_batch, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'social_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__== '__main__':
    main()
