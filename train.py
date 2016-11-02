import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from model import Model
from utils import DataLoader


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Number of layers parameter
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Type of recurrent unit parameter
    # Model currently not used. Only LSTM implemented
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=50,
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
    # Gradient value at which it should be clipped
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout probability parameter
    # Dropout not implemented.
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    args = parser.parse_args()
    train(args)


def train(args):
    datasets = range(2)
    # Remove the leaveDataset from datasets
    datasets.remove(args.leaveDataset)

    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    data_loader = DataLoader(args.batch_size, args.seq_length, datasets, forcePreProcess=True)

    # Save the arguments int the config file
    with open(os.path.join('save', 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a Vanilla LSTM model with the arguments
    model = Model(args)

    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Initialize all the variables in the graph
        sess.run(tf.initialize_all_variables())
        # Add all the variables to the list of variables to be saved
        saver = tf.train.Saver(tf.all_variables())

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate (decayed acc. to the epoch number)
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # Reset the pointers in the data loader object
            data_loader.reset_batch_pointer()
            # Get the initial cell state of the LSTM
            state = sess.run(model.initial_state)

            # For each batch in this epoch
            for b in range(data_loader.num_batches):
                # Tic
                start = time.time()
                # Get the source and target data of the current batch
                # x has the source data, y has the target data
                x, y = data_loader.next_batch()

                # Feed the source, target data and the initial LSTM state to the model
                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                # Fetch the loss of the model on this batch, the final LSTM state from the session
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                # Toc
                end = time.time()
                # Print epoch, batch, loss and time taken
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        train_loss, end - start))

                # Save the model if the current epoch and batch number match the frequency
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
