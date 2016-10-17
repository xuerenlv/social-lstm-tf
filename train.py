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
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
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
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding dimension for the spatial coordinates')
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = DataLoader(args.batch_size, args.seq_length)

    with open(os.path.join('save', 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)

            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()

                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                train_loss, state, _, lr = sess.run([model.cost, model.final_state, model.train_op, model.lr], feed)
                # print result
                # print output_b
                # print gradients
                # print input_data
                # print lr
                end = time.time()
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        train_loss, end - start))

                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
