import numpy as np
import tensorflow as tf

import time
import os
import pickle
import argparse

from utils import DataLoader
from model import Model


def get_mean_error(predicted_traj, true_traj, observed_length):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    error = np.zeros(len(true_traj) - observed_length)
    for i in range(observed_length, len(true_traj)):
        pred_pos = predicted_traj[i, :]
        true_pos = true_traj[i, :]

        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    return np.mean(error)

# main code (not in a main function, so that we can run this script in ipython as well)

parser = argparse.ArgumentParser()
parser.add_argument('--obs_length', type=int, default=5,
                    help='Observed length of the trajectory')
parser.add_argument('--pred_length', type=int, default=3,
                    help='Predicted length of the trajectory')

sample_args = parser.parse_args()

with open(os.path.join('save', 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state('save')
print('loading model: ', ckpt.model_checkpoint_path)

saver.restore(sess, ckpt.model_checkpoint_path)

# Get sequences of length obs_length+pred_length
data_loader = DataLoader(1, sample_args.pred_length + sample_args.obs_length)

data_loader.reset_batch_pointer()

total_error = 0.
for b in range(data_loader.num_batches):
    start = time.time()
    x, y = data_loader.next_batch()

    obs_traj = x[0][:sample_args.obs_length]
    complete_traj = model.sample(sess, obs_traj, num=sample_args.pred_length)

    total_error += get_mean_error(complete_traj, x[0], sample_args.obs_length)
    print "Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories"

print "Total mean error of the model is ", total_error/data_loader.num_batches
