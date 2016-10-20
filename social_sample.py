import numpy as np
import tensorflow as tf

import time
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
    pred_peds_list = pred_x[:, 0].tolist()

    error = 0
    counter = 0
    for ped in pred_peds_list:
        pred_pos = pred_x[pred_x[:, 0] == ped, :]
        true_pos = true_y[true_y[:, 0] == ped, :]
        if true_pos.size == 0:
            # Ped not present in true_y
            continue

        # Else ped is present in true_y
        error += np.linalg.norm(pred_pos[:, [1, 2]] - true_pos[:, [1, 2]])
        counter += 1

    if counter != 0:
        return error / counter
    else:
        return error

def sample_gaussian_2d(mux, muy, sx, sy, rho):
    '''
    Function to sample from a 2D gaussian
    '''
    mean = [mux, muy]
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]

    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')

    sample_args = parser.parse_args()

    with open(os.path.join('save', 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)


    model = SocialModel(saved_args, True)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state('save')
    print ('loading model: ', ckpt.model_checkpoint_path)

    saver.restore(sess, ckpt.model_checkpoint_path)

    data_loader = SocialDataLoader(1, sample_args.pred_length + sample_args.obs_length)

    data_loader.reset_batch_pointer()

    total_error = 0
    for b in range(data_loader.num_batches):
        # For each batch
        start = time.time()
        x, y, d = data_loader.next_batch()
        
        # Batch size is 1
        x_batch, y_batch, d_batch = x[0], y[0], d[0]

        pedIDs = np.unique(x_batch[:, :, 0])
        pedIDs = pedIDs[pedIDs != 0]

        states = {}
        lstm_states = {}
        
        for ped in pedIDs:
            states[ped] = np.zeros((1, saved_args.rnn_size))
            lstm_states[ped] = sess.run(model.initial_state)

        for seq in range(sample_args.obs_length-1):
            # For each observed frame
            x_batch_seq, y_batch_seq,  d_batch_seq = x_batch[seq, :, :], y_batch[seq, :, :], d_batch

            # Extract only the data of pedestrians in current frame
            x_batch_seq = x_batch_seq[x_batch_seq[:, 0] != 0, :]
            y_batch_seq = y_batch_seq[y_batch_seq[:, 0] != 0, :]
            
            grid_batch_seq = getSocialGrid(x_batch_seq, d_batch_seq, saved_args)
            
            peds_batch_seq = x_batch_seq[:, 0].tolist()
            
            for ped in peds_batch_seq:
                # For each pedestrian in the observed frame
                if np.all(y_batch_seq[:, 0]!=ped):
                    continue
                
                x_ped_batch_seq = x_batch_seq[x_batch_seq[:,0]==ped, [1, 2]]
                y_ped_batch_seq = y_batch_seq[y_batch_seq[:,0]==ped, [1, 2]]
                grid_ped_batch_seq = grid_batch_seq[ped]
                
                # NOTE: need to add a non-linear ReLU layer before computing the tensor
                social_tensor = getSocialTensor(grid_ped_batch_seq, states, saved_args)                                                    
                
                # reshape input data
                x_ped_batch_seq = np.reshape(x_ped_batch_seq, (1, 2))
                y_ped_batch_seq = np.reshape(y_ped_batch_seq, (1, 2))
                
                # reshape tensor data
                social_tensor = np.reshape(social_tensor, (1, saved_args.grid_size*saved_args.grid_size*saved_args.rnn_size))

                feed = {model.input_data: x_ped_batch_seq, model.initial_state: lstm_states[ped], model.social_tensor: social_tensor}

                states[ped], lstm_states[ped] = sess.run([model.output, model.final_state], feed)

        # Store last positions of all the observed pedestrians
        last_obs_frame = sample_args.obs_length-1
        x_batch_seq, y_batch_seq, d_batch_seq = x_batch[last_obs_frame, :, :], y_batch[last_obs_frame, :, :], d_batch
    
        for seq in range(sample_args.obs_length, sample_args.pred_length + sample_args.obs_length):
            # For each frame to be predicted
            # Extract only the data of pedestrians in current frame
            x_batch_seq = x_batch_seq[x_batch_seq[:, 0] != 0, :]
            y_batch_seq = y_batch_seq[y_batch_seq[:, 0] != 0, :]

            grid_batch_seq = getSocialGrid(x_batch_seq, d_batch_seq, saved_args)

            peds_batch_seq = x_batch_seq[:, 0].tolist()

            x_batch_next_seq = np.copy(x_batch_seq)

            for ped in peds_batch_seq:
                # For each pedestrian to be predicted

                x_ped_batch_seq = x_batch_seq[x_batch_seq[:, 0] == ped, [1, 2]]
                grid_ped_batch_seq = grid_batch_seq[ped]

                # NOTE: need to add a non-linear ReLU layer before computing the tensor
                social_tensor = getSocialTensor(grid_ped_batch_seq, states, saved_args)                                                    
            
                # reshape input data
                x_ped_batch_seq = np.reshape(x_ped_batch_seq, (1, 2))
                y_ped_batch_seq = np.reshape(y_ped_batch_seq, (1, 2))
            
                # reshape tensor data
                social_tensor = np.reshape(social_tensor, (1, saved_args.grid_size*saved_args.grid_size*saved_args.rnn_size))

                feed = {model.input_data: x_ped_batch_seq, model.initial_state: lstm_states[ped], model.social_tensor: social_tensor}
                states[ped], lstm_states[ped], o_mux, o_muy, o_sx, o_sy, o_corr = sess.run([model.output, model.final_state, model.mux, model.muy, model.sx, model.sy, model.corr], feed)

                next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])

                x_batch_next_seq[x_batch_next_seq[:, 0]==ped, 1] = next_x
                x_batch_next_seq[x_batch_next_seq[:, 0]==ped, 2] = next_y

            total_error += get_mean_error(x_batch_next_seq, y_batch_seq)
            x_batch_seq = x_batch_next_seq
            y_batch_seq = y_batch[seq, :, :]

        print "Processed batch number : ", b, " of the dataset ", d_batch
            
    print "Total mean error of the model is ", total_error/data_loader.num_batches

if __name__ == '__main__':
    main()
