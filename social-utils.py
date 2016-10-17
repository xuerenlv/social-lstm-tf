import os
import pickle
import numpy as np
import random


class SocialDataLoader():

    def __init__(self, batch_size=50, seq_length=5, neighborhood_size=32, grid_size=8):
        '''
        Initialiser function for the SocialDataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : RNN sequence length
        neighborhood_size : Size of neighborhood to be considered
        grid_size : Size of the social grid constructed
        '''
        # List of data directories where raw data resides
        self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                          './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                          './data/ucy/univ']

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.neighborhood_size = neighborhood_size
        self.grid_size = grid_size

        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        if not(os.path.exists(data_file)):
            print("Creating pre-processed data from raw data")
            self.preprocess(self.data_dirs, data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_ped_data would be a dictionary with mapping from each ped to their
        # social trajectories given by list of lists where each element of the
        # to list is fId, x, y, list of peds in (1,1) grid cell, ....., list of peds
        # in (m, m) grid cell. So each element is a list of size m^2 + 3

        all_ped_data = {}
        dataset_indices = []
        current_ped = 0
        dataset_index = 0  # TODO : (add) increment dataset_index at the end of the loop
        for directory in data_dirs:
            file_path = os.path.join(directory, 'pixel_pos.csv')

            # Data is a 4 x numTrajPoints matrix
            # where each column is a (frameId, pedId, y, x) vector
            data = np.genfromtxt(file_path, delimiter=',')

            # Get the number of pedestrians in the current dataset
            numPeds = np.size(np.unique(data[1, :]))

            # bounds to calculate the normalized neighborhood size
            if dataset_index == 0:
                # ETH univ dataset
                width = 640.
                height = 480.
            else:
                # Other datasets
                width = 720.
                height = 576.
                
            width_bound = self.neighborhood_size/width
            height_bound = self.neighborhood_size/height

            for ped in range(1, numPeds+1):
                # Extract trajectory of the current ped
                traj = data[:, data[1, :] == ped]
                # Format it as (frameId, x, y)
                traj = traj[[0, 3, 2], :]
                # Convert the numpy matrix to list
                traj = traj.tolist()

                for point in traj:
                    # For each point in the trajectory
                    # Extract pedestrians in the same frame
                    current_frame = point[0]
                    current_x = point[1]
                    current_y = point[2]

                    # Get lower and upper bounds for the grid
                    width_low = current_x - width_bound/2.
                    height_low = current_y - height_bound/2.
                    width_high = current_x + width_bound/2.
                    height_high = current_y + height_bound/2.

                    # pedsInFrame is a matrix with (fId, pedId, y, x) of
                    # all peds in the current frame i.e. fId = current_frame
                    # Remove the current ped
                    pedsInFrame = data[:, data[0, :] == current_frame]
                    pedsInFrame = pedsInFrame[:, pedsInFrame[1, :] != ped]

                    # Get the pedestrians who are in the surrounding
                    pedsInFrame = pedsInFrame[:, pedsInFrame[3, :] <= width_high]
                    pedsInFrame = pedsInFrame[:, pedsInFrame[3, :] >= width_low]
                    pedsInFrame = pedsInFrame[:, pedsInFrame[2, :] <= height_high]
                    pedsInFrame = pedsInFrame[:, pedsInFrame[2, :] >= height_low]

                    # Surrounding peds inside the social grid are now in pedsInFrame
                    # Discretize the grid and get occupancy
                    cell_x = np.floor(((pedsInFrame[3, :] - width_low)/(width_bound)) * self.grid_size)
                    cell_y = np.floor(((pedsInFrame[2, :] - height_low)/(height_bound)) * self.grid_size)

                    social_grid = [[] for i in range(self.grid_size*self.grid_size)]
                    for m in range(self.grid_size):
                        for n in range(self.grid_size):
                            # peds in (m,n) grid cell
                            pedsInMN = pedsInFrame[1, cell_x == m]
