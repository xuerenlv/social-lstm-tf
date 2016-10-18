import os
import pickle
import numpy as np
import random

# Debugging
import ipdb


class SocialDataLoader():

    def __init__(self, batch_size=50, seq_length=5, neighborhood_size=100, grid_size=2, forcePreProcess=False):
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

        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.data_dirs, data_file)

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
        dataset_index = 0
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
                traj = traj.T.tolist()

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
                    pedsInFrameSurr = pedsInFrame[:, pedsInFrame[3, :] <= width_high]
                    pedsInFrameSurr = pedsInFrameSurr[:, pedsInFrameSurr[3, :] >= width_low]
                    pedsInFrameSurr = pedsInFrameSurr[:, pedsInFrameSurr[2, :] <= height_high]
                    pedsInFrameSurr = pedsInFrameSurr[:, pedsInFrameSurr[2, :] >= height_low]

                    # Surrounding peds inside the social grid are now in pedsInFrameSurr
                    # Discretize the grid and get occupancy
                    cell_x = np.floor(((pedsInFrameSurr[3, :] - width_low)/(width_bound)) * self.grid_size)
                    cell_y = np.floor(((pedsInFrameSurr[2, :] - height_low)/(height_bound)) * self.grid_size)
                    
                    social_grid = [[] for i in range(self.grid_size*self.grid_size)]
                    for m in range(self.grid_size):
                        for n in range(self.grid_size):
                            # peds in (m,n) grid cell
                            ind = np.all([cell_x == m, cell_y == n], axis=0)
                            pedsInMN = pedsInFrameSurr[1, ind] + current_ped
                            social_grid[m + n*self.grid_size] = map(int, pedsInMN.tolist())
                    point.append(social_grid)
                all_ped_data[current_ped + ped] = traj

            dataset_indices.append(current_ped + numPeds)
            current_ped += numPeds
            dataset_index += 1

        complete_data = (all_ped_data, dataset_indices)
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()

    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        # all_frame_data would be a list of dictionaries with mapping from each frame to
        # a list of all peds with their current location and their social grids,
        # for each dataset

        all_frame_data = []
        frameList_data = []
        dataset_index = 0

        for directory in data_dirs:

            all_frame_data.append({})

            file_path = os.path.join(directory, 'pixel_pos.csv')

            data = np.genfromtxt(file_path, delimiter=',')

            numFrames = np.size(np.unique(data[0, :]))
            frameList = np.unique(data[0, :]).tolist()

            frameList_data.append(frameList)

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

            for frame in frameList:

                # Extract all pedestrians in current frame
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist()

                pedsWithGrid = []

                for ped in pedsList:
                    # For each ped. Exclude the ped from others
                    pedsInFrameBut = pedsInFrame[:, pedsInFrame[1, :] != ped]
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped]

                    # Get lower and upper bounds for the grid
                    width_low = current_x - width_bound/2.
                    height_low = current_y - height_bound/2.
                    width_high = current_x + width_bound/2.
                    height_high = current_y + height_bound/2.

                    # Get the pedestrians who are in the surrounding
                    pedsInFrameSurr = pedsInFrameBut[:, pedsInFrameBut[3, :] <= width_high]
                    pedsInFrameSurr = pedsInFrameSurr[:, pedsInFrameSurr[3, :] >= width_low]
                    pedsInFrameSurr = pedsInFrameSurr[:, pedsInFrameSurr[2, :] <= height_high]
                    pedsInFrameSurr = pedsInFrameSurr[:, pedsInFrameSurr[2, :] >= height_low]

                    cell_x = np.floor(((pedsInFrameSurr[3, :] - width_low)/(width_bound)) * self.grid_size)
                    cell_y = np.floor(((pedsInFrameSurr[2, :] - height_low)/(height_bound)) * self.grid_size)

                    social_grid = [[] for i in range(self.grid_size*self.grid_size)]
                    for m in range(self.grid_size):
                        for n in range(self.grid_size):
                            # peds in (m,n) grid cell
                            ind = np.all([cell_x == m, cell_y == n], axis=0)
                            pedsInMN = pedsInFrameSurr[1, ind]
                            social_grid[m + n*self.grid_size] = map(int, pedsInMN.tolist())
                    pedsWithGrid.append([ped, social_grid])

                all_frame_data[dataset_index][frame] = pedsWithGrid
            dataset_index += 1

        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        counter = 0

        for dataset in range(len(self.raw_data)):
            # get the frame data for the current dataset
            all_frame_data = self.raw_data[dataset]
            counter += int(len(all_frame_data) / (self.seq_length+2))

        self.num_batches = int(counter/self.batch_size)

    def next_batch(self):
        '''
        Function to get the next batch of data
        '''
        x_batch = []
        y_batch = []

        frame_data = self.data[self.dataset_pointer]
        frames = self.frameList[self.dataset_pointer]

        for i in range(self.batch_size):
            current_x_seq = []
            current_y_seq = []
            idx = self.frame_pointer
            if idx + self.seq_length + 1 < len(frame_data):
                for offset in range(self.seq_length+1):
                    current_x_seq.append(frame_data[frames[idx+offset]])
                    current_y_seq.append(frame_data[frames[idx+offset+1]])
                self.frame_pointer += self.seq_length + 1
            else:
                # Not enough frames left
                self.tick_batch_pointer()

            x_batch.append(current_x_seq)
            y_batch.append(current_y_seq)

        return x_batch, y_batch

    def tick_batch_pointer(self):
        '''
        Advance the dataset pointer
        '''
        self.dataset_pointer += 1
        self.frame_pointer = 0
        if self.dataset_pointer >= len(self.data):
            self.dataset_pointer = 0

    def reset_batch_pointer(self):
        self.dataset_pointer = 0
        self.frame_pointer = 0
