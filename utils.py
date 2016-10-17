import os
import pickle
import numpy as np
import random


# TODO : (improve) Add functionality to retrieve data only from specific datasets
class DataLoader():

    def __init__(self, batch_size=50, seq_length=5):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : RNN sequence length
        '''
        # List of data directories where raw data resides
        self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                          './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                          './data/ucy/univ']

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        self.batch_size = batch_size
        self.seq_length = seq_length

        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        if not(os.path.exists(data_file)):
            print("Creating pre-processed data from raw data")
            self.preprocess(self.data_dirs, data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        '''
        The function that pre-processes the pixel_pos.csv files of each dataset
        into data that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_ped_data would be a dictionary with mapping from each ped to their
        # trajectories given by matrix 3 x numPoints with each column
        # in the order x, y, frameId
        # Pedestrians from all datasets are combined
        # Dataset pedestrian indices are stored in dataset_indices
        all_ped_data = {}
        dataset_indices = []
        current_ped = 0
        for directory in data_dirs:
            file_path = os.path.join(directory, 'pixel_pos.csv')

            # Data is a 4 x numTrajPoints matrix
            # where each column is a (frameId, pedId, y, x) vector
            data = np.genfromtxt(file_path, delimiter=',')

            # Get the number of pedestrians in the current dataset
            numPeds = np.size(np.unique(data[1, :]))

            for ped in range(1, numPeds+1):
                # Extract trajectory of the current ped
                traj = data[:, data[1, :] == ped]
                # Format it as (x, y, frameId)
                traj = traj[[3, 2, 0], :]

                # Store this in the dictionary
                all_ped_data[current_ped + ped] = traj

            # Current dataset done
            dataset_indices.append(numPeds)
            current_ped = numPeds

        complete_data = (all_ped_data, dataset_indices)
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : The path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        all_ped_data = self.raw_data[0]
        # Not using dataset_indices for now
        # dataset_indices = self.raw_data[1]

        self.data = []
        counter = 0

        for ped in all_ped_data:
            traj = all_ped_data[ped]
            if traj.shape[1] > (self.seq_length+2):
                # TODO: (Improve) Store only the (x,y) coordinates for now
                self.data.append(traj[[0, 1], :].T)
                # Number of batches this datapoint is worth
                counter += int(traj.shape[1] / ((self.seq_length+2)))

        self.num_batches = int(counter / self.batch_size)

    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            traj = self.data[self.pointer]
            n_batch = int(traj.shape[0] / (self.seq_length+2))
            idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
            x_batch.append(np.copy(traj[idx:idx+self.seq_length, :]))
            y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :]))

            if random.random() < (1.0/float(n_batch)):
                # Adjust sampling probability
                # if this is a long datapoint, sample this data more with
                # higher probability
                self.tick_batch_pointer()

        return x_batch, y_batch

    def tick_batch_pointer(self):
        '''
        Advance the data pointer
        '''
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        self.pointer = 0
