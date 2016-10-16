import numpy as np
import tensorflow as tf

import time
import os
import pickle
import argparse

from utils import DataLoader
from model import Model
import random


# main code (not in a main function, so that we can run this script in ipython as well)

parser = argparse.ArgumentParser()
parser.add_argument('--obs_length', type=int, default=5,
                    help='Observed length of the trajectory')
parser.add_argument('--pred_length', type=int, default=3,
                    help='Predicted length of the trajectory')

sample_args = parser.parse_args()

with open(os.path.join('save', 'config.pkl'))
