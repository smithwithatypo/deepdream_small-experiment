import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helper_functions as hf

import IPython.display as display
import PIL.Image

import os
import pathlib
import itertools
import random

# make a list of all folders in small_data_original/
# make a list of all files in each {folder}/images/
# augment one file by running tutorial code
# export augmented_image to data/small_data_augmented/{folder}/{file}
