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

filepath = f"./data/small_data_original/"
directories = hf.make_list_of_directories_from_filepath(filepath)

for folder in directories:
    filepath = f"./data/small_data_original/{folder}/images/"
    files = hf.make_list_of_files_from_filepath(filepath)
    for file in files:
        original_img = hf.read_image_from_local_storage(
            file, folder=folder, route=None, small_data=True)
        print(f"Original Image:", type(original_img), original_img.shape)
        hf.show(original_img)

        '''experimental below this line'''
        new_image = original_img.copy()
        print(f"New Image:", type(new_image), new_image.shape)
        hf.show(new_image)

        hf.export_image_to_local_storage(
            new_image, folder=folder, file=file)

        break

    break
