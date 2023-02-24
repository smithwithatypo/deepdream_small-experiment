import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import stephens_helper_functions as hf


import IPython.display as display
import PIL.Image

import os
import pathlib
import itertools
import random


# TODO: run augmentation on a whole class and export to augmented_small_data/

# class_name = "n01443537"
# id = 0

# input_file = hf.make_filename(class_name, id, extension="JPEG")
# original_img = hf.read_image_from_local_storage(input_file, folder=class_name)
# print(f"Original Image:")
# hf.show(original_img)

# new_image = original_img.copy()
# print(f"New Image:")
# hf.show(new_image)


# hf.export_image_to_local_storage(new_image, id, folder=class_name)


'''
Plan
- make list of all classes and numbers (crawl the filepath)
- set variables for id and number  (this is where the for-loop will go)
- read one image from local storage
- augment one image
- export one image to local storage
- repeat loop

'''

test = "n01443537"
filepath = f"./small_data/tiny-imagenet-200/train/{test}/images/"
files = hf.make_list_of_files_from_filepath(filepath)
files.sort()
# print(type(files))
# print(files)


filepath = f"./small_data/tiny-imagenet-200/train/"
directories = hf.make_list_of_directories_from_filepath(filepath)
directories.sort()
# print(type(directories))
# print(directories)

# TODO: now augment an image!
for folder in directories:
    for file in files:
        original_img = hf.read_image_from_local_storage(
            file, folder=folder, small_data=True)
        print(f"Original Image:")
        hf.show(original_img)

        new_image = original_img.copy()
        print(f"New Image:")
        hf.show(new_image)

        break  # to do only one image for testing
        # hf.export_image_to_local_storage(new_image, file, folder=folder)  # uncomment to export
    break  # to only do once for testing
