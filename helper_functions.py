import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import IPython.display as display
import PIL.Image

import os
import pathlib
import itertools
import random


def import_file(experiment, route, folder, metadata, file):
    ''' UNDER CONSTRUCTION '''

    image_path = tf.keras.utils.get_file(
        file, f"file:///home/wpx1/{experiment}/{route}/{folder}/{metadata}/{file}")
    print(
        f"On this image: file:///home/wpx1/{experiment}/{route}/{folder}/{metadata}/{file}")

    img = PIL.Image.open(image_path)
    final_img = np.array(img)

    return final_img


def export_file(image, experiment, route, folder, metadata, file, file_suffix=None):
    ''' UNDER CONSTRUCTION '''

    if file_suffix:
        filename = file.split("."[0])
        filetype = file.split("."[1])
        file = f"{filename}_{file_suffix}.{filetype}"

    image_path = tf.keras.utils.get_file(
        file, f"file:///home/wpx1/{experiment}/{route}/{folder}/{metadata}/{file}")
    print(
        f"On this image: file:///home/wpx1/{experiment}/{route}/{folder}/{metadata}/{file}")

    # export code is below this
    if not os.path.exists(f"./data/small_data_augmented/{folder}/images/"):
        os.makedirs(
            f"./data/small_data_augmented/{folder}/images/")
    else:
        pass

    # if the file already exists, pass
    if os.path.exists(f"./{route}/{folder}/{metadata}/{file}"):
        print(
            f"File already exists: ./{route}/{folder}/{metadata}/{file}")
    else:
        tf.keras.utils.save_img(
            f"./{route}/{folder}/{metadata}/{file}", image)

        print(
            f"Saved to ./{route}/{folder}/{metadata}/{file}")

    # TODO: handle metadata


def make_list_of_directories_from_filepath(filepath):
    ''' make a list of directories from a filepath'''

    directories = [d for d in os.listdir(
        filepath) if os.path.isdir(os.path.join(filepath, d))]

    return directories


def make_list_of_files_from_filepath(filepath):
    ''' make a list of files from a filepath'''

    files = [f for f in os.listdir(filepath) if os.path.isfile(
        os.path.join(filepath, f))]

    return files


def make_filename(class_name, id=0, extension="JPEG"):
    ''' append number and file extension to a filename '''

    result = f"{class_name}_{id}.{extension}"
    return result


# def find_all_combinations(start=0, end=0):
#     ''' generate a list of all possible combinations
#         between start and end (inclusive) '''

#     array = range(start, end + 1)
#     result = []

#     for i in range(start, end + 1):
#         for subset in itertools.combinations(array, i):
#             result.append(subset)
#     return result


# def pick_random_choices(arr, quantity, seed=None):
#     ''' pick a quantity of random items in an array.
#         (optional random seed) '''

#     if seed:
#         random.seed(seed)

#     return random.choices(arr, k=quantity)

def pick_random_choices(start=0, end=10, quantity=4, seed=None):
    array = range(start, end + 1)
    result = []

    for i in range(start, end + 1):
        for subset in itertools.combinations(array, i):
            result.append(subset) if len(subset) <= quantity else None

    if seed:
        random.seed(seed)

    return random.choices(result, k=1)


def add_prefix(input):
    ''' converts tuples to lists, then prefixes each index with "mixed",
        then converts each item to a string '''
    result = []

    for item in input:
        item = list(item)
        item = [f"mixed{i}" for i in item]
        item = [str(i) for i in item]
        result.append(item)
    return result[0]


def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))


def calc_loss(img, model):
    ''' Pass forward the image through the model to retrieve the activations.
        Converts the image into a batch of size 1. '''
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


def create_layer_activated_model(base_model, layers):
    ''' returns a model with specified layers activated '''

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    return dream_model


### Archive / Trash


# def read_image_from_local_storage(image, folder, route="train", small_data=False):
#     ''' read an image from file:///home/wpx1/deepdream/data/tiny-imagenet-200/ '''

#     if (route == "train" or
#             route == "test" or
#             route == "val" or
#             route == None):
#         pass
#     else:
#         print("Please input route=\"train\" or \"test\" or \"val\" ")

#     if small_data:
#         test_image_path = tf.keras.utils.get_file(
#             image, f"file:///home/wpx1/deepdream_small/data/small_data_original/{folder}/images/{image}")
#     else:
#         # test_image_path = tf.keras.utils.get_file(
#         #     image, f"file:///home/wpx1/deepdream/data/tiny-imagenet-200/{route}/{folder}/images/{image}")
#         # print(f"Look here for the file: {test_image_path}")  # for debugging
#         pass

#     print(
#         f"On this image: file:///home/wpx1/deepdream_small/data/small_data_original/{folder}/images/{image}")
#     img = PIL.Image.open(test_image_path)
#     final_img = np.array(img)

#     return final_img


# def export_image_to_local_storage(image, folder, file):
#     ''' export an image to .data/small_data_augmented/ '''

#     # mkdir if it doesn't exist
#     if not os.path.exists(f"./data/small_data_augmented/{folder}/images/"):
#         os.makedirs(
#             f"./data/small_data_augmented/{folder}/images/")
#     else:
#         pass

#     # if the file already exists, pass
#     if os.path.exists(f"./data/small_data_augmented/{folder}/images/{file}"):
#         print(
#             f"File already exists: ./data/small_data_augmented/{folder}/images/{file}")
#     else:
#         tf.keras.utils.save_img(
#             f"./data/small_data_augmented/{folder}/images/{file}", image)

#         print(
#             f"Saved to ./data/small_data_augmented/{folder}/images/{file}")
