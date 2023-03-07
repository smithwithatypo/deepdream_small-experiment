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
import time


# make a list of all folders in small_data_original/
# make a list of all files in each {folder}/images/
# augment one file by running tutorial code
# export augmented_image to data/small_data_augmented/{folder}/{file}

start_time = time.time()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fifth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Allow TensorFlow to allocate only as much GPU memory as needed
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet')

random_choices = hf.pick_random_choices(seed=0)
activated_layers = hf.add_prefix(random_choices)
layers = [base_model.get_layer(name).output for name in activated_layers]
print(f"layers: {activated_layers}")

dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))


def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


class DeepDream(tf.Module):

    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


deepdream = DeepDream(dream_model)


def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        display.clear_output(wait=True)
        show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)

    return result


black_white_images = set()

route_dict = {"train": "data/tiny-imagenet-200/train",
              "val": "data/tiny-imagenet-200/val",
              "test": "data/tiny-imagenet-200/test",
              "words": "data/tiny-imagenet-200/words.txt",
              "augmented": "data/augmented",
              "small_orig": "data/small_data_original",
              "small_aug": "data/small_data_augmented"}
experiment = "deepdream_small"    # "deepdream" or "deepdream_small"
metadata = "images"    # "images" or "metadata"
batch_number = None    # to track activated layers per batch

filepath = f"./data/small_data_original"
directories = hf.make_list_of_directories_from_filepath(filepath)

for folder in directories:
    nested_filepath = f"./{route_dict['small_orig']}/{folder}/{metadata}/"
    files = hf.make_list_of_files_from_filepath(nested_filepath)
    for file in files:
        if os.path.exists(f"./{route_dict['small_aug']}/{folder}/images/{file}"):
            print("Already augmented")
            continue

        # original_img = hf.read_image_from_local_storage(
        #     file, folder=folder, route=None, small_data=True)
        # print(f"Original Image:", type(original_img), original_img.shape)

        route = route_dict["small_orig"]
        original_img = hf.import_file(
            experiment=experiment,
            route=route,
            folder=folder,
            metadata=metadata,
            file=file
        )
        print(f"Original Image:", type(original_img), original_img.shape)

        if original_img.shape == (64, 64):
            print("Image is not RGB")
            black_white_images.add(file)
            continue

        dream_img = run_deep_dream_simple(img=original_img,
                                          steps=40, step_size=0.01)

        hf.show(dream_img)
        # break  # debugging before export

        route = route_dict["small_aug"]
        batch_number = batch_number
        hf.export_file(
            dream_img,
            experiment=experiment,
            route=route,
            folder=folder,
            metadata=metadata,
            file=file,
            file_suffix=batch_number,
        )
        # break  # comment out to run on all files

    break  # comment out to run on all folders

print(f"There are {len(black_white_images)} black and white images:")
print(black_white_images)

end_time = time.time()
total_time = end_time - start_time

print("Time elapsed:", round(total_time, 2), "seconds")
