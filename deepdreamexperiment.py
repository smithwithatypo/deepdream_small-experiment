
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


"""# Define functions from the DeepDream official tutorial"""

# Create the base model: InceptionV3

base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet')


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
                loss = hf.calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


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

        # display.clear_output(wait=True)
        # show(deprocess(img))
        #print ("Step {}, loss {}".format(step, loss))

    result = hf.deprocess(img)
    # display.clear_output(wait=True)
    hf.show(result)

    return result


"""# Testing

"""  # Executing code"""

id = "n01443537"
number = 103              # try 0, 10, 100, 102, 103

input_file = hf.make_filename(id, number, extension="JPEG")
test_img = hf.read_image_from_local_storage(input_file, folder=id)

original_img = test_img

# Decides which random layers will be activated

quantity_of_choices = 5
seed = None    # select None for pure randomness

all_combos = hf.find_all_combinations(start=0, end=10)
random_layer_combinations = hf.pick_random_choices(
    all_combos, quantity=quantity_of_choices, seed=seed)
random_layer_combinations_with_prefix = hf.add_prefix(
    random_layer_combinations)

"""## Q: How long does it take to run 1 model on 5 different images?
## A: 29 seconds
"""

activation_layers = random_layer_combinations_with_prefix[0]

print(f"Activation layers: {activation_layers}")
layers = [base_model.get_layer(name).output for name in activation_layers]
dream_model = hf.create_layer_activated_model(
    base_model=base_model, layers=layers)
deepdream = DeepDream(dream_model)


id = "n01443537"
numbers = [0, 10, 100, 102, 103]

for number in numbers:
    input_file = hf.make_filename(
        id, number, extension="JPEG")  # comment this out
    original_img = hf.read_image_from_local_storage(
        input_file, folder=id)  # comment this out
    # TODO: input file/folder for-loop here from test.py.

    print(f"Original Image:")
    hf.show(original_img)

    augmented_image = run_deep_dream_simple(img=original_img, steps=10,
                                            step_size=0.01)  # prints augmented image
    print(type(augmented_image))
    hf.show(augmented_image)

##### Archive / Trash
