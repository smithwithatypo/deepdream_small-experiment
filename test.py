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

choices = hf.pick_random_choices()
print(type(choices))
print(choices)
activated_layers = hf.add_prefix(choices)
print(type(activated_layers))
print(activated_layers)
