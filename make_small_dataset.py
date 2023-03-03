from pathlib import Path
import shutil
import os

# TODO:  handle black and white images. Keep shape (64,64,3) - Keras is auto-converting to (64, 64)
class_name = "n02769748"
source = f"/home/wpx1/deepdream/data/tiny-imagenet-200/train/{class_name}/images/"
target = f"/home/wpx1/deepdream/small_data/tiny-imagenet-200/train/{class_name}/images/"

try:
    os.makedirs(target)
except OSError as error:
    print(error)


files = os.listdir(source)

# iterating over all the files in the source directory
for fname in files[0:21]:
    # copying the files to the destination directory
    # print(os.path.join(source, fname), target)
    shutil.copy2(os.path.join(source, fname), target)
