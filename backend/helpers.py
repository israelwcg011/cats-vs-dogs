# imports
import numpy as np
from PIL import Image
from config import IMAGE_SHAPE

# function to prepare new images
def prepare_image(image_path):
    im = Image.open(image_path)
    im = im.resize(IMAGE_SHAPE)
    im = np.expand_dims(im, axis=0)
    im = np.array(im)
    im = im / 255
    return im