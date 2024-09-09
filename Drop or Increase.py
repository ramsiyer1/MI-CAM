from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from skimage import io
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
def apply_mask(image, cam_mask, threshold=0.3):
    # Normalize the CAM mask to the range [0, 1] (if not already)
    cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min())

    # Expand cam_mask to have the same number of channels as the image
    cam_mask = np.expand_dims(cam_mask, axis=-1)
    cam_mask = np.repeat(cam_mask, image.shape[-1], axis=-1)  # repeat the mask for each channel

    # Create the mask where cam_mask values below the threshold are set to 0
    masked_image = np.where(cam_mask < threshold, 0, image)

    return masked_image
