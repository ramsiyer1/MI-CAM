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
from MI CAM import smoothen_cam
from MI CAM import get_mi_cam

def apply_mask(image, cam_mask, threshold=0.3):
    # Normalize the CAM mask to the range [0, 1] (if not already)
    cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min())

    # Expand cam_mask to have the same number of channels as the image
    cam_mask = np.expand_dims(cam_mask, axis=-1)
    cam_mask = np.repeat(cam_mask, image.shape[-1], axis=-1)  # repeat the mask for each channel

    # Create the mask where cam_mask values below the threshold are set to 0
    masked_image = np.where(cam_mask < threshold, 0, image)

    return masked_image

model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
img_path = "MI-CAM/Input Images/2 birds.jpg"    #change accordingly
img = image.load_img(img_path, target_size=(224, 224))

pred = model.predict(x)
pred_class = np.argmax(pred)
#original probabilities
original_prob = pred[:,pred_class]
#get mi_cam
mi_cam = get_mi_cam(model, img, 'block5_conv3')
#mask the image and get masked probabilities
masked_image = apply_mask(np.array(img), mi_cam)
masked_pred = model.predict(np.expand_dims(masked_image, axis=0))
masked_prob = masked_pred[:,pred_class]
#average drop and average increase function
def drop_or_increase(masked_prob, original_prob):
    if masked_prob < original_prob:
        drop = (original_prob - masked_prob) / original_prob * 100
        print("Drop is {drop:.2f}%")
        return drop
    elif masked_prob > original_prob:
        increase = (masked_prob - original_prob) / original_prob * 100
        print("Increase is {increase:.2f}%")
        return increase
