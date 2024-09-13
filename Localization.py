#Inorder to get the localization results on the proposed CAM.
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
import cv2
from MI CAM import smoothen_cam
from MI CAM import get_mi_cam
from Drop_Increase import apply_mask

model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
img_path = "MI-CAM/Input Images/2 birds.jpg"    #change accordingly
img = image.load_img(img_path, target_size=(224, 224))

cam = get_mi_cam(model, img, 'block5_conv3')
masked_image = apply_mask(np.array(img), cam)

plt.imshow(masked_image)
plt.show
