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

model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
img_path = "MI-CAM/Input Images/2 birds.jpg"    #change accordingly
img = image.load_img(img_path, target_size=(224, 224))

def smoothen_cam(cam, method, kernel_size=11, sigma=7):
  if method == 'gaussian':
    smoothed_cam = cv2.GaussianBlur(cam, (kernel_size, kernel_size), sigma)
  elif method == 'bilateral':
    smoothed_cam = cv2.bilateralFilter(cam.astype(np.float32), kernel_size, sigma, sigma)
  elif method == 'average':
    smoothed_cam = cv2.blur(cam, (kernel_size, kernel_size))
  else:
    raise ValueError(f"Unknown method '{method}'. Choose from 'gaussian', 'bilateral', or 'average'.")

  return smoothed_cam

def get_mi_cam(model, img, layer_name):
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  mi_cam_model = tf.keras.models.Model(model.inputs, model.get_layer(layer_name).output)
  feature = mi_cam_model.predict(x)
  #upscale feature map
  input_shape = np.squeeze(feature).shape  #(14, 14, 512) for layer_name = 'block5_conv3'
  original_image_shape = (224, 224, input_shape[-1]) #(224, 224,512)
  upsampling_factor = (original_image_shape[0] // input_shape[0], original_image_shape[1] // input_shape[1])
  upsample_layer = tf.keras.layers.UpSampling2D(size=upsampling_factor)
  upsampled_feature_map = upsample_layer(feature)
  upsampled_feature_map_1 = tf.image.resize(upsampled_feature_map, (224,224))
  #convert image to grayscale and flatten both image and feature maps.
  y = image.img_to_array(tf.image.rgb_to_grayscale(img))
  image_flattened = y.flatten()
  entropy_image = entropy(image_flattened)
  #calculate mutual information scores
  mutual_information = []
  for i in range(upsampled_feature_map_1.shape[-1]):
    feature_map_flattened = np.array(upsampled_feature_map_1[:,:,:,i]).flatten()
    mutual_information.append(mutual_info_score(image_flattened, feature_map_flattened))
  #get final cam output
  final_output = np.zeros((1, 224, 224))
  for i in range(upsampled_feature_map_1.shape[-1]):
    final_output = final_output + (mutual_information[i] * np.array(upsampled_feature_map_1[:,:,:,i]))
  final_output_resized = np.squeeze(final_output)
  smoothed_cam = smoothen_cam(final_output_resized, 'gaussian', kernel_size=11, sigma=7)
  smoothed_cam = np.max(smoothed_cam, 0)
  return smoothed_cam  
