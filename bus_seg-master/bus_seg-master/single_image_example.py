import numpy as np
import matplotlib.pyplot as plt
import cv2
from seg_lib import dice_coef_np,selective_unet

plt.rcParams["figure.figsize"] = (15, 15)
plt.rc('font', size=12)

img = cv2.imread('data/example.png', 0) 
roi = cv2.imread('data/example_roi.png', 0)/255
img = np.expand_dims(img, 0)
img = np.expand_dims(img, 3)
print('image shape:', img.shape, 'roi shape:', roi.shape)

model = selective_unet()
model.load_weights('models/skunet_weights.h5')

# import os
# os.environ['PATH'] += os.pathsep + r'C:/Program Files/Graphviz/bin'
# from tensorflow.python.keras.utils.vis_utils import plot_model
# plot_model(model, show_shapes=True,\
#            show_layer_names=False,\
#            to_file='_architecture.png') 

roi_predicted = model.predict(img).squeeze().round()

plt.figure()

plt.subplot(131)
plt.imshow(np.round(img.squeeze()), cmap='gray')

plt.subplot(132)
plt.imshow(roi, cmap='gray')

plt.subplot(133)
plt.imshow(roi_predicted, cmap='gray')

plt.show()

print('Dice score:', dice_coef_np(roi_predicted, roi).round(3))