import numpy as np
import matplotlib.pyplot as plt
import cv2 

from os import listdir
from seg_lib import dice_coef_np, selective_unet

images_udiat, rois_udiat = [], []

dsize = (224, 224)

path = 'data/udiat/'
img_file = listdir(path+'original')

for i, file in enumerate(img_file):
    
    img = cv2.imread(path+'original/'+file, 0)
    roi = cv2.imread(path+'GT/'+file, 0)/255
    
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
    roi = cv2.resize(roi, dsize, interpolation=cv2.INTER_NEAREST)
    
    images_udiat.append(img)
    rois_udiat.append(roi)
    
images_udiat = np.array(images_udiat, dtype=np.float32)
images_udiat = np.expand_dims(images_udiat, 3)

rois_udiat = np.array(rois_udiat, dtype=np.int16)
rois_udiat = np.expand_dims(rois_udiat, 3)

model = selective_unet()
model.load_weights('models/skunet_weights.h5')

rois_predicted = model.predict(images_udiat).squeeze().round()

dices = np.zeros(rois_predicted.shape[0])

for i in range(rois_predicted .shape[0]):
    
    dices[i] = dice_coef_np(rois_predicted[i], rois_udiat[i])

# Results are slightly different compared to our paper. Originally, we additionally preprocessed the images in Matlab.
print('Dice scores | mean:', np.mean(dices).round(3), 'median:', np.median(dices).round(3), 'mean Dice>0.5:', np.mean(dices[dices>0.5]).round(3))