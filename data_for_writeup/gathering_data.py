from keras.models import load_model,Model,Sequential
from keras.layers import Cropping2D
import cv2
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
with open('../../../opt/F:/data_three_laps/driving_log.csv','r') as csvfile:

    lines  = csv.reader(csvfile)
    
    lines = np.asarray(list(lines))[1:,:]
    
    sample = lines[40,:]
img_path = "../../../opt/"+"/".join(sample[0].split("\\"))
# plt.imshow(cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
# plt.savefig("original_feature_map_img")
model = load_model('model.h5')
model = Model(inputs = model.inputs,outputs = model.layers[4].output)
img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
output = model.predict(np.expand_dims(img,axis = 0))
ix = 1
print(output.shape)
fig, ax = plt.subplots(4,4)
print(ax)
# for _ in range(4):
# 	for _ in range(4):
# 		# specify subplot and turn of axis
# 		ax = plt.subplot(4, 4, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# plot filter channel in grayscale
# 		plt.imshow(output[0, :, :, ix-1],cmap='gray')
# 		ix += 1
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i][j].imshow(output[0,:,:,ix-1],cmap='gray')
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
        ix = ix+1
plt.savefig("feature_maps_conv_2")
    