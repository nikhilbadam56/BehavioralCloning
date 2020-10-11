import os
import pandas as pd
import numpy as np
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import  Dense, Conv2D, Lambda,Flatten,Dropout,MaxPooling2D,Cropping2D
from math import ceil
import matplotlib.pyplot as plt

#defining some of the model hyperparameters
batch_size = 32
epochs = 6

#defining a header for both the dataframes to be combined
columns=["left image","center image", "right image", "steering","throttle","brake","some"]

#loading the dataframe of normal three laps data and dataframe of opposite drive three laps 
df  = pd.read_csv('../../../opt/F:/data_three_laps/driving_log.csv',names = columns)
df2 = pd.read_csv('../../../opt/F:/data_three_laps/data_reverse_two_laps/driving_log.csv',names = columns)

#dataframe loading of map2 data
# df3 = pd.read_csv('../../../opt/F:/data_refined_four_laps/map2_data/driving_log.csv',names = columns)
# df4 = pd.read_csv('../../../opt/F:/data_refined_four_laps/map2_data/map2_reverse/driving_log.csv',names = columns)

#combining the dataframes for obtainig the data in combined faashion
final_df_csv =pd.concat([df, df2],ignore_index=True)

#combining the map1 and mp2 dataframes 
# final_df_csv = pd.concat([df,df2,df3,df4],ignore_index=True)

#saving the concatenated dataframes into another csv file
final_df_csv.to_csv('../../../opt/F:/data_three_laps/driving_log_combined.csv')
#loading the csv data from the driving log file

samples = []

with open('../../../opt/F:/data_three_laps/driving_log_combined.csv','r') as csvfile:

    lines  = csv.reader(csvfile)
    
    lines = np.asarray(list(lines))[1:,:]

    for line in lines:
        
        samples.append(line)


#defining a generator function to load the specific amount of data per batch due to memory constraints
def data_generator(data,batch_size):

    number_of_samples = len(data)
        
    #not ending the generator from generating or yielding data whenever called
    while 1:

        shuffle(data) #randomly shuffle data before trying to batch the data

        for offset in range(0,number_of_samples,batch_size):

            batch_data = data[offset:offset+batch_size]
            
            images = [] #batch images
            labels = [] #corresponding labels

            for batch_sample in batch_data:
                
                for i in [1,2,3]:
                
                    path = "../../../opt/"+"/".join(batch_sample[i].split("\\"))

                    img = cv2.imread(path)

                    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                    images.append(image)
                    
                    #using images from all the cameras as data for training , this can also serve purpose of data augmentation
                    
                    if i ==1:
                        
                        #center image
                        
                        labels.append(float((batch_sample[4])))
                        
                    if i == 2:
                        
                        #left image
                        
                        labels.append(float((batch_sample[4]))+0.2) #offseting the left angle by 0.2 owing to the distance of camera from                                                                       center 
                                                                     
                        
                    if i == 3:
                        
                        #right image 
                        
                        labels.append(float((batch_sample[4]))-0.2) #offeting the right angle by -0.2 owing to the distance of camera from 
                                                                    #center
                    
                    #augmenting the data further by flipping the images and thereby flipping the steering angles
                    
                    image = cv2.flip(image,1)
                    
                    images.append(image)
                    
                    if i==1:
                        
                        labels.append(float(batch_sample[4])*-1)#flipping center image steering angle
                    
                    if i ==2:
                    
                        labels.append((float(batch_sample[4])+0.2)*-1)#flipping left image steering angle
                    
                    if i==3:
                        
                        labels.append((float(batch_sample[4])-0.2)*-1) #flipping right image steering angle
                    

            yield shuffle(np.asarray(images),np.asarray(labels)) #yielding the batch data by shuffling in batch data

#splitting data into test and train , here test actually refers to the validation dataset
train_data,test_data,train_labels,test_labels = train_test_split(samples,np.array(samples)[:,3],test_size = 0.3,shuffle = True)

train_data_generator = data_generator(train_data,batch_size) #generator for the training data
valid_data_generator = data_generator(test_data,batch_size) #generator for the test data

def normalize_channel_change(x):
    
    """
    Funtion to normalize the image 
    Input: x(image) shape: 160X320X3
    Output : x(normalized_image) shape: 160X320X3
    
    """
    
    x = x/127.5 - 1
    
    return x

# defining the model architectur using keras sequential api
ch,row,col = 3,160,320

#using keras sequential api for defining the model
model  = Sequential()
model.add(Lambda(normalize_channel_change ,input_shape=(row,col,ch)) )

#using cropping layer provided by the keras to crop off the 50% of image height to remove not useful information for prediction.
model.add(Cropping2D(cropping=((50,25),(0,0))))

#First convolution layer with 24 filters of size 5X5,strides of 2 along each axis
model.add(Conv2D(filters = 24,kernel_size = (5,5),strides = (2,2),activation='elu'))

#Second convolution layer with 36 filters of size 5X5,strides of 2 along each axis
model.add(Conv2D(filters = 36,kernel_size = (5,5),strides = (2,2),activation='elu'))

#Third convolution layer with 48 filters of size 5X5,strides of 2 along each axis
model.add(Conv2D(filters = 48,kernel_size = (5,5),strides = (2,2),activation='elu'))

#Fourth convolution layer with 64 filters of size 3X3,strides of 1 along each axis
model.add(Conv2D(filters = 64,kernel_size = (3,3),activation='elu'))

#Fifth convolution layer with 64 filters of size 3X3,strides of 1 along each axis
model.add(Conv2D(filters = 64,kernel_size = (3,3),activation='elu'))

#flattening the previous layer's output to feed into neural network for a prediction
model.add(Flatten())

#fully connected layer 1
model.add(Dense(1164,activation='elu'))
model.add(Dropout(0.2))

#fully connected layer2
model.add(Dense(100,activation='elu'))
model.add(Dropout(0.3))

#fully connected layer3
model.add(Dense(50,activation='elu'))
# model.add(Dropout(0.3))#adding additional regularization for data mixed with map 2 data to prevent model overfitting the data
#fully connected layer4
model.add(Dense(10,activation='elu'))

#Final fully connected layer5 
model.add(Dense(1))

model.compile(loss='mse',optimizer = 'adam') #compiling the model with "mean squared error" loss function and with "Adam" optimizer algorithm.

#using previous train and valid generator defined before for getting the data on the go without flooding the memory
history =model.fit_generator(train_data_generator,steps_per_epoch = ceil(len(train_data)/batch_size),validation_data = valid_data_generator,validation_steps = ceil(len(test_data)/batch_size),epochs = epochs,verbose = 2)  

#saving the model trained on map 1 and corresponding augmented data
model.save('./model.h5')

#saving the model trained on both map 1 and corresponding augmented ata
# model.save('./model_map2.h5')

#plotting the performance of the model over the course of epochs

plt.plot(history.history['loss']) #traiing data loss
plt.plot(history.history['val_loss']) #validation data loss

plt.xlabel("epochs")
plt.ylabel("Loss")

plt.legend(['training set', 'validation set'], loc='upper right')

plt.title('model mean squared error loss')

print("PLOT BEING SAVED......")

plt.savefig("plt_of_loss")

print("PLOT SAVED.")