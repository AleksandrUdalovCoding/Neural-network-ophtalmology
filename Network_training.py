#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import os
import numpy as np
import tensorflow as tf
import glob
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# In[2]:


# set parameters of your PC (Uncomment if necessary)

#config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 8} )
#sess = tf.compat.v1.Session(config=config) 
#K.set_session(sess)

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())'''


# In[3]:


# Train catalog
train_dir = 'train'

# Validation catalog
val_dir = 'val'

# Test catalog
test_dir = 'test'

# Current folder
curr_fold = os.getcwd()

# Image resolution
img_width, img_height = 150, 150

# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)

# Batch size
batch_size = 1

# Number of samples for training
# nb_train_samples = 110
nb_train_samples = sum(1 for i in glob.iglob(train_dir+ '/*/*.png'))
print('Number of samples for training:', nb_train_samples)

# Number of samples for checking
#nb_validation_samples = 24
nb_validation_samples = sum(1 for i in glob.iglob(val_dir+ '/*/*.png'))
print('Number of samples for checking:', nb_validation_samples)

# Number of samples for testing
#nb_test_samples = 24
nb_test_samples = sum(1 for i in glob.iglob(test_dir+ '/*/*.png'))
print('Number of samples for testing:', nb_test_samples)


# In[4]:


# Increasing the number of training images 
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  zoom_range=0.2,
                                  channel_shift_range=True,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

# image example (Uncomment if necessary)

#image_file_name = train_dir + '/Correct_astigmatism_cut/Correct_astigmatism_cut (1).png'
#img = image.load_img(image_file_name, target_size=(img_width, img_height))
#plt.imshow(img)


# In[5]:


# examples of changed images (Uncomment if necessary)
#x = image.img_to_array(img)
#x = x.reshape((1,) + x.shape)
#i = 0
#for batch in train_datagen.flow(x, batch_size=1):
    #examples of chenged image (it necessary)
    #plt.figure(i)
    #imgplot = plt.imshow(image.array_to_img(batch[0]))
    #i += 1
    #if i % 4 == 0:
        #break
#plt.show()


# In[6]:


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[7]:


test_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[8]:


# Loading the VGG16 network
vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
vgg16_net.trainable = False

# VGG16 Network summary (Uncomment if necessary)
#vgg16_net.summary() 


# In[9]:


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[10]:


model = Sequential()

# Add VGG16 instead input layer in our network
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Network summary (Uncomment if necessary)
#model.summary()


# In[11]:


model.compile(loss='binary_crossentropy',
              optimizer=Adam(), 
              metrics=['accuracy'])


# In[12]:


checkpoint_filepath = curr_fold + '/input_layers_training_results/best_input.hdf5'
best_result_dense = [ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True,save_weights_only=True)]


# In[13]:


# Train our network (except VGG16 layers)
# !!! Rewrite old training result in the folder /input_layers_training_results!!!
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=val_generator,
    validation_steps=20,
    callbacks=best_result_dense
)


# In[14]:


# Add the best of saved weights inside the network
model.load_weights(curr_fold + '/input_layers_training_results/best_input.hdf5')


# In[15]:


# PLot the accuracy during all the epochs
plt.plot(history.history['accuracy'], label='The percentage of correct answers on the train dataset')
plt.plot(history.history['val_accuracy'], label='The percentage of correct answers on the validation dataset')
plt.xlabel('Epoch')
plt.ylabel('The percentage of correct answers')
plt.legend()
plt.show()
print(nb_test_samples)
print(batch_size)
scores = model.evaluate(test_generator)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))


# In[16]:


# Set the VGG16 trainable only
vgg16_net.trainable = True
trainable = False
for layer in vgg16_net.layers:
    if layer.name == 'block1_conv1':
        trainable = True
    layer.trainable = trainable
    
# Network summary (Uncomment if necessary)
#model.summary()


# In[17]:


model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-5), 
              metrics=['accuracy'])


# In[18]:


checkpoint_filepath = curr_fold + '/vgg16_layers_training_results/best_conv.hdf5'
results_conv = [ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True,)]


# In[19]:


# Training the VGG16 network
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=results_conv)


# In[20]:


# Add the best of saved weights inside the network
model.load_weights(curr_fold + '/vgg16_layers_training_results/best_conv.hdf5')


# In[21]:


# Plot the accuracy during all the epochs
plt.plot(history.history['accuracy'], label='The percentage of correct answers on the train dataset')
plt.plot(history.history['val_accuracy'], label='The percentage of correct answers on the validation dataset')
plt.xlabel('Epoch')
plt.ylabel('The percentage of correct answers')
plt.legend()
plt.show()
scores = model.evaluate(test_generator)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))


# In[22]:


# Save parameters of our network
model_json = model.to_json()
json_file = open(curr_fold + "/final_network_params/astigmatism_binary.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights(curr_fold + "/final_network_params/astigmatism_binary.h5", save_format="h5")


# In[23]:


# An example of a neural network working on a single image (Uncomment if necessary)

#img_path = curr_fold + '/Images for testing final network/Correct_astigmatism_cut (70).png'
#img = image.load_img(img_path, target_size=(150,150))
#x = image.img_to_array(img)
#x /= 255
#x = np.expand_dims(x, axis = 0)
#plt.imshow(img)


# In[24]:


#loaded_model.summary()


# In[25]:


#conv_layers = loaded_model.get_layer(index=0)
#conv_layers.summary()


# In[26]:


# Indexes of convolutional layers - 1,2,4,5,7,8,9,11,12,13,15,16,17    3, 6, 10, 14, 18
#activation_model = Model(inputs=conv_layers.input, outputs=conv_layers.layers[11].output)
#activation_model.summary()


# In[27]:


#activation = activation_model.predict(x)


# In[28]:


#print(activation.shape)


# In[29]:


#plt.matshow(activation[0, :, :, 18], cmap='viridis')


# In[30]:


#images_per_row = 10
#n_filters = activation.shape[-1]
#size = activation.shape[2]
#n_cols = n_filters // images_per_row


# In[31]:


#display_grid = np.zeros((n_cols * size, images_per_row * size))


# In[32]:


#for col in range(n_cols):
    #for row in range(images_per_row):
        #channel_image = activation[0, :, :, col * images_per_row + row]
        #channel_image -= channel_image.mean()
        #channel_image /= channel_image.std()
        #channel_image *= 64
        #channel_image += 128
        #channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        #display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image


# In[33]:


#scale = 1. / size
#plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
#plt.grid(False)
#plt.imshow(display_grid, aspect='auto', cmap='viridis')

