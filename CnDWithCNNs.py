#!/usr/bin/env python
# coding: utf-8

# ## Yoda Daniel
# ### Cats and Dogs with Conv. NN

# #### We begin with importing our previously saved data form the .npz file 
# #### Then assign the correct array values with train/test variables ( make sure to keep same order ) 
# #### With Conv NN, make sure to have correct shape of data
#     The X training and testing data should have four dimensions, with 1 being the last one Ex: ( X, X, X, 1)
#     The Y training and test data stays two dimensions with 1 being the last one (unlike in linear reg, Standard NN Keras)
# #### Lastly, print arrays to ensure correct shape

# In[1]:


import os, cv2, itertools # cv2 -- OpenCV
import numpy as np 
import pandas as pd 
import time
 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = np.load('cat_equal_2.npz')


# In[3]:


data.files


# In[4]:


X_train = data['arr_0']
X_test = data['arr_1']
Y_train = data['arr_2']
Y_test = data['arr_3']


# In[5]:


X_train = X_train.T/255
X_test = X_test.T/255
Y_train= Y_train.T
Y_test = Y_test.T


# In[6]:


print('Shape of X_train : {}'. format(X_train.shape))
print('Shape of X_test : {}'. format(X_test.shape))
print('Shape of Y_train : {}'. format(Y_train.shape))
print('Shape of Y_test : {}'. format(Y_test.shape))


# In[7]:


X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)   #samples, w, h, channels
X_test = X_test.reshape(X_test.shape[0], 64, 64, 3)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)


# In[8]:


print('Shape of x_train2 is {}'.format(X_train.shape))
print('Shape of x_test2 is {}'.format(X_test.shape))
print('Shape of y_train is {}'.format(Y_train.shape))
print('Shape of y_test is {}'.format(Y_test.shape))


# In[9]:


X=X_train
Y=Y_train

Y.shape


# In[10]:


X.shape


# # With CNNS
# 
# #### I began by importing the many packages needed, the utils was not needed
# #### Then I constructed the Model intself which consists of
#     2 two dimensions Convelutional layers
#     2 two imension max pooling layers
#     2 Dropout layers
#     A dense layer with 300 nuerons and a relu activation function
#     A dense ouput layer with sigmoid activation
# #### The compiler consists of the Adam optimizer and a binary cross entropy loss function
# #### The model is ran with paramters of 5 iterations, a batch size of 50, and 20% split of data
#     Takes about 5-10 mins
# #### Using the .evalute from Keras, see the accuracy of the model: Mine was 79.6%
# #### Then using matlabplotlib, print the loss and accuracy over iterations on a chart to evalate the slope
# 
# #### Lastly, using .predict, run the Trainging data and Testing data in the model
#     My training accuracy rate is 85.0%
#     My testing accuracy rate is 79.6% 
#     
#     

# In[11]:


# Attribution: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten


# In[17]:


model_cnn = Sequential()
model_cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size = (2,2)))
model_cnn.add(Dropout(0.2))
model_cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size = (2,2)))
model_cnn.add(Dropout(0.2))
model_cnn.add(Flatten())
model_cnn.add(Dense(units = 300, activation = 'relu'))
model_cnn.add(Dense(units = 1, activation = 'sigmoid'))


# In[18]:


model_cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[19]:


#MD= model_cnn.fit(X.T,Y.T, validation_split=0.2, epochs=10, batch_size=50, verbose = 1)


# In[20]:


history = model_cnn.fit(X,Y,epochs=5,batch_size=50,validation_split=.2,verbose=1)


# In[21]:


score = model_cnn.evaluate(X_test, Y_test, verbose=1)


# In[22]:


plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('model loss')
plt.ylabel('loss & accuracy')
plt.xlabel('epoch')


# In[26]:


#Yh = model.predict(X.T)
Yh = model_cnn.predict(X)

acc = np.mean(np.rint(Yh) == Y_train) # caulate train accuracy rate
print('Training accuracy rate: [{}]'.format(acc*100))


# In[29]:


Yh_test = model_cnn.predict(X_test)
acc = np.mean(np.rint(Yh_test) == Y_test) # caulate test accuracy rate
print('Test accuracy rate: [{}]'.format(acc*100))


# ### Printing the model's prediction
# #### First, use the previous function read_image to resize images
# #### After, that the training and testing datas are also reshaped
# #### The model is then trained and saved to yhat2
# #### Yhat2 is then reshaped for when we start assigning indicees
# #### We then assign our indices to either correct'incorrect  and print there shapes
#      The shape of the "correctindices" equals how many of the images the model predicited correctly and "incorrect indices" shows how many were incorrectly predicited.
# #### Finally, we use matlabplot library to show the images, first the images the model INCORRECTLY predicted and the second, the CORRECTLY predicted. 

# In[32]:


def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)


# In[43]:


ROWS = 64
COLS = 64
CHANNELS = 3


# In[50]:


X_train2 = X_train.reshape(X_train.shape[0], 64, 64, 3) 
X_test2 = X_test.reshape(X_test.shape[0], 64, 64, 3)


Y_TrainR = Y_train.reshape(-1,1)
Y_testR = Y_test.reshape(-1,1)


# In[52]:


yhat2 = model_cnn.predict_classes(X_test2)
yhat2 = yhat2.reshape(-1,1)  #Need to ensure column vector in order for np.nonzero to work
print(yhat2.shape)
correct_indices = np.nonzero(yhat2 == Y_test)[0]


# In[54]:


correct_indides = np.nonzero(Y_test==yhat2)[0]
incorrect_indides = np.nonzero(Y_test!=yhat2)[0]
print(correct_indides.shape)
print(incorrect_indides.shape)


# In[94]:


plt.figure()
for i, incorrect in enumerate(incorrect_indides[:3]):
    #plt.subplot(3,3,i+1)
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(64,64,3), cmap='Blues', interpolation='none')
    plt.title("Predicted {}, Class {}".format(yhat2[incorrect], Y_test[incorrect]))


# In[96]:


plt.figure()
for i, correct in enumerate(correct_indides[:1]):
    #plt.subplot(3,3,i+1)
    plt.subplot(1,1,i+1)
    plt.imshow(X_test[correct].reshape(64,64,3), cmap='Greens', interpolation='none')
    plt.title("Predicted {}, Class {}".format(yhat2[correct], Y_test[correct]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




