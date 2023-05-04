#!/usr/bin/env python
# coding: utf-8

# # Multiclass Image Classifiation with ANN

# Multiclass image classification is a type of machine learning problem where the goal is to classify an image into one of several possible classes. For example, given an image of a fruit, the task might be to classify the fruit as an apple, banana, or orange.

# # Importing The Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import tensorflow as tf


# In[3]:


from tensorflow.keras.datasets import fashion_mnist


# In[4]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[5]:


x_train.shape, x_test.shape


# In[6]:


y_train.shape, y_test.shape


# In[7]:


x_train


# In[8]:


np.max(x_train)


# In[10]:


np.min(x_train), np.mean(x_train)


# In[11]:


y_train


# In[12]:


np.max(y_train), np.min(y_train)


# In[14]:


class_names = ['0 Top]T-shirt', '1 Trouser', '2 Pullover', '3 Dress', '4 Coat', '5 Sandal', '6 Shirt', '7 Sneaker', '8 Bag', '9 Ankle boot']
print(class_names)


# # Data Exploration

# Data exploration is the process of getting to know the data you are working with in a machine learning project. It involves performing various types of analyses and visualizations to gain insights into the structure, quality, and characteristics of the data. The goal of data exploration is to identify any issues or patterns in the data that may need to be addressed before building a machine learning model.

# In[16]:


plt.figure()
plt.imshow(x_train[5])


# # Normalizing The Data

# Normalizing the data refers to the process of scaling the input data so that all features have a similar scale and distribution. This is often done as a preprocessing step before feeding the data into a machine learning model.

# Normalization is important because features that are on different scales can have a disproportionate impact on the model's training process and can make it difficult for the model to learn meaningful patterns and relationships. For example, if one feature has values ranging from 0 to 100, and another feature has values ranging from 0 to 0.1, the second feature will have much less impact on the model's training process.

# In[17]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[18]:


plt.figure()
plt.imshow(x_train[20])
plt.colorbar()


# # Flattening The DataSet

# Flattening the data refers to the process of converting multi-dimensional data (such as images or tensors) into a one-dimensional array or vector. This is often done as a preprocessing step before feeding the data into a machine learning model.

# By flattening the input data, we can more easily process it in a neural network and extract meaningful patterns and relationships.

# In[19]:


x_train.shape, x_test.shape


# In[20]:


x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


# In[21]:


x_train.shape, x_test.shape


# # Building The Model

# In[22]:


model = tf.keras.models.Sequential()


# In[24]:


model.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape = (784, )))


# In[25]:


model.add(tf.keras.layers.Dropout(0.3))


# In[26]:


model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))


# # Training The Model

# To train a machine learning model, we need to define the model architecture and compile it with an optimizer, a loss function, and one or more evaluation metrics. We also need to prepare the training data by preprocessing it and splitting it into batches.

# In[28]:


model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = 'sparse_categorical_accuracy')


# In[29]:


model.summary()


# In[30]:


model.fit(x_train, y_train, epochs = 10)


# In[31]:


test_loss, test_accuracy = model.evaluate(x_test, y_test)


# In[32]:


print('Accuracy: .{}'.format(test_accuracy))


# In[36]:


y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)


# In[37]:


print(y_pred)


# In[38]:


y_pred[100], y_test[100] #predicted and actual output


# In[39]:


print(class_names)


# In[41]:


from sklearn.metrics import confusion_matrix, accuracy_score


# A confusion matrix is a table that is often used to evaluate the performance of a classification model. It shows the number of true positives, true negatives, false positives, and false negatives for each class in the predicted versus actual classification.

# In[42]:


confusion_matrix = confusion_matrix(y_pred, y_test)


# In[43]:


confusion_matrix


# In[44]:


confusion_matrix_accuracy = accuracy_score(y_test, y_pred)


# In[45]:


confusion_matrix_accuracy


# In[ ]:




