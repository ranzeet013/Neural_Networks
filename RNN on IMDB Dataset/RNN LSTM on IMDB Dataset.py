#!/usr/bin/env python
# coding: utf-8

# # RNN LSTM on IMDB Dataset

# To building an LSTM-based recurrent neural network (RNN) model for sentiment analysis on the IMDB dataset using Python and Keras, which is a popular deep learning library. Sentiment analysis involves classifying text as positive or negative based on its sentiment.
# 
# 0 For Negative Review
# 
# 1 For Positive Review

# # Importing Libraries

# Importing libraries is an essential step in any data analysis or machine learning project. These libraries provide various functions and tools to manipulate, visualize, and analyze data efficiently. Here are explanations of some popular data analysis libraries:

# Pandas: Pandas is a powerful and widely used library for data manipulation and analysis. It provides data structures like DataFrames and Series, which allow you to store and manipulate tabular data. Pandas offers a wide range of functions for data cleaning, filtering, aggregation, merging, and more

# NumPy: NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides efficient data structures like arrays and matrices and a vast collection of mathematical functions. NumPy enables you to perform various numerical operations on large datasets, such as element-wise calculations, linear algebra, Fourier transforms, and random number generation

# Matplotlib: Matplotlib is a popular plotting library that enables you to create a wide range of static, animated, and interactive visualizations. It provides a MATLAB-like interface and supports various types of plots, including line plots, scatter plots, bar plots, histograms, and more

# TensorFlow is an open-source deep learning framework developed by Google. It provides a comprehensive ecosystem of tools, libraries, and resources for building and deploying machine learning models. TensorFlow is widely used in various domains, including computer vision, natural language processing, and reinforcement learning

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset From keras.datasets

# In[2]:


import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.

# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[3]:


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 20000)


# In[4]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[5]:


x_train


# In[6]:


x_test


# In[8]:


x_train = pad_sequences(x_train, maxlen = 100)
x_test = pad_sequences(x_test, maxlen = 100)


# In[9]:


x_train


# In[10]:


x_test


# In[11]:


x_train.shape, x_test.shape


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping


# In[15]:


model = Sequential()

model.add(Embedding(input_dim=20000, output_dim=128, input_shape=(100,)))

model.add(LSTM(128, activation = 'tanh'))

model.add(Dense(1, activation = 'sigmoid'))


# In[16]:


model.summary()


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[17]:


model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Early stopping is a technique used in machine learning to prevent overfitting and improve the generalization ability of a model. It involves monitoring the performance of a model during training and stopping the training process when the performance on a validation set starts to deteriorate.

# In[19]:


early_stop = EarlyStopping(monitor = 'val_loss', patience = 3)


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance.

# In[21]:


model.fit(x_train, y_train,
          batch_size = 128, 
          epochs = 20, 
          validation_data = (x_test, y_test), 
          callbacks = [early_stop])


# In[22]:


model.save('LSTM_model_for_IMDB.h5')


# # Learning Curve

# A learning curve is a graphical representation that shows how the performance of a machine learning model improves or stabilizes as the amount of training data increases. It is a useful tool for evaluating the effectiveness of a model and understanding its behavior.

# In[23]:


loss = pd.DataFrame(model.history.history)


# In[24]:


loss


# In[25]:


loss.plot()


# In[26]:


loss[['val_loss', 'loss']].plot()


# In[27]:


loss[['val_accuracy', 'accuracy']].plot()


# # Prediction 

# In[28]:


y_pred = model.predict(x_test)


# In[31]:


print(y_test[5]), print(y_pred[5])


# In[32]:


print(y_test[50]), print(y_pred[50])

