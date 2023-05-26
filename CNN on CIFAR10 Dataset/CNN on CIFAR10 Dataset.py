#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:

# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Dataset

# In[2]:


from tensorflow.keras.datasets import cifar10


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.

# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[3]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[5]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[6]:


plt.imshow(x_train[20])


# In[7]:


x_train[20]


# In[8]:


x_train.min(), x_train.max()


# In[9]:


x_train = x_train / 255
x_test = x_test / 255


# In[10]:


x_train.shape, x_test.shape


# # Converting To Categorical

# Converting data to categorical format is often necessary for certain machine learning tasks, especially when dealing with categorical or nominal variables. This conversion is typically performed to represent the categorical variables as numeric values that can be processed by machine learning algorithms.

# In[11]:


from tensorflow.keras.utils import to_categorical


# In[12]:


y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)


# In[13]:


y_train_categorical


# In[14]:


y_test_categorical


# # Buliding Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[16]:


from tensorflow.keras.callbacks import EarlyStopping


# In[17]:


from sklearn.metrics import classification_report, confusion_matrix


# In[18]:


model = Sequential()
model.add(Conv2D(filters = 32, input_shape = (32, 32, 3), kernel_size = (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(filters = 32, input_shape = (32, 32, 3), kernel_size = (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))


# In[19]:


model.summary()


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[20]:


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[21]:


early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[23]:


model.fit(x_train, y_train_categorical,
          validation_data = (x_test, y_test_categorical),
          epochs = 20, 
          callbacks = [early_stopping])


# In[24]:


model.save('CNN_model_for_CIFAR10.h5')


# In[25]:


loss = pd.DataFrame(model.history.history)


# In[26]:


loss.head()


# In[27]:


loss.plot()


# In[28]:


loss[['loss', 'val_loss']].plot()


# In[29]:


loss[['accuracy', 'val_accuracy']].plot()


# In[34]:


prediction = model.predict(x_test)
prediction_classes = prediction.argmax(axis = 1)


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during predictions to gain insights into the patterns, sources, and potential improvements.

# In[35]:


print(classification_report(y_test, prediction_classes))


# In[37]:


print(confusion_matrix(prediction_classes, y_test))


# In[41]:


plt.figure(figsize = (12, 6))
sns.heatmap(confusion_matrix(prediction_classes, y_test),
            annot = True, 
            cmap = 'coolwarm')


# # Prediction

# In[57]:


prediction = x_train[5]


# In[58]:


plt.imshow(prediction)


# In[59]:


predictions = model.predict(prediction.reshape(1, 32, 32, 3))
predicted_class = np.argmax(predictions)

print("Predicted class:", predicted_class)

