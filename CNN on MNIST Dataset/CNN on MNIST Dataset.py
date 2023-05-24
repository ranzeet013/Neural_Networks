#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# Importing libraries is an essential step in any data analysis or machine learning project. These libraries provide various functions and tools to manipulate, visualize, and analyze data efficiently. Here are explanations of some popular data analysis libraries

# Pandas: Pandas is a powerful and widely used library for data manipulation and analysis. It provides data structures like DataFrames and Series, which allow you to store and manipulate tabular data. Pandas offers a wide range of functions for data cleaning, filtering, aggregation, merging, and more. It also supports reading and writing data from various file formats such as CSV, Excel, SQL databases, and more.

# NumPy: NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides efficient data structures like arrays and matrices and a vast collection of mathematical functions. NumPy enables you to perform various numerical operations on large datasets, such as element-wise calculations, linear algebra, Fourier transforms, and random number generation. It also integrates well with other libraries for data analysis and machine learning.

# Matplotlib: Matplotlib is a popular plotting library that enables you to create a wide range of static, animated, and interactive visualizations. It provides a MATLAB-like interface and supports various types of plots, including line plots, scatter plots, bar plots, histograms, and more. Matplotlib gives you extensive control over plot customization, including labels, colors, legends, and annotations, allowing you to effectively communicate insights from your data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset From keras.datasets

# In[2]:


from tensorflow.keras.datasets import mnist


# # Splitting Dataset

# Splitting the data refers to dividing the dataset into separate subsets for training, validation, and testing purposes. This division is essential to assess the performance of a machine learning model on unseen data and prevent overfitting. Here are the common types of data splits:

# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[3]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[5]:


plt.imshow(x_train[4])


# # Converting To Categorical

# Converting data to categorical format is often necessary for certain machine learning tasks, especially when dealing with categorical or nominal variables. This conversion is typically performed to represent the categorical variables as numeric values that can be processed by machine learning algorithms.

# In[6]:


from tensorflow.keras.utils import to_categorical


# In[7]:


y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)


# In[8]:


x_train.max()


# In[9]:


x_train = x_train / 255
x_test = x_test / 255


# In[10]:


x_train.shape


# In[11]:


x_test.shape


# # Reshaping Images

# Reshaping data is a common operation in machine learning when you need to adjust the dimensions or structure of your data to meet the requirements of a particular algorithm or model. 

# In[12]:


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# In[13]:


x_train.shape, x_test.shape


# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix


# In[15]:


from tensorflow.keras.callbacks import EarlyStopping


# # Creating Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[16]:


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (4, 4), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[17]:


model.summary()


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[18]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[19]:


early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[20]:


model.fit(x_train,
          y_train_categorical, 
          validation_data = (x_test, y_test_categorical),
          epochs = 20,
          callbacks = [early_stopping])


# In[21]:


model.save('CNN_model_for_MNIST.h5')


# In[22]:


loss = pd.DataFrame(model.history.history)


# In[23]:


loss.head()


# In[24]:


loss.plot()


# In[25]:


loss[['loss', 'val_loss']].plot()


# In[26]:


loss[['accuracy', 'val_accuracy']].plot()


# In[29]:


predictions = model.predict(x_test)
predicted_classes = predictions.argmax(axis=1)


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during predictions to gain insights into the patterns, sources, and potential improvements.

# In[31]:


print(classification_report(y_test,predicted_classes))


# In[32]:


confusion_matrix(y_test,predicted_classes)


# In[33]:


plt.figure(figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test,predicted_classes),annot=True)


# # Predicting 

# In[34]:


test_prediction = x_test[5]


# In[35]:


plt.imshow(test_prediction.reshape(28,28))


# In[38]:


predictions = model.predict(test_prediction.reshape(1, 28, 28, 1))
predicted_class = np.argmax(predictions)

print("Predicted class:", predicted_class)

