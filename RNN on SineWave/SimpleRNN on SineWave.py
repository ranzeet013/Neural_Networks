#!/usr/bin/env python
# coding: utf-8

# # SimpleRNN on SineWave

# The goal of this project is to implement a simple Recurrent Neural Network (RNN) model to predict the values of a sine wave. The RNN will be trained on a sequence of sine wave data points and learn to predict the next value in the sequence.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:

# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Making DataFrame

# In[3]:


x = np.linspace(0,50,501)
y = np.sin(x)


# In[4]:


x


# In[5]:


y


# In[6]:


plt.plot(x, y)


# In[7]:


dataframe = pd.DataFrame(data = y, index = x, columns = ['Sine'])


# In[8]:


dataframe.head()


# In[9]:


dataframe.tail()


# In[10]:


test_percent = 0.1


# In[12]:


test_point = np.round(len(dataframe)*test_percent)


# In[13]:


test_index = int(len(dataframe) - test_point)


# In[14]:


test_index


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.

# In[15]:


train = dataframe.iloc[:test_index]
test = dataframe.iloc[test_index:]


# # Scaling

# Scaling is a common preprocessing step in data analysis and machine learning. It involves transforming the features of a dataset to a standard scale, which can help improve the performance and stability of models

# MinMaxScaler is a popular scaling technique used in data preprocessing. It scales the features to a specified range, typically between 0 and 1. 

# In[16]:


from sklearn.preprocessing import MinMaxScaler


# In[17]:


scaler = MinMaxScaler()


# In[18]:


scaler.fit(train)


# In[19]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[20]:


scaled_train


# In[21]:


scaled_test


# # Timeseries Generator

# In time series analysis, a common approach is to use a time series generator to generate batches of sequential data for training recurrent neural networks (RNNs) or other time-based models. This allows you to efficiently process and train models on large time series datasets. Here's an example of how you can create a time series generator using the TimeseriesGenerator calss.

# In[22]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[23]:


length = 50 
batch_size = 1 
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[26]:


n_features = 1


# In[27]:


model = Sequential()
model.add(SimpleRNN(50, input_shape =(length, n_features)))
model.add(Dense(1))


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[29]:


model.compile(optimizer = 'adam', loss = 'mse')


# In[30]:


model.summary()


# # Training The Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[31]:


model.fit_generator(generator, epochs = 10)


# In[34]:


model.save('model_simpleRNN.h5')


# In[35]:


loss = pd.DataFrame(model.history.history)


# In[36]:


loss.plot()


# # Predicting Values 

# In[38]:


prediction = []
evaluation_batch = scaled_train[-length:]
current_batch = evaluation_batch.reshape(1, length, n_features)
for i in range(len(test)):
    current_prediction = model.predict(current_batch)[0]
    prediction.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis = 1)


# In[39]:


prediction


# In[40]:


prediction = scaler.inverse_transform(prediction)


# In[41]:


prediction


# In[42]:


test['Prediction'] = prediction


# In[43]:


test


# # Chart Showing Predicted Value And Actual Value

# In[46]:


test.plot(figsize = (10, 5))

