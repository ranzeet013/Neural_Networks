#!/usr/bin/env python
# coding: utf-8

# # LSTM on SineWave 

# In this project, we aim to use a Long Short-Term Memory (LSTM) neural network to predict and generate a sine wave pattern. The LSTM is a type of recurrent neural network (RNN) that is particularly effective in capturing and learning patterns in sequential data.

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
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# # Dataset

# In[2]:


x = np.linspace(0,50,501)
y = np.sin(x)


# In[3]:


x


# In[4]:


y


# In[5]:


plt.plot(x, y)


# In[6]:


dataframe = pd.DataFrame(data=y,index=x,columns=['Sine'])


# In[7]:


dataframe.head()


# In[8]:


dataframe.tail()


# In[9]:


test_percent = 0.1


# In[10]:


len(dataframe)*test_percent


# In[11]:


test_point = np.round(len(dataframe)*test_percent)


# In[12]:


test_point


# In[13]:


test_index = int(len(dataframe) - test_point)


# In[14]:


test_index


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.

# In[15]:


train = dataframe.iloc[:test_index]
test = dataframe.iloc[test_index:]


# In[16]:


train


# In[17]:


test


# # Scaling

# Scaling is a common preprocessing step in data analysis and machine learning. It involves transforming the features of a dataset to a standard scale, which can help improve the performance and stability of models

# MinMaxScaler is a popular scaling technique used in data preprocessing. It scales the features to a specified range, typically between 0 and 1.

# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler = MinMaxScaler()


# In[20]:


scaler.fit(train)


# In[21]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[25]:


scaled_train


# In[26]:


scaled_test


# # Timeseries Generator

# In time series analysis, a common approach is to use a time series generator to generate batches of sequential data for training recurrent neural networks (RNNs) or other time-based models. This allows you to efficiently process and train models on large time series datasets. Here's an example of how you can create a time series generator using the TimeseriesGenerator calss.

# In[22]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[23]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# # EarlyStopping

# Early stopping is a technique commonly used in machine learning projects, including those involving neural networks such as LSTM, to prevent overfitting and determine the optimal number of training iterations. It allows you to monitor the model's performance during training and stop the training process when the model starts to show signs of overfitting or when further training is unlikely to improve the model's performance.

# In[24]:


from tensorflow.keras.callbacks import EarlyStopping


# In[27]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[28]:


length = 49
generator = TimeseriesGenerator(scaled_train,scaled_train,
                               length=length,batch_size=1)


validation_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                          length=length,batch_size=1)


# In[29]:


n_features = 1


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[30]:


model = Sequential()
model.add(LSTM(49, input_shape = (length, n_features)))
model.add(Dense(1))


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[31]:


model.compile(optimizer = 'adam', loss = 'mse')


# # Training The Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[32]:


model.fit_generator(generator, epochs = 10, 
                    validation_data = validation_generator, 
                    callbacks = [early_stop])


# In[33]:


model.save('model_LSTM.h5')


# In[34]:


loss = pd.DataFrame(model.history.history)


# In[35]:


loss.plot()


# # Predicting Values

# In[36]:


prediction = []
evaluation_batch = scaled_train[-length:]
current_batch = evaluation_batch.reshape(1, length, n_features)
for i in range (len(test)):
    current_prediction = model.predict(current_batch)[0]
    prediction.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis = 1)


# In[37]:


prediction = scaler.inverse_transform(prediction)


# In[38]:


prediction


# In[39]:


test['LSTM Prediction'] = prediction


# In[40]:


test.head()


# # Chart Showing Predicted Value And Actual Value

# In[41]:


test.plot(figsize = (10, 4))


# # Predicting New Range

# In[43]:


full_scaler = MinMaxScaler()


# In[44]:


full_data_scale = scaler.transform(dataframe)


# In[45]:


length = 50 
generator = TimeseriesGenerator(full_data_scale, full_data_scale, length=length, batch_size=1)


# In[46]:


model = Sequential()
model.add(LSTM(50, input_shape=(length, n_features)))
model.add(Dense(1))


# In[47]:


model.compile(optimizer='adam', loss='mse')


# In[48]:


model.fit_generator(generator,epochs=6)


# In[55]:


loss = pd.DataFrame(model.history.history)


# In[49]:


prediction = []
evaluation_batch = full_data_scale[-length:]
current_batch = evaluation_batch.reshape(1, length, n_features)
for i in range (len(test)):
    current_prediction = model.predict(current_batch)[0]
    prediction.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis = 1)


# In[50]:


prediction


# In[51]:


prediction = scaler.inverse_transform(prediction)


# In[52]:


prediction


# In[53]:


prediction_index = np.arange(50.1,55.1,step=0.1)


# # New Range Along With Old Range

# In[54]:


plt.plot(dataframe.index,dataframe['Sine'])
plt.plot(prediction_index,prediction)

