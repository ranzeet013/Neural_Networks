#!/usr/bin/env python
# coding: utf-8

# # LSTM on Cosine Wave

# The goal of this project is to train an LSTM (Long Short-Term Memory) model to predict the next values in a cosine wave time series. The LSTM model will learn to capture the underlying patterns and dependencies in the data and generate accurate predictions.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
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


# # Generating Data Sequence

# To generate a data sequence for the cosine wave, you can use the numpy library in Python. 

# In[4]:


x = np.linspace(0, 50, 501)
y = np.cos(x)


# In[5]:


x


# In[6]:


y


# # Data Visualization

# 
# To visualize the generated data sequence representing a cosine wave, you can use the matplotlib library in Python. 

# In[19]:


plt.plot(x, y)


# In[7]:


x.shape, y.shape


# # Creating Dataframe

# In[9]:


dataframe = pd.DataFrame(data = y, 
                         index = x, 
                         columns = ['Cosine'])


# In[10]:


dataframe.head()


# In[11]:


dataframe.tail()


# Calculating the test index for a given dataset taking the test percentage is 0.1 (10%), we can perform the following ways:

# In[12]:


test_precent = 0.1


# In[13]:


len(dataframe)*test_precent


# In[15]:


test_point = np.round(len(dataframe)*test_precent)


# In[16]:


test_point


# In[17]:


test_index = int(len(dataframe) - test_point)


# In[18]:


test_index


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[20]:


train_data = dataframe.iloc[:test_index]
test_data = dataframe.iloc[test_index:]


# In[21]:


train_data


# In[22]:


test_data


# In[23]:


train_data.shape, test_data.shape


# # Scaling Dataset

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.

# MinMaxScaler is a popular scaling technique used in data preprocessing. It scales the features to a specified range, typically between 0 and 1.

# In[24]:


from sklearn.preprocessing import MinMaxScaler


# In[25]:


scaler = MinMaxScaler()


# In[26]:


scaler.fit(train_data)


# In[27]:


scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)


# In[28]:


scaled_train


# In[29]:


scaled_test


# # Timeseries Generator

# In time series analysis, a common approach is to use a time series generator to generate batches of sequential data for training recurrent neural networks (RNNs) or other time-based models. This allows you to efficiently process and train models on large time series datasets. Here's an example of how you can create a time series generator using the TimeseriesGenerator calss.

# In[30]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[32]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping


# Early stopping is a technique commonly used in machine learning projects, including those involving neural networks such as LSTM, to prevent overfitting and determine the optimal number of training iterations. It allows you to monitor the model's performance during training and stop the training process when the model starts to show signs of overfitting or when further training is unlikely to improve the model's performance.

# In[33]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[34]:


length = 49
generator = TimeseriesGenerator(scaled_train,scaled_train,
                               length=length,batch_size=1)


validation_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                          length=length,batch_size=1)


# In[35]:


n_features = 1


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[36]:


model = Sequential()
model.add(LSTM(49, input_shape = (length, n_features)))
model.add(Dense(1))


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[37]:


model.compile(optimizer = 'adam', loss = 'mse')


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance

# In[38]:


model.fit_generator(generator, epochs = 10, 
                    validation_data = validation_generator, 
                    callbacks = [early_stop])


# In[39]:


model.save('model_cosine_wave.h5')


# # Learning Curve for x-Test Data

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[40]:


loss = pd.DataFrame(model.history.history)


# In[41]:


loss.head()


# In[42]:


loss.plot()


# # Predicting On x-Test Data

# In[44]:


prediction = []
evaluation_batch = scaled_train[-length:]
current_batch = evaluation_batch.reshape(1, length, n_features)
for i in range (len(test_data)):
    current_prediction = model.predict(current_batch)[0]
    prediction.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis = 1)


# In[45]:


prediction = scaler.inverse_transform(prediction)


# In[46]:


prediction


# In[47]:


test_data['Prediction'] = prediction


# In[48]:


test_data.head()


# # Predicted x-Test Value And Actual Value

# In[49]:


test_data.plot(figsize = (12, 5))


# # Predicting For Full Dataframe

# In[50]:


full_scaler = MinMaxScaler()


# In[51]:


full_data_scale = scaler.transform(dataframe)
length = 50 
generator = TimeseriesGenerator(full_data_scale, full_data_scale, length=length, batch_size=1)
model = Sequential()
model.add(LSTM(50, input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator,epochs=6)


# In[52]:


model.save('model_full_cosine.h5')


# In[53]:


losses = pd.DataFrame(model.history.history)


# In[54]:


losses.head()


# In[55]:


losses.plot()


# In[57]:


prediction = []
evaluation_batch = full_data_scale[-length:]
current_batch = evaluation_batch.reshape(1, length, n_features)
for i in range (len(test_data)):
    current_prediction = model.predict(current_batch)[0]
    prediction.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis = 1)


# In[58]:


prediction = scaler.inverse_transform(prediction)


# In[60]:


prediction


# # Predicted Value Along With Actual Value

# In[61]:


prediction_index = np.arange(50.1,55.1,step=0.1)


# In[63]:


plt.plot(dataframe.index,dataframe['Cosine'])
plt.plot(prediction_index,prediction)


# # Thanks !
