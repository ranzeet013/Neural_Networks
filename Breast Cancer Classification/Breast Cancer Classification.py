#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np


# In[57]:


df = pd.read_csv('../DATA/cancer_classification.csv')


# In[58]:


df.info()


# In[59]:


df.describe().transpose()


# ## EDA

# In[60]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[62]:


sns.countplot(x='benign_0__mal_1',data=df)


# In[63]:


sns.heatmap(df.corr())


# In[66]:


df.corr()['benign_0__mal_1'].sort_values()


# In[68]:


df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')


# In[70]:


df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# ## Train Test Split

# In[73]:


X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values


# In[74]:


from sklearn.model_selection import train_test_split


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)


# 
# ## Scaling Data

# In[77]:


from sklearn.preprocessing import MinMaxScaler


# In[78]:


scaler = MinMaxScaler()


# In[79]:


scaler.fit(X_train)


# In[80]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Creating the Model
# 
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#                   
#     

# In[98]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout


# In[99]:


X_train.shape


# In[111]:


model = Sequential()


model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))


model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# ## Training the Model 
# 

# In[112]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1
          )


# In[114]:


model_loss = pd.DataFrame(model.history.history)


# In[116]:


model_loss.plot()


# ## Example Two: Early Stopping
# 

# In[117]:


model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[119]:


from tensorflow.keras.callbacks import EarlyStopping


# Stop training when a monitored quantity has stopped improving.
# 
#     Arguments:
#         monitor: Quantity to be monitored.
#         min_delta: Minimum change in the monitored quantity
#             to qualify as an improvement, i.e. an absolute
#             change of less than min_delta, will count as no
#             improvement.
#         patience: Number of epochs with no improvement
#             after which training will be stopped.
#         verbose: verbosity mode.
#         mode: One of `{"auto", "min", "max"}`. In `min` mode,
#             training will stop when the quantity
#             monitored has stopped decreasing; in `max`
#             mode it will stop when the quantity
#             monitored has stopped increasing; in `auto`
#             mode, the direction is automatically inferred
#             from the name of the monitored quantity.

# In[121]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[122]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[124]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# ## Example Three: Adding in DropOut Layers

# In[125]:


from tensorflow.keras.layers import Dropout


# In[126]:


model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[127]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[128]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# # Model Evaluation

# In[129]:


predictions = model.predict_classes(X_test)


# In[130]:


from sklearn.metrics import classification_report,confusion_matrix


# In[131]:


print(classification_report(y_test,predictions))


# In[133]:


print(confusion_matrix(y_test,predictions))

