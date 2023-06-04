#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction with CNN

# A breast cancer prediction project aims to develop a model or system that can accurately predict the likelihood of a person having breast cancer based on certain factors or indicators. The project typically involves using machine learning algorithms and data analysis techniques to analyze a dataset containing information about individuals, including their medical history, genetic markers, lifestyle factors, and any previous instances of breast cancer.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset From sklearn.datasets

# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer


# In[5]:


print(cancer.DESCR)


# In[6]:


x = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)


# # Data Exploration

# Data exploration is a crucial step in any data analysis or machine learning project, including a breast cancer prediction project. It involves examining and understanding the characteristics, patterns, and relationships within the dataset to gain insights and inform subsequent analysis

# In[7]:


x.head()


# In[8]:


x.tail()


# In[9]:


y = cancer.target


# In[10]:


y


# In[11]:


x.info()


# In[12]:


x.isna().sum()


# # Statical Info

# In[13]:


x.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[14]:


corr_matrix = x.corr()


# In[15]:


corr_matrix


# In[16]:


plt.figure(figsize = (25, 15))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'coolwarm')


# In[17]:


x.shape, y.shape


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                   y, 
                                                   test_size = 0.2, 
                                                   random_state = 101)


# In[20]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling 

# Scaling is a preprocessing step in machine learning that involves transforming the features or variables of your dataset to a consistent scale. It is important because many machine learning algorithms are sensitive to the scale of the input features. Scaling helps ensure that all features have a similar range and distribution, which can improve the performance and convergence of the model.
# 
# StandardScaler is a popular scaling technique used in machine learning to standardize features by removing the mean and scaling to unit variance. It is available in the scikit-learn library, which provides a wide range of machine learning tools and preprocessing functions.

# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


scaler = StandardScaler()


# In[23]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[24]:


x_train


# In[25]:


x_test


# In[26]:


x_train.shape, x_test.shape


# In[27]:


x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)


# In[28]:


x_train.shape, x_test.shape


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[29]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# In[30]:


model = Sequential()
model.add(Conv1D(filters = 32, kernel_size = 2, activation = 'relu', input_shape = (30, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters = 64, kernel_size = 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[31]:


model.summary()


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[32]:


model.compile(optimizer = Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# A model checkpoint, in the context of machine learning, refers to a saved copy of a trained model at a specific point during the training process. It includes the model's architecture, weights, and optimizer state.

# In[33]:


checkpoint_path = "breast_cancer_classification"
checkpoint_callback = ModelCheckpoint(checkpoint_path, 
                                      save_weights_only = True, 
                                      monitor = 'val_accuracy', 
                                      save_best_only = True)


# Early stopping is a technique used during the training of machine learning models to prevent overfitting and find the optimal point at which to stop training. It involves monitoring the performance of the model on a validation dataset and stopping the training process when the model's performance on the validation dataset starts to degrade.

# In[34]:


early_stop = EarlyStopping(monitor = 'val_loss', 
                           patience = 5, 
                           restore_best_weights = True)


# "ReduceLROnPlateau" is a learning rate scheduling technique commonly used in the training of deep learning models. It stands for "Reduce Learning Rate on Plateau" and is used to dynamically adjust the learning rate during training to optimize model performance.

# In[36]:


reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                              factor = 0.2, 
                              patience = 3, 
                              min_lr = 1e-10)


# In[37]:


from tensorflow.keras.callbacks import TensorBoard
import datetime

def create_tensorboard_callback(log_dir, experiment_name):
    log_dir = log_dir + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance.

# In[40]:


model.fit(x_train, y_train, 
          validation_data = (x_test, y_test), 
          epochs = 100, 
          callbacks = [early_stop, 
                       create_tensorboard_callback("training logd", 
                                                  "breast_cancer_classification"), 
                       checkpoint_callback, 
                       reduce_lr])


# In[41]:


model.save('model_breast_cancer.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[42]:


loss = pd.DataFrame(model.history.history)


# In[43]:


loss.head()


# In[44]:


loss.plot()


# In[45]:


loss[['loss', 'val_loss']].plot()


# In[46]:


loss[['accuracy', 'val_accuracy']].plot()


# # Prediction

# In[47]:


y_pred = model.predict(x_test)


# In[48]:


print(y_test[20]), print(y_pred[20])


# In[49]:


predict_calss = y_pred.argmax(axis = 1)


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# In[51]:


from sklearn.metrics import accuracy_score, confusion_matrix


# A confusion matrix is a tabular representation that summarizes the performance of a classification model by showing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.

# In[52]:


confusion_matrix = confusion_matrix(y_test, predict_calss)


# In[53]:


confusion_matrix


# In[54]:


print(confusion_matrix)

