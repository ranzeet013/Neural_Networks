#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Classification with CNN

# The aim of this project is to develop a Convolutional Neural Network (CNN) model for classifying heart disease. Heart disease is a significant global health issue, and early detection and accurate classification of different heart conditions can greatly assist in providing timely treatment and improving patient outcomes. CNNs are well-suited for image classification tasks and can effectively learn features from medical imaging data such as electrocardiograms (ECGs) or echocardiograms.

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


# # Importing Dataset

# In[2]:


dataframe = pd.read_csv('heart.csv')


# # Exploratory Analysis

# Before building the CNN model, it is essential to perform exploratory analysis on the dataset to gain insights and understand the underlying patterns and characteristics of the data

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.shape


# In[6]:


dataframe.info()


# In[7]:


dataframe.isna().sum()


# # Statical Info

# Here are some statistical measures and techniques that can be applied during the exploratory analysis of the heart disease classification dataset:

# In[8]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[9]:


corr_matrix = dataframe.corr()


# In[10]:


corr_matrix


# In[11]:


plt.figure(figsize = (18, 8))
sns.heatmap(
    corr_matrix, 
    annot = True, 
    cmap = 'coolwarm'
)


# In[12]:


dataframe.columns


# In[13]:


dataframe.head()


# In[14]:


x = dataframe.drop('target', axis = 1)
y = dataframe['target']


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = 0.2, 
    random_state = 42
)


# In[17]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling 

# Scaling is a preprocessing step in machine learning that involves transforming the features or variables of your dataset to a consistent scale. It is important because many machine learning algorithms are sensitive to the scale of the input features. Scaling helps ensure that all features have a similar range and distribution, which can improve the performance and convergence of the model.
# 
# StandardScaler is a popular scaling technique used in machine learning to standardize features by removing the mean and scaling to unit variance. It is available in the scikit-learn library, which provides a wide range of machine learning tools and preprocessing functions.

# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


scaler = StandardScaler()


# In[20]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[21]:


x_train


# In[22]:


x_test


# In[23]:


x_train.shape, x_test.shape


# # Reshaping

# Reshaping to 1D refers to transforming multidimensional data, such as images, into a 1-dimensional vector representation. In the context of heart disease classification with CNNs, reshaping to 1D is typically performed on the input data to prepare it for feeding into the neural network model.

# In[24]:


x_train = x_train.reshape(820, 13, 1)
x_test = x_test.reshape(205, 13, 1)


# In[25]:


x_train.shape, x_test.shape


# # Building Model

# Creating a deep learning model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[26]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# In[27]:


model = Sequential()
model.add(Conv1D(filters = 32, kernel_size = 2,activation = 'relu', input_shape = (13, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters = 64,kernel_size = 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[28]:


model.summary()


# A model checkpoint, in the context of machine learning, refers to a saved copy of a trained model at a specific point during the training process. It includes the model's architecture, weights, and optimizer state.

# In[29]:


checkpoint_path = "heart_disease_classificaion"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, 
    save_weights_only = True, 
    monitor = 'val_accuracy',
    save_best_only = True
)


# Early stopping is a technique used during the training of machine learning models to prevent overfitting and find the optimal point at which to stop training. It involves monitoring the performance of the model on a validation dataset and stopping the training process when the model's performance on the validation dataset starts to degrade.

# In[30]:


early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True
)


# ReduceLROnPlateau" is a learning rate scheduling technique commonly used in the training of deep learning models. It stands for "Reduce Learning Rate on Plateau" and is used to dynamically adjust the learning rate during training to optimize model performance.

# In[31]:


reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.2,
    patience = 3,
    min_le = 1e-10
)


# In[32]:


from tensorflow.keras.callbacks import TensorBoard
import datetime

def create_tensorboard_callback(log_dir, experiment_name):
    log_dir = log_dir + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# # Compiling Model

# Compiling the model in deep learning involves configuring essential components that define how the model will be trained.

# In[33]:


model.compile(optimizer = Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance.

# In[34]:


model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 250, 
    callbacks = [early_stop, 
                 create_tensorboard_callback("training log",
                                            "heart_disease_classification"),
                 checkpoint_callback, 
                 reduce_lr]
)


# In[35]:


model.save('model_heart_disease_classification.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[36]:


loss = pd.DataFrame(model.history.history)


# In[37]:


loss.head()


# In[38]:


loss.plot()


# In[39]:


loss[['loss', 'val_loss']].plot()


# In[40]:


loss[['accuracy', 'val_accuracy']].plot()


# # Prediction 

# In[41]:


y_pred = model.predict(x_test)
predict_class = y_pred.argmax(axis = 1)


# In[42]:


print(y_test.iloc[4]), print(predict_class[4])


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# A confusion matrix is a tabular representation that visualizes the performance of a classification model by displaying the counts of true positive, true negative, false positive, and false negative predictions. It helps assess the effectiveness of the model's predictions and identify any patterns of misclassifications.

# In[43]:


from sklearn.metrics import confusion_matrix


# In[44]:


confusion_matrix = confusion_matrix(y_test, predict_class)


# In[45]:


confusion_matrix

