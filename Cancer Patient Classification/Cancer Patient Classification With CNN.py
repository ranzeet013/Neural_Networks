#!/usr/bin/env python
# coding: utf-8

# # Cancer Patient Classification With CNN

# The "Cancer Patient Classification with Convolutional Neural Networks (CNN) Project" is a research endeavor aimed at revolutionizing cancer diagnosis and treatment through the application of cutting-edge deep learning techniques. Leveraging the power of CNNs, this project seeks to develop a robust and accurate model capable of classifying cancer patients based on medical imaging data.

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


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('Cancer_Data.csv')


# # Exploratory Data Analysis

# Before building the CNN model, it is essential to perform exploratory analysis on the dataset to gain insights and understand the underlying patterns and characteristics of the data

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.columns


# In[7]:


dataframe = dataframe.drop(['Unnamed: 32', 'id'], axis = 1)


# In[8]:


dataframe.head()


# # Encoding 

# Encoding is the process of converting data or information from one format or representation to another, typically to make it more suitable for a specific purpose, such as storage, transmission, or processing. 

# Label Encoding is a data preprocessing technique that converts categorical variables into numerical values by assigning a unique integer to each category. It facilitates the use of machine learning algorithms with textual data but may introduce an implicit order between labels, which should be considered when dealing with non-ordinal variables.

# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


encoder = LabelEncoder()


# In[11]:


dataframe['diagnosis'] = encoder.fit_transform(dataframe['diagnosis'])


# In[12]:


dataframe.head()


# In[13]:


dataframe.info()


# # Checking Null Values

# In[14]:


dataframe.isna().sum()


# # Distribution Of Cancer Diagnosis

# In[15]:


dataframe['diagnosis'].value_counts()


# In[16]:


dataframe['diagnosis'].value_counts().plot(kind = 'bar', 
                                           title = 'Distribution Of Cnacer Diagnosis', 
                                           figsize = (8, 3), 
                                           cmap = 'ocean') 


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[17]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[18]:


corr_matrix = dataframe.corr()


# In[19]:


corr_matrix


# In[20]:


plt.figure(figsize = (18, 8))
sns.heatmap(
    corr_matrix, 
    annot = True, 
    cmap = 'coolwarm'
)


# In[21]:


dataframe.columns


# In[22]:


dataset = dataframe.drop('diagnosis', axis = 1)


# In[23]:


dataset.head()


# # Correlation Of Cancer Diagnosis With Different Attibutes

# In[24]:


dataset.corrwith(dataframe['diagnosis']).plot.bar(
    figsize = (12, 5), 
    title = 'Correlation Of Cance Diagnosis', 
    cmap = 'plasma'
)


# In[25]:


dataframe.head()


# # Splitting Dataset

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[26]:


x = dataframe.drop('diagnosis', axis = 1)
y = dataframe['diagnosis']


# In[27]:


x.shape, y.shape


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = 0.2, 
    random_state = 42
)


# In[30]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling 

# Scaling is a preprocessing step in machine learning that involves transforming the features or variables of your dataset to a consistent scale. It is important because many machine learning algorithms are sensitive to the scale of the input features. Scaling helps ensure that all features have a similar range and distribution, which can improve the performance and convergence of the model.
# 
# StandardScaler is a popular scaling technique used in machine learning to standardize features by removing the mean and scaling to unit variance. It is available in the scikit-learn library, which provides a wide range of machine learning tools and preprocessing functions.

# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


scaler = StandardScaler()


# In[33]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[34]:


x_train


# In[35]:


x_test


# # Reshaping 

# Reshaping to 1D refers to transforming multidimensional data, such as images, into a 1-dimensional vector representation. In the context of heart disease classification with CNNs, reshaping to 1D is typically performed on the input data to prepare it for feeding into the neural network model.

# In[36]:


x_train.shape, x_test.shape


# In[37]:


x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)


# In[38]:


x_train.shape, x_test.shape


# # Building Model

# Creating a CNN model involves defining the architecture and structure of the neural network, specifying the layers, and configuring the parameters for training.

# In[39]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# In[40]:


model = Sequential()
model.add(Conv1D(filters = 32, kernel_size = 4,activation = 'relu', input_shape = (30, 1)))
model.add(MaxPooling1D(pool_size = 3))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters = 64,kernel_size = 3, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 3))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[41]:


model.summary()


# A model checkpoint, in the context of machine learning, refers to a saved copy of a trained model at a specific point during the training process. It includes the model's architecture, weights, and optimizer state.

# In[42]:


checkpoint_path = "cancer_patient_classificaion"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, 
    save_weights_only = True, 
    monitor = 'val_accuracy',
    save_best_only = True
)


# Early stopping is a technique used during the training of machine learning models to prevent overfitting and find the optimal point at which to stop training. It involves monitoring the performance of the model on a validation dataset and stopping the training process when the model's performance on the validation dataset starts to degrade.

# In[43]:


early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True
)


# ReduceLROnPlateau" is a learning rate scheduling technique commonly used in the training of deep learning models. It stands for "Reduce Learning Rate on Plateau" and is used to dynamically adjust the learning rate during training to optimize model performance.

# In[44]:


reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.2,
    patience = 3,
    min_le = 1e-10
)


# In[45]:


from tensorflow.keras.callbacks import TensorBoard
import datetime

def create_tensorboard_callback(log_dir, experiment_name):
    log_dir = log_dir + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# In[46]:


model.compile(optimizer = Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training Model

# Training the model in deep learning involves the process of iteratively updating the model's parameters (weights and biases) based on the provided training data to minimize the loss function and improve the model's performance.

# In[47]:


model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 250, 
    callbacks = [early_stop, 
                 create_tensorboard_callback("training log",
                                             "cancer_patient_classification"),
                 checkpoint_callback, 
                 reduce_lr]
)


# In[48]:


model.save('cancer_patient_classifier_model.h5')


# # Learning Curve

# The learning curve is a plot that shows how the loss and accuracy of a model change during training. It provides insights into how well the model is learning from the training data and how it generalizes to unseen data. The learning curve typically shows the training and validation loss/accuracy on the y-axis and the number of epochs on the x-axis. By analyzing the learning curve, you can identify if the model is overfitting (high training loss, low validation loss) or underfitting (high training and validation loss). It is a useful tool for monitoring and evaluating the performance of machine learning models.

# In[49]:


losses = pd.DataFrame(model.history.history)


# In[50]:


losses.head()


# In[51]:


losses.plot()


# # Loss Curve

# Loss curve is a graphical representation that shows how the loss of a machine learning model changes over the course of training. Loss refers to the discrepancy between the predicted output of the model and the true or expected output. The loss curve helps in monitoring the progress of model training and assessing the convergence and performance of the model.

# In[52]:


losses[['loss', 'val_loss']].plot()


# # Accuracy Curve

# An accuracy curve is a graphical representation that depicts how the accuracy of a machine learning model changes as a specific aspect of the model or training process varies. The accuracy curve is typically plotted against the varying parameter or condition to analyze its impact on the model's accuracy

# In[53]:


losses[['accuracy', 'val_accuracy']].plot()


# In[54]:


y_pred = model.predict(x_test)
predict_class = y_pred.argmax(axis = 1)


# In[55]:


print(y_test.iloc[44]), print(predict_class[44])


# In[56]:


print(y_test.iloc[78]), print(predict_class[78])


# # Error Analysis

# Error analysis is a crucial step in the evaluation and improvement of machine learning models. It involves the systematic examination and understanding of the errors made by the model during prediction. The primary goal of error analysis is to identify patterns and sources of mistakes made by the model, which can provide valuable insights into its performance and guide improvements.

# In[57]:


from sklearn.metrics import confusion_matrix


# In[58]:


confusion_matrix = confusion_matrix(y_test, predict_class)


# In[59]:


confusion_matrix


# In[60]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')


# # Thanks !
