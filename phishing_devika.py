#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten


# In[2]:


data = pd.read_csv('dataset_full.csv')


# In[3]:


data


# In[4]:


data.info

data.describe
# In[5]:


data.dropna(inplace=True)


# In[6]:


data.info()


# In[8]:


# Split the dataset into features (X) and labels (y)
X = data.drop('phishing', axis=1)
y = data['phishing']


# In[9]:


X.shape


# In[10]:


y.shape


# In[26]:


# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[27]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


# Reshape the feature data for the CNN input
X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)


# In[29]:


# Create the CNN model
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[30]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[31]:


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[32]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


# In[ ]:




