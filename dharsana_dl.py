#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns


# pip install tensorflow
# 

# In[2]:


pip install seaborn


# In[8]:


pip install tensorflow


# In[ ]:





# In[3]:


file_path = 'C:/Users/Admin/Downloads/playground-series-s3e2/train.csv'
df_train = pd.read_csv(file_path)
df_train.head()


# In[4]:


df_train.isnull().sum()


# In[7]:


pip install scikit-learn


# In[8]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_train.iloc[:,1:2]= le.fit_transform(df_train.iloc[:,1:2])
df_train.iloc[:,5:6]= le.fit_transform(df_train.iloc[:,5:6])
df_train.iloc[:,6:7]= le.fit_transform(df_train.iloc[:,6:7])
df_train.iloc[:,7:8]= le.fit_transform(df_train.iloc[:,7:8])
df_train.iloc[:,10:11]= le.fit_transform(df_train.iloc[:,10:11])


# In[9]:


df_train=df_train.drop(['id'], axis=1)
df_train


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Check the distribution of values in the "stroke" column
print(df_train['stroke'].value_counts())

# Create a countplot
sns.countplot(data=df_train, x='stroke')

# Show the plot
plt.show()


# In[17]:


from sklearn.utils import resample

# Create two different dataframes for the majority and minority class
df_majority = df_train[df_train['stroke'] == 0]
df_minority = df_train[df_train['stroke'] == 1]

# Upsample the minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,  # sample with replacement
    n_samples=500,  # to match majority class
    random_state=42  # reproducible results
)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

import seaborn as sns
import matplotlib.pyplot as plt

# Create a countplot for the "stroke" column in df_upsampled
sns.countplot(data=df_upsampled, x='stroke')

# Show the plot
plt.show()


# In[11]:


x = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]


# In[12]:


x


# In[13]:


y


# In[14]:


df_train['stroke'].unique()


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[19]:


model=keras.Sequential()
model.add(keras.layers.Dense(10,input_dim=10,activation='relu'))
model.add(keras.layers.Dense(200,input_dim=10,activation='relu'))
model.add(keras.layers.Dense(300,input_dim=200,activation='relu'))
model.add(keras.layers.Dense(400,input_dim=300,activation='relu'))
model.add(keras.layers.Dense(300,input_dim=400,activation='relu'))
model.add(keras.layers.Dense(1,input_dim=300,activation='sigmoid'))


# In[20]:


model.summary()


# In[21]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
model_history=model.fit(x_train,y_train,epochs=10)


# In[22]:


y_pred=model.predict(x_test)


# In[23]:


y_pred_label = [np.argmax(i) for i in y_pred]
y_pred_label


# In[24]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_label))


# In[25]:


model.evaluate(x_test,y_test)


# In[26]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_label)
print(cm)


# In[27]:


pd.DataFrame(model_history.history).plot()
plt.grid(True)


# In[28]:


model.predict([[0 ,52, 0, 0, 1, 2, 0, 85.16, 32.5 ,0]])


# In[29]:


model.save('D:/DL/model1.h5')


# In[30]:


pip install streamlit


# In[32]:


pip install streamlit


# In[33]:


import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras


# In[35]:


model = keras.models.load_model('D:/DL/model1.h5')


# In[36]:


st.title('Stroke Prediction App')


# In[37]:


gender=st.slider('gender',0,1)
age = st.slider('Age', 0, 100, 50)
hypertension=st.slider('hypertension',0,1)
heart_disease=st.slider('heart disease',0,1)
ever_married=st.slider('ever married',0,1)
work_type=st.slider('work_type',0,3)
residence_type=st.slider('residence type',0,1)
avg_glucose_level=st.slider('avgerage glucose level',0.00,400.00,200.00)
bmi=st.slider('bmi',0.00,100.00,50.00)
smoking_status=st.slider('smoking status:',0,2)


# In[42]:


import numpy as np

# Define input_data with the features you want to use
input_data = np.array([[0, age, 0, 0, 1, 2, 0, 82.54, 33.4, 0]])


# In[43]:


prediction = model.predict(input_data)

# Display the prediction result
if prediction[0][0] > 0.5:
    print("Stroke is predicted.")
else:
    print("No stroke is predicted.")

