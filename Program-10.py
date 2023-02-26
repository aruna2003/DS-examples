#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("user_data.csv")
dataset
x= dataset.iloc[:, [2,3]].values  
y= dataset.iloc[:, 4].values  
print(x)
print(y)


# In[ ]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)


# In[12]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)    


# In[4]:


from sklearn.neighbors  import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=10)  
classifier.fit(x_train, y_train)  


# In[14]:


y_pred= classifier.predict(x_test)  
y_pred


# In[6]:


print(classifier.score(x_test, y_test))


# In[7]:


y_pred


# In[15]:


from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)


# In[16]:


cm


# In[17]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))


# In[ ]:




