#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("user_data.csv")
dataset


# In[35]:


x= dataset.iloc[:, [2,3]].values  
y= dataset.iloc[:, 4].values  
print(x)
print(y)


# In[36]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)


# In[37]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)    


# In[38]:


from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(x_train, y_train)  


# In[39]:


y_pred= classifier.predict(x_test)  


# In[40]:


y_pred


# In[41]:


from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  


# In[42]:


cm


# In[43]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))


# In[ ]:




