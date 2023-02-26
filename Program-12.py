#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]
plt.scatter(x,y)
plt.show()


# In[2]:


from scipy.cluster.hierarchy import dendrogram,linkage
data=list(zip(x,y))
lin_data=linkage(data,method='ward',metric='euclidean')
dendrogram(lin_data)
plt.show()


# In[3]:


from sklearn.cluster import AgglomerativeClustering
h_clu=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
hlabels=h_clu.fit_predict(data)
print(hlabels)
plt.scatter(x,y,c=hlabels)
plt.show()


# In[4]:


h_clu.fit(data)


# In[5]:


labels=h_clu.labels_
labels


# In[7]:


from sklearn.metrics import silhouette_score
sco=silhouette_score(data,labels)
sco


# In[ ]:




