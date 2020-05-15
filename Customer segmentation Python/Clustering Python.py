#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster,dendrogram, linkage, cophenet, inconsistent


# In[3]:


data = pd.read_csv("C:\\Users\\Master\\Desktop\\E-3 An\\customer-segmentation_orig.csv")


# In[4]:


data.head()


# In[4]:


df = data.iloc[:,2:5]


# In[5]:


df_norm = (df-df.mean())/(df.max()-df.min())


# In[7]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =df_norm['Value']
y =df_norm['Recency']
z =df_norm['Quantity']



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Valor medio')
ax.set_ylabel('Última compra')
ax.set_zlabel('Número de veces')

plt.show()
plt.savefig("Scatter 3D.jpg", bbox_inches='tight')


# In[6]:


Z = linkage(df_norm,'ward', metric='euclidean')
Z


# In[9]:


plt.figure(figsize=(25,10))
dendrogram(Z, leaf_rotation=90)
plt.show()


# In[10]:


c, coph_dist = cophenet(Z,pdist(df_norm))
c


# ### Regla del codo

# In[11]:


last = Z[-10:,2]
last_rev = last[::-1]
print(last_rev)
idx= np.arange(1,len(last)+1)
plt.plot(idx,last_rev)

acc = np.diff(last,2)
acc_rev = acc[::-1]
plt.plot(idx[:-2]+1,acc_rev)
plt.show()
k = acc_rev.argmax()+2
print('El número óptimo de clusters es: %s'%str(k))


# In[7]:


k = 6
clusters = fcluster(Z, k, criterion='maxclust')
clusters


# In[8]:


plt.hist(clusters)


# ##  Aglomerativo y K-means 

# In[14]:


from sklearn.cluster import AgglomerativeClustering, KMeans


# In[15]:


df_norm = (df-df.mean())/(df.max()-df.min())


# In[9]:


import seaborn as sns; sns.set(color_codes=True)
g = sns.clustermap(df_norm)
plt.savefig("Heat map.jpg", bbox_inches='tight')


# In[16]:


clus = AgglomerativeClustering(n_clusters = 6, linkage = 'ward').fit(df_norm)


# In[17]:


md_h=pd.Series(clus.labels_)
plt.hist(md_h)
plt.savefig("Histograma aglomerativo.jpg", bbox_inches='tight')


# In[18]:


Z = linkage(df_norm,'ward')


# In[18]:


model = KMeans(n_clusters=6)
model.fit(df_norm)


# In[19]:


md_k = pd.Series(model.labels_)


# In[20]:


df_norm['clust_h'] = md_h
df_norm['clust_k'] = md_k
df_norm['clusters'] = clusters
df['clust_k']=md_k


# In[21]:


plt.hist(md_k)
plt.savefig("Histograma K-means.jpg", bbox_inches='tight')


# In[22]:


model.cluster_centers_


# In[21]:


model.inertia_


# In[23]:


fig=plt.figure()
ax=Axes3D(fig)
color = []
for i in df_norm['clust_k']:
    if i==0:
        color.append('red',)
    elif i==1:
        color.append('green')
    elif i==2:
        color.append('blue')
    elif i==3:
        color.append('pink')
    elif i==4:
        color.append('purple')
    elif i==5:
        color.append('black')
    
ax.scatter(df_norm['Recency'],df_norm['Quantity'],df_norm['Value'],color=color)
plt.show()
plt.savefig("Scatter 3D K-means.jpg", bbox_inches='tight')


# #### Interpretación final

# In[24]:


df.groupby('clust_k').mean()


# ## Árbol de decisión

# In[25]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[26]:


df_tree = df.iloc[:,0:4]
df_tree_train = df_tree.iloc[:,0:3]


# In[27]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_tree_train, df_tree['clust_k'], random_state=42)


# In[28]:


tree = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
tree.fit(Xtrain,Ytrain)


# In[29]:


preds = tree.predict(Xtest)


# In[30]:


pd.crosstab(Ytest,preds,rownames=['Actual'],colnames=['Predicción'])


# In[31]:


accuracy_score(preds, Ytest)


# ## Recuperación clientes olvidados

# In[32]:


outliers = pd.read_csv('C:\\Users\\Master\\Desktop\\E-3 An\\outliers.csv')


# In[33]:


outliers=outliers.iloc[:,2:5]


# In[34]:


pred_outliers = tree.predict(outliers)


# In[35]:


pred_outliers


# In[36]:


outliers['clust_k']=pred_outliers


# In[37]:


outliers.hist()
plt.savefig("Outliers Histograma.jpg", bbox_inches='tight')


# In[38]:


df = pd.concat([outliers,df])


# In[39]:


df['clust_k'] # Sería el dataset final con todos los clientes clasificados incluidos los outliers.
              # Los outliers son los quitamos porque sesgaban demasiado el clustering

