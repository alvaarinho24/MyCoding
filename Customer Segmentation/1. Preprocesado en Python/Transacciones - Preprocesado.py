#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[26]:


data = pd.read_csv("C:\\Users\\Master\\Desktop\\E-3 An\\Udemy\\ML+Python\\datasets\\trabajo-final\\transacciones.csv", sep =";", header = None)


# In[27]:


data.head()


# In[30]:


datos = pd.DataFrame({
    'ID': data[0],
    'Fecha': data[1],
    'Valor': data[3]
})


# In[34]:


datos.head(), datos.shape


# In[61]:


lista_ID = pd.unique(datos['ID']).tolist()
len(Lista_ID)


# In[67]:


valor=[]
for i in lista_ID:
    x = datos[datos['ID']==i]
    y = x['Valor'].mean()
    valor.append(y)
len(valor)


# In[71]:


cantidad = []
for i in lista_ID:
    x = datos[datos['ID']==i]
    cantidad.append(len(x))
len(cantidad)


# In[76]:


ultima_compra=[]
for i in lista_ID:
    x = datos[datos['ID']==2]
    ultima_compra.append(max(x['Fecha']))
len(ultima_compra)


# In[78]:


df = pd.DataFrame({
    'ID':lista_ID,
    'Value':valor,
    'Quantity':cantidad,
    'Recency':ultima_compra
})


# In[79]:


df.head()


# In[80]:


df.to_csv("C:\\Users\\Master\\Desktop\\E-3 An\\Udemy\\ML+Python\\datasets\\trabajo-final\\customer-segmentation.csv")

