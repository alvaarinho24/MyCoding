#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = pd.read_csv("C:\\Users\\Master\\Desktop\\E-3 An\\Udemy\\ML+Python\\datasets\\trabajo-final\\transacciones.csv", sep =";", header = None)


# In[3]:


data.head()


# In[4]:


datos = pd.DataFrame({
    'ID': data[0],
    'Fecha': data[1],
    'Valor': data[3],
    'Articulos': data[2]
})


# In[5]:


datos.head()


# In[7]:


lista_ID = pd.unique(datos['ID']).tolist()
len(lista_ID)


# In[8]:


valor=[]
for i in lista_ID:
    x = datos[datos['ID']==i]
    y = x['Valor'].mean()
    valor.append(y)
len(valor)


# In[9]:


cantidad = []
for i in lista_ID:
    x = datos[datos['ID']==i]
    cantidad.append(sum(x['Articulos']))
len(cantidad)


# In[10]:


ultima_compra=[]
for i in lista_ID:
    x = datos[datos['ID']==i]
    ultima_compra.append(max(x['Fecha']))
len(ultima_compra)


# In[11]:


df = pd.DataFrame({
    'ID':lista_ID,
    'Value':valor,
    'Quantity':cantidad,
    'Recency':ultima_compra
})


# In[12]:


df['Recency'] = pd.to_datetime(df['Recency'],format='%Y%m%d', errors='ignore')
df['Recency'] = -(df['Recency']-max(df['Recency']))
df['Recency'] = df['Recency'].astype('timedelta64[D]')


# In[13]:


df.head()


# ## Análisis de la serie temporal

# In[6]:


datos['Fecha'] = pd.to_datetime(datos['Fecha'],format='%Y%m%d', errors='ignore')


# In[13]:


time_serie=datos.groupby('Fecha').sum()
time_serie.head()


# In[17]:


from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
series = time_serie['Valor']
result = seasonal_decompose(series, model='additive')
result.plot()
pyplot.show()


# In[21]:


result.seasonal.tail(30).plot() #Estacionalidad de los últimos 30 días


# ## Outliers

# In[14]:


boxplot = df.boxplot(column=['Value', 'Quantity', 'Recency'])
plt.savefig("Boxplot.jpg", bbox_inches='tight')


# In[15]:


lower_bound = 0.01
upper_bound = 0.99
res = df[['Recency','Value','Quantity']].quantile([lower_bound,upper_bound])
res


# In[16]:


outliers = pd.concat([df[df['Value']>137],df[df['Quantity']>60]])


# In[17]:


outliers


# In[18]:


df = df[df['Value']<137]
df = df[df['Quantity']<60]


# In[19]:


df.to_csv("C:\\Users\\Master\\Desktop\\E-3 An\\customer-segmentation_orig.csv")
outliers.to_csv('C:\\Users\\Master\\Desktop\\E-3 An\\outliers.csv')


# ## Ánalisis de las componentes principales

# ### 1. Descomposicion y calculo de vectores propios
# * Usando matriz de covarianzas

# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import chart_studio.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import chart_studio


# In[98]:


chart_studio.tools.set_credentials_file(username = 'alvaarinho24', api_key='••••••••••')


# In[5]:


df = pd.read_csv("C:\\Users\\Master\\Desktop\\E-3 An\\customer-segmentation_orig.csv")


# In[6]:


df= df.iloc[:,2:5]


# In[7]:


df.values


# In[8]:


df_std = StandardScaler().fit_transform(df)
cov_matrix = np.cov(df_std.T)


# In[9]:


cov_matrix


# In[10]:


eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
print('Valores propios \n%s'%eig_vals)
print('Vectores propios \n%s'%eig_vectors)


# ## 2- Las componentes principales

# In[11]:


for ev in eig_vectors:
    print('La longitud del vector propio es: %s'%np.linalg.norm(ev))


# In[12]:


eig_pairs = [(np.abs(eig_vals[i]), eig_vectors[:,i]) for i in range(len(eig_vals))]


# In[13]:


total_sum = sum(eig_vals)
var_exp = [(i/total_sum)*100 for i in sorted(eig_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[14]:


var_exp, cum_var_exp


# In[15]:


plot1 = Bar(x=['CP %s'%i for i in range(1,5)], y = var_exp, showlegend=False)
plot2 = Scatter(x=['CP %s'%i for i in range(1,5)],y = cum_var_exp,showlegend=True,name = '% de Varianza Explicada Acumulada')

data = Data([plot1,plot2])

layout= Layout(xaxis = XAxis(title='Componentes principales'),
              yaxis = YAxis(title = 'Porcentaje de varianza explicada'),
              title = 'Porcentaje de variabilidad explicada por cada componente principal')

fig = Figure(data = data,layout = layout)
fig.show()


# In[23]:


acp = PCA(n_components = 2)
Y = acp.fit_transform(df_std)
Y


# In[21]:


X = df
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    
pca = PCA()
x_new = pca.fit_transform(X)
labels=['Value','Quantity','Recency']

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


# In[22]:


myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]),labels=labels)
plt.show()

