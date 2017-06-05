
# coding: utf-8

# In[2]:

#ler dados do iris
from sklearn.datasets import load_iris

iris = load_iris()

#criar  X (recursos) e y (resposta)

x = iris.data
y = iris.target 


# In[3]:

#import classe 
from sklearn.linear_model import LogisticRegression

#instanciar o modelo(usando parâmetro)
logreg = LogisticRegression()

#ajusta o modelo com dados
logreg.fit(x,y)

#prever as resposta s dos valores para observação em x 
logreg.predict(x)


# In[4]:

#armazenar os valores reposta previstos 

y_pred = logreg.predict(x)

# verificar quantas previsão foram geradas
len(y_pred)


# In[7]:

# calcula  a precisão  da classificacao para o modelo de regressão  logistica 
from sklearn import metrics
print(metrics.accuracy_score(y,y_pred))


# In[9]:

#conhecida  como precisão  de treinamento quando  voce  treinar  e testar 
#o modelo  nos  mesmos dados knn (k = 5)

from sklearn.neighbors import KNeighborsClassifier
knn  = KNeighborsClassifier(n_neighbors=5)

knn.fit(x,y)

y_pred = knn.predict(x)

print(metrics.accuracy_score(y,y_pred))

 
# In[10]:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)
y_pred = knn.predict(x)
print(metrics.accuracy_score(y,y_pred))


# In[ ]:



