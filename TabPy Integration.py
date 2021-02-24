#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import _ssl


# In[2]:


data= pd.read_csv("/Users/pujasonawane/Documents/DA fall 2020/capstone project/NewData26Nov/cap27oct.csv")


# In[3]:


data.describe()


# In[4]:


data.isna().sum()


# In[5]:


data.dtypes


# In[6]:


wscore=data


# In[7]:


wscore=wscore.drop(['University name','City','State','Year'],axis=1)


# In[ ]:





# In[8]:


train_data,test_data = train_test_split(wscore,test_size=0.33)


# In[9]:


train_data.columns


# In[10]:


train_x= train_data.iloc[:,train_data.columns!='Rank']
train_y= train_data['Rank']
test_x=test_data.iloc[:,test_data.columns!='Rank']
test_y= test_data['Rank']


# In[11]:


train_x.dtypes


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


#train_x = StandardScaler().fit_transform(train_x)
#test_x = StandardScaler().fit_transform(test_x)


# train_x=preprocessing.scale(train_x)
# test_x=preprocessing.scale(test_x)

# In[14]:


import mord as m


# In[15]:


model=m.OrdinalRidge(alpha=0.1, max_iter=10000)


# In[16]:


train_x[:1]


# In[17]:


model.fit(train_x, train_y)


# In[18]:


model.score(test_x,test_y)


# In[19]:


pred=model.predict(test_x)


# In[20]:


mae=mean_absolute_error(test_y,pred)


# In[21]:


mae


# In[22]:


d=pd.DataFrame([1.3,1.4,140,0.095,23100000,355800,23,216,0.137])


# In[23]:


#d = StandardScaler().fit_transform(d)


# In[24]:


d=np.array(d).reshape(1,-1)


# In[25]:


print(d)


# In[26]:


model.predict(d)


# In[27]:


from sklearn.metrics import r2_score


# In[28]:


import joblib 


# In[29]:


joblib.dump(model, '/Users/pujasonawane/Documents/DA fall 2020/capstone project/model_depl1204.pkl') 


# In[30]:


num_test=np.array(test_y)


# In[31]:


r2=r2_score(num_test, pred)


# In[32]:


r2


# In[40]:


q=10
b=20
y=30


# In[41]:


a=np.column_stack((q,b,y))
print(a[0])


# In[42]:


print(a)


# ##  Connecting to Tableau

# In[60]:


def Rank(_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,_arg8,_arg9):
    #from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import joblib

    #df=np.column_stack((_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,_arg8,_arg9,_arg10))
    df = pd.DataFrame([_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,_arg8,_arg9])
    #df=preprocessing.scale(df)
    print(df)
    #df = StandardScaler().fit_transform(df)
    df=np.array(df).reshape(1,-1)
    
    model= joblib.load('/Users/pujasonawane/Documents/DA fall 2020/capstone project/model_depl.pkl')
    print(model)
    print(df)
    
    predicted= model.predict(df)
    #print(predicted)
    predi=int(predicted)
    return predi


# In[61]:


import tabpy.tabpy_tools.client


# In[62]:


connection = tabpy.tabpy_tools.client.Client('http://localhost:9004/')


# In[63]:


connection.deploy('Rank', Rank, 'Predicting the ranking', override=True)


# In[64]:


Rank(3.3,3.4,164,0.095,23100000,355800,23,216,0.137)


# In[58]:


def fun(_arg1):
    a=_arg1
    return a


# In[59]:


connection.deploy('fun', fun, 'testModel', override=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




