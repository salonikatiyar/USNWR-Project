#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


def ts2022rank(f):
    print(f)
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    import matplotlib.pyplot as plt
    from statsmodels.tsa.api import VAR
    from scipy.stats import pearsonr
    import matplotlib as mlp
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.sequence import TimeseriesGenerator
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import tensorflow as tf
    input_df=pd.read_csv('/Users/pujasonawane/Desktop/input_df.csv')
    features=input_df
    target=input_df['Rank']
    features=features.values.tolist()
    target=target.values.tolist()
    leng=10
    batch_size=2
    num_features=10
    train_generator=TimeseriesGenerator(features,target,length=leng,batch_size=batch_size)
    
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(leng,num_features),return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.LSTM(16,return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(8,return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(loss=tf.losses.MeanSquaredError(),optimizer=tf.optimizers.Adam(),metrics=[tf.metrics.MeanAbsoluteError()])
    
    history= model.fit_generator(train_generator, epochs=10000, shuffle=False)
    pred=model.predict(train_generator)
    
    return int(pred[0][0])
    

    
    
    


# In[2]:


d=ts2022rank(4)


# In[3]:


print(d)


# In[ ]:




