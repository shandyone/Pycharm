#coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#—————————————————import data——————————————————————
f=open('log-insight-2016.csv')
f_test=open('log-insight-2017.csv')
df=pd.read_csv(f)     #read data
df_test=pd.read_csv(f_test)
data=np.array(df['num'])
data_test=np.array(df_test['num'])
data=data[::-1]
data_test=data_test[::-1]

normalize_data=(data-np.mean(data))/np.std(data)  #normalization
normalize_data=normalize_data[:,np.newaxis]       #add axis
normalize_data_test=(data_test-np.mean(data_test))/np.std(data_test)  #normalization
normalize_data_test=normalize_data_test[:,np.newaxis]       #add axis