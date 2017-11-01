import tensorflow as tf
import read_csv
import numpy as np
import matplotlib.pyplot as plt

lr=0.0006
input_size=30
predict_str=4
output_size=1

data=read_csv.data
data_test=read_csv.data_test
normalize_data=read_csv.normalize_data
normalize_data_test=read_csv.normalize_data_test

train_x,train_y=[],[]   #training dataset
for i in range(len(normalize_data)-predict_str-1):
    x=normalize_data[i:i+predict_str]
    y=normalize_data[i+predict_str+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

x=tf.placeholder(tf.float32,[None,input_size])
y_actual=tf.placeholder(tf.float32,shape=[None,output_size])

def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv1d(value,filter):
    return tf.nn.conv1d(value,filter,stride=1,padding='SAME')

x_input=tf.reshape(x,[-1,input_size,1])
w_conv1=weight_variable([predict_str,1,24])
b_conv1=bias_variable([24])
h_conv1=tf.nn.relu(conv1d(x_input,w_conv1)+b_conv1)
#def max_pool(x):
#    return tf.nn.max_pool()

cross_entropy=-tf.reduce_sum(y_actual*tf.log(y_pre))
train_op=tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        start=0
        end=0
