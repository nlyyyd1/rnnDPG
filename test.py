#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf
import numpy as np

#m = np.array([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]])
m = np.array([1,2,3,4,5,6,7])
print m.shape
sess = tf.Session()

a = tf.placeholder(tf.int32,shape=())
w = tf.Variable(tf.ones(1))
q = [tf.pack([-1,a])]
print q
print type(q)
q.append(2)
print q
n = tf.reshape(m,shape=q)

nn = tf.placeholder(tf.float32,[None,None,5])

y = tf.mul(nn,w)
print y
print m
print n
print nn

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(4,state_is_tuple=True)
state = lstm_cell.zero_state(a,dtype=tf.float32)
print tf.nn.dynamic_rnn(cell=lstm_cell,inputs=nn,initial_state = state)