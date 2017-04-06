#! /usr/bin/env python
#coding=utf-8

import numpy as np
import tensorflow as tf
import math

from tensorflow.models.rnn.ptb import reader

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.5

class ActorNet:
    '''
    actor network model of DDPG algorthm
    '''
    
    def __init__(self,num_states,num_actions):
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            #self.unroll = tf.placeholder(tf.int32,shape=(),name='actorUnroll')
            #self.batch_size = tf.placeholder(tf.int32,shape=())
            #actor network
            self.actor_parameters,self.actor_model,self.input,self.statec_t,self.stateh_t,self.statec,self.stateh = self.create_actor(num_states,num_actions,'actor')
            self.target_actor_parameters,self.target_actor_model,self.target_input,self.target_statec_t,self.target_stateh_t,self.target_statec,self.target_stateh = self.create_actor(num_states,num_actions,'target')
            
            self.target_actor_parameters = self.target_actor_parameters[8:]
            #cost
            self.q_gradient_input = tf.placeholder('float',[None,num_actions])
            gradients = tf.gradients(self.actor_model,self.actor_parameters,-self.q_gradient_input)
            #tf.gradient(x,y,z)表示的是求z×dx/dy
            self.parameters_gradients,_ =tf.clip_by_global_norm(gradients,5) 
            #要控制一下梯度膨胀
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))
            
            sync_vars_list = []
            for (item,t_item) in zip(self.actor_parameters,self.target_actor_parameters):
                sync_vars_list.append(tf.assign(t_item,TAU*item+(1-TAU)*t_item))
            self.sync_vars = tf.group(*sync_vars_list)
                                    
            
            self.saver = tf.train.Saver()
            self.sess.run(tf.initialize_all_variables())
            
            for (item,t_item) in zip(self.actor_parameters,self.target_actor_parameters):
                self.sess.run(t_item.assign(item))            
                                    
                                    
            
    def create_actor(self,num_states,num_actions,models):
        
        N_HIDDEN_1 = 40
        N_HIDDEN_2 = 30
        LSTM_SIZE = 40
        input = tf.placeholder(tf.float32,[None,num_states])
        statec = tf.placeholder(tf.float32,[None,LSTM_SIZE])
        stateh = tf.placeholder(tf.float32,[None,LSTM_SIZE])
        init_state = tuple([statec,stateh])
                        
        with tf.variable_scope(models):
            
            w1=tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            w2=tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
            w3=tf.Variable(tf.random_uniform([LSTM_SIZE,1],-0.003,0.003))
            b1=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            b2=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_states),1/math.sqrt(N_HIDDEN_1+num_states)))
            b3=tf.Variable(tf.random_uniform([1],-0.003,0.003))
                                
            h1 = tf.nn.relu(tf.matmul(input,w1)+b1)
            h = tf.nn.relu(tf.matmul(h1,w2)+b2)
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE,state_is_tuple=True)
            init_state = lstm_cell.zero_state(BATCH_SIZE,dtype=tf.float32)
            
            lstm_inputs = tf.reshape(h,shape=(BATCH_SIZE,-1,N_HIDDEN_2))
            h2,state= tf.nn.dynamic_rnn(cell=lstm_cell,inputs=lstm_inputs,initial_state=init_state)
            h2=tf.reshape(h2,shape=(-1,LSTM_SIZE))        
        
            actor_model = tf.matmul(h2,w3)+b3
            #6 parameters + 2 lstm para
            actor_parameters = tf.trainable_variables()
            return actor_parameters,actor_model,input,state[0],state[1],statec,stateh
        
         
    def action(self,state_t,c,h):
        #这个函数是用来给state来得到actor的，actor用actor_model来表示
        return self.sess.run([self.actor_model,self.statec_t,self.stateh_t],feed_dict={self.input:state_t,self.statec:c,self.stateh:h})

    def evaluate_actor(self,state_t):
        return self.sess.run(self.actor_model,feed_dict={self.input:state_t})
            
    def evaluate_target_actor(self,state_t):
        return self.sess.run(self.target_actor_model,feed_dict={self.target_input:state_t})
    
    def train_actor(self,actor_state_in,q_gradient_input):
        self.sess.run(self.optimizer,feed_dict={self.input:actor_state_in,self.q_gradient_input:q_gradient_input})

    def update_target_actor(self):
        self.sess.run(self.sync_vars)
        
    def save_actor(self,path_name):
        save_path = self.saver.save(self.sess,path_name)
        print "Actor model saved in file %s" % save_path
        
    def load_actor(self,path_name):
        self.saver.restore(self.sess,path_name)
        print "Actor model restored"
        
