#! /usr/bin/env python
#coding=utf-8
#! /usr/bin/env python
#coding=utf-8

import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.001
BATCH_SIZE = 64
TAU = 0.5

class CriticNet:
    '''
    actor network model of DDPG algorthm
    '''
    
    def __init__(self,num_states,num_actions):
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            self.unroll = tf.placeholder(tf.int32,shape=(),name='criticUnroll')
            #in critic network I think the unroll is unchanged
            #critic network
            self.parameters,self.critic_q_model,self.critic_state_in,self.critic_action_in=self.create_critic_net(num_states,num_actions,'critic')
            
            #target network
            self.t_parameters,self.t_critic_q_model,self.t_critic_state_in,self.t_critic_action_in=self.create_critic_net(num_states,num_actions,'target')
            
            self.t_parameters = self.t_parameters[9:]
            
            #cost
            self.q_value_in = tf.placeholder('float',[None,1])
            #self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.w2,2))+0.0001*tf.reduce_sum(tf.pow(self.b2,2))
            self.cost = tf.reduce_mean(tf.pow(self.critic_q_model-self.q_value_in,2)) #+self.l2_regularizer_loss
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
            
            self.act_grad_v = tf.gradients(self.critic_q_model,self.critic_action_in)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])]#这是归一了？
            #DELQ is flatted
            #跟batch有关？
            self.check_fl = self.action_gradients
            
            sync_vars_list = []
            for (item,t_item) in zip(self.parameters,self.t_parameters):
                sync_vars_list.append(tf.assign(t_item,TAU*item+(1-TAU)*t_item))
            self.sync_vars = tf.group(*sync_vars_list)
            
            self.saver = tf.train.Saver()
            self.sess.run(tf.initialize_all_variables())
            
            for (item,t_item) in zip(self.parameters,self.t_parameters):
                self.sess.run(t_item.assign(item))            
            
    def create_critic_net(self,num_states,num_actions,models):
        '''
        network that takes states and return action
        '''
        
        #todo:network struck
        N_HIDDEN_1 = 40
        N_HIDDEN_2 = 30
        LSTM_SIZE = 2
        critic_state_in = tf.placeholder('float',[None,num_states])
        critic_action_in = tf.placeholder('float',[None,num_actions])
        with tf.variable_scope(models):
            w1=tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            w2=tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
            w2_action=tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
            w3=tf.Variable(tf.random_uniform([LSTM_SIZE,1],-0.003,0.003))
            b1=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            b2=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_states),1/math.sqrt(N_HIDDEN_1+num_states)))
            b3=tf.Variable(tf.random_uniform([1],-0.003,0.003))
                        
            h1 = tf.nn.relu(tf.matmul(critic_state_in,w1)+b1)
            h = tf.nn.relu(tf.matmul(h1,w2)+tf.matmul(critic_action_in,w2_action)+b2)
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE,state_is_tuple=True)
            initial_lstm_state = lstm_cell.zero_state(BATCH_SIZE,dtype=tf.float32)
            lstm_inputs = tf.reshape(h,shape=(BATCH_SIZE,-1,N_HIDDEN_2))
            h2,_= tf.nn.dynamic_rnn(cell=lstm_cell,inputs=lstm_inputs,initial_state=initial_lstm_state)
            h2=tf.reshape(h2,shape=(-1,LSTM_SIZE))
                    
            critic_q_model = tf.matmul(h2,w3)+b3
            critic_q_model = tf.reshape(critic_q_model,shape=(-1,1))#this is all the Q in history flatted
            #7 parameters + 2 lstm para
            parameters = tf.trainable_variables()
            return parameters,critic_q_model,critic_state_in,critic_action_in    
    
    def evaluate_critic(self,state_t,action_t):
        #这个函数是用来给state和action来得到q函数的
        return self.sess.run(self.critic_q_model,feed_dict={self.critic_state_in:state_t,self.critic_aciton_in:action_t})
    
    def evaluate_target_critic(self,state_t_1,action_t_1):
        #这个是target网络的
        return self.sess.run(self.t_critic_q_model,feed_dict={self.t_critic_state_in:state_t_1,self.t_critic_action_in:action_t_1})
    
    def train_critic(self,state_t_batch,action_batch,y_i_batch):
        self.sess.run(self.optimizer,feed_dict={self.critic_state_in:state_t_batch,self.critic_action_in:action_batch,self.q_value_in:y_i_batch})
        
    def update_target_critic(self):
        self.sess.run(self.sync_vars)
    
    def save_actor(self,path_name):
        save_path = self.saver.save(self.sess,path_name)
        print "Critic model saved in file %s" % save_path
        
    def load_actor(self,path_name):
        self.saver.restore(self.sess,path_name)
        print "Critic model restored"
        
    def compute_delQ_a(self,state_t,action_t):
        return self.sess.run(self.action_gradients,feed_dict={self.critic_state_in:state_t,self.critic_action_in:action_t})
        
