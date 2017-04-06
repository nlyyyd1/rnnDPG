#! /usr/bin/env python
#coding=utf-8
import numpy as np
from actor import ActorNet
#actor network only add state in input and output
from critic import CriticNet
#critic network only change in input

from collections import deque
from gym.spaces import Box,Discrete
import random
from tensorflow_grad_inverter import grad_inverter
import time
import pickle

REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
is_grad_inverter = True
unroll = 500

class DDPG:
    '''
    deep deterministic policy gradient algorithm
    '''
    #初始化一个ddpg，基本就是定义好state，action，因为action是输出，还要定义好他的大小范围，初始化两个网络类    
    def __init__(self,env,is_batch_norm):
        self.env = env
        self.num_states = env.observation_space.shape[0]-1
        self.num_actions = env.action_space.shape[0]
        self.num_hidden_states = 5
        
        if is_batch_norm:
            self.critic_net = CriticNet(self.num_states,sself.num_actions)
            self.actor_net = ActorNet(self.num_states,self.num_actions)
        else:
            self.critic_net = CriticNet(self.num_states,self.num_actions)
            self.actor_net = ActorNet(self.num_states,self.num_actions)
        
        #因为是连续的，刚开始要确定action是多大，范围是多少。
        action_max = np.array(env.action_space.high).tolist()
        action_min = np.array(env.action_space.low).tolist()
        action_bounds = [action_max,action_min]
        
        #初始化一个东西，计算Q的梯度用的
        self.grad_inv = grad_inverter(action_bounds)
        self.replay_memory = deque()
     
    #用来输出action用的
    def evaluate_actor(self,state_t,c,h):
        return self.actor_net.action(state_t,c,h)
    
    def init_experience(self,file_name):
        self.replay_memory = pick.load(open(file_name,'r'))
    
    def add_experience(self,exp):
        self.replay_memory.append(exp)
        if(len(self.replay_memory)>REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()
            
    def minibatches(self,unroll):
        batches = random.sample(self.replay_memory,BATCH_SIZE)
        self.states = []
        self.actions = []
        self.next_states = []
        self.reward = []
        self.done = []
        for episode in batches:
            if len(episode)>unroll:
                episode = episode[:unroll]
            else:
                episode = episode + [([0],[-1],[0],0,True)]*(unroll-len(episode))
            self.states.append([x[0] for x in episode])
            self.actions.append([x[1] for x in episode])
            self.next_states.append([x[2] for x in episode])
            self.reward.append([x[3] for x in episode])
            self.done.append([x[4] for x in episode])
        self.states = np.array(self.states).reshape(-1,self.num_states)
        self.actions = np.array(self.actions).reshape(-1,self.num_actions)
        self.next_states = np.array(self.next_states).reshape(-1,self.num_states)       
        self.reward = np.array(self.reward).reshape(-1,1)
        self.done = np.array(self.done).reshape(-1,1)
        
        
             
    def train(self):
        
        # lets make unroll 500 first
        self.minibatches(unroll)
        LSTM_SIZE = 40
        
        self.action_t_1 = self.actor_net.evaluate_target_actor(self.next_states)
        #下一个时刻的action倒是用target的actor网络计算出来的
        
        q_t_1 = self.critic_net.evaluate_target_critic(self.next_states,self.action_t_1)
        #下一个时刻的value值也是通过target的critic网络计算出来的
        
        self.y_i = []#我们根据target网络求得时间差分的前一项rt+1+gamma*qt+1
        for i in range(0,BATCH_SIZE*unroll):
            if self.done[i]:
                self.y_i.append(self.reward[i])
            else:
                self.y_i.append(self.reward[i]+GAMMA*q_t_1[i][0])
        self.y_i = np.array(self.y_i)
        self.y_i = np.reshape(self.y_i,[len(self.y_i),1])#我感觉就是为了防止他不是列向量
        
        #用target网络求出来时间差分钱项y之后，就可以更新q函数的网络了。
        self.critic_net.train_critic(self.states,self.actions,self.y_i)
        
        action_for_delQ = self.actor_net.evaluate_actor(self.states)
        
        if is_grad_inverter:
            #求梯度
            self.detq = self.critic_net.compute_delQ_a(self.states,action_for_delQ)
            self.detq = self.grad_inv.invert(self.detq,action_for_delQ)
        else:
            self.detq = self.critic_net.compute_delQ_a(self.states,action_for_delQ)[0]
        
        #将梯度带入actor网络进行训练
        self.actor_net.train_actor(self.states,self.detq)
        #不用target网络的，用真实网络的试一试，target只用来计算label
        #训练好后要更新target网络了
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        
    def save(self,actor_path,critic_path):
        self.critic_net.save_critic(critic_path)
        self.actor_net.save_actor(actor_path)
        
    def load(self,actor_path,critic_path):
        self.critic_net.load_critic(critic_path)
        self.actor_net.load_actor(actor_path)
