#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import numpy as np
import matplotlib.pyplot as plt
from simple_vehicle import SimpleVehicle
from simple_robot import SimpleRobot, SimpleRobotEnv
from rl_agent import ReinforceAgent

import time
import json

EPISODES = 3000

env  = SimpleRobotEnv()
#env  = SimpleVehicle()
state_size = env.state_size
action_size = env.rl_actions.number_of_available_action
agent = ReinforceAgent(state_size, action_size)
scores, episodes = [], []
global_step = 0
start_time = time.time()

for e in range(agent.load_episode + 1, EPISODES):
    done = False
    state = env.reset()
    #discarding first observation
    action = agent.getAction(state)
    env.discrete_step(action)
    env.render()
    score = 0
    for t in range(agent.episode_step):
        action = agent.getAction(state)

        next_state, reward, done = env.discrete_step(action)
        env.render()
        agent.appendMemory(state, action, reward, next_state, done)
        
        if len(agent.memory) >= agent.train_start:
            if global_step <= agent.target_update:
                agent.trainModel()
            else:
                agent.trainModel(True)

        score += reward
        state = next_state
        
        if e % 10 == 0:
            agent.model.save(agent.dirPath + str(e) + '.h5')
            with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                json.dump(param_dictionary, outfile)

        if t >= 500:
            print("Time out!!")
            done = True

        if done:
            agent.updateTargetModel()
            scores.append(score)
            episodes.append(e)
            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)

            print('Ep: {:d} score: {:.2f} memory: {:d} epsilon: {:.2f} time: {:d}:{:02d}:{:02d}'.format( e, score, len(agent.memory), agent.epsilon, h, m, s))
            param_keys = ['epsilon']
            param_values = [agent.epsilon]
            param_dictionary = dict(zip(param_keys, param_values))
            break

        global_step += 1
        if global_step % agent.target_update == 0:
            print("UPDATE TARGET NETWORK")

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
