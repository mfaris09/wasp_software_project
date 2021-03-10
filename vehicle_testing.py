import os
import numpy as np
import unittest
from simple_vehicle import SimpleVehicle
import copy
from rl_agent import ReinforceAgent

class testing_vehicle(unittest.TestCase):
	
	# 1. Initial position x
	def test_init_pos_x(self):
		print('initial x-axis position test')		
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertGreater(self.observation[0],-10, 'robot is not started from above x-position -10') 
		self.assertLess(self.observation[0],10, 'robot is not started below x-position 10')

	# 2.Initial position y
	def  test_init_pos_y(self):		
		print('initial y-axis position test')			
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertGreater(self.observation[1],10, 'robot is not started from above y-position 10') 
		self.assertLess(self.observation[1],20, 'robot is not started below y-position 20')		

	# 3. Initial heading (radian)	 
	def test_init_heading(self):
		print('initial heading orientation test')	
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertLess(self.observation[2],2*np.pi, 'robot heading is not started from below 2*pi')		

	# 4. environment condition test
	def test_env(self):
		print('test vehicle environment')		
		env  = SimpleVehicle()
		
		self.assertIsNone(env.fig)
		self.assertIsNone(env.ax)			
		
		self.assertGreater(env.xlim[1], env.xlim[0], 'lower x-limit is greater than higher x-limit')
		self.assertGreater(env.ylim[1], env.ylim[0], 'lower y-limit is greater than higher y-limit')
		
		self.assertGreater(env.dt, 0., 'sampling time is zero')
	
	# 5. move straight
	def test_next_state_straight(self):		
		print('test straight move')
		env = SimpleVehicle()		
		next_state = env.reset()
		print(next_state)
		init = copy.copy(next_state)
		action = 2# move straight		
		next_state, reward, done = env.discrete_step(action)
		print(next_state)
		print(init)
				
		self.assertNotEqual(init[0],next_state[0], 'initial and next state x-position are the same')	
		self.assertNotEqual(init[1],next_state[1], 'initial and next state y-position are the same')	
		self.assertEqual(init[2],next_state[2], 'initial and next state heading are NOT the same')							

	# 6. move straight fast
	def test_next_state_straight_fast(self):		
		print('test straight move fast')
		env = SimpleVehicle()		
		next_state = env.reset()
		print(next_state)
		init = copy.copy(next_state)
		action = 3# move straight fast		
		next_state, reward, done = env.discrete_step(action)
		print(next_state)
		print(init)
				
		self.assertNotEqual(init[0],next_state[0], 'initial and next state x-position are the same')	
		self.assertNotEqual(init[1],next_state[1], 'initial and next state y-position are the same')					
		self.assertEqual(init[2],next_state[2], 'initial and next state heading are NOT the same')									

	# 7. turn left
	def test_next_state_left(self):		
		print('test left move')
		env = SimpleVehicle()		
		next_state = env.reset()
		print(next_state)
		init = copy.copy(next_state)
		action = 0# move left		
		next_state, reward, done = env.discrete_step(action)
		print(next_state)
		print(init)
				
		self.assertNotEqual(init[0],next_state[0], 'initial and next state x-position are the same')	
		self.assertNotEqual(init[1],next_state[1], 'initial and next state y-position are the same')					
		self.assertLess(init[2],next_state[2], 'next state heading is less than initial')						
	
	# 8. turn left fast		
	def test_next_state_left_fast(self):
		print('test left move fast')				
		env = SimpleVehicle()		
		next_state = env.reset()
		print(next_state)
		init = copy.copy(next_state)
		action = 1# move left		
		next_state, reward, done = env.discrete_step(action)
		print(next_state)
		print(init)
				
		self.assertNotEqual(init[0],next_state[0], 'initial and next state x-position are the same')	
		self.assertNotEqual(init[1],next_state[1], 'initial and next state y-position are the same')					
		self.assertLess(init[2],next_state[2], 'next state heading is less than initial')	

	# 9.
	def test_next_state_right(self):
		print('test right move')				
		env = SimpleVehicle()		
		next_state = env.reset()
		print(next_state)
		init = copy.copy(next_state)
		action = 4# move right		
		next_state, reward, done = env.discrete_step(action)
		print(next_state)
		print(init)
				
		self.assertNotEqual(init[0],next_state[0], 'initial and next state x-position are the same')	
		self.assertNotEqual(init[1],next_state[1], 'initial and next state y-position are the same')					
		self.assertGreater(init[2],next_state[2], 'next state heading is greater than initial')							

	# 10.
	def test_next_state_right_fast(self):
		print('test right move fast')				
		env = SimpleVehicle()		
		next_state = env.reset()
		print(next_state)
		init = copy.copy(next_state)
		action = 5# move right fast		
		next_state, reward, done = env.discrete_step(action)
		print(next_state)
		print(init)
				
		self.assertNotEqual(init[0],next_state[0], 'initial and next state x-position are the same')	
		self.assertNotEqual(init[1],next_state[1], 'initial and next state y-position are the same')					
		self.assertGreater(init[2],next_state[2], 'next state heading is greater than initial')	

	# 11. test RL action
	def test_rl_action(self):
		print('test RL action')
		env  = SimpleVehicle()
		state = env.reset()		
		state_size = env.state_size
		action_size = env.discrete_action_size
		agent = ReinforceAgent(state_size, action_size)
		action = agent.getAction(state)
		print('action: ', action)
		
		self.assertGreater(action,-1, 'action number is lower than 0')
		self.assertLess(action,6, 'action number is greater than 5')												

	# 12. Check minimum calculated reward
	def test_min_reward(self):
		print('test minimum reward')
		env  = SimpleVehicle()
		state = env.reset()		
		state_size = env.state_size
		action_size = env.discrete_action_size
		agent = ReinforceAgent(state_size, action_size)
		action = agent.getAction(state)
		next_state, reward, done = env.discrete_step(action)		
		print('action: ', action)
		print('reward: ', reward)
		
		self.assertGreater(reward,0.001, 'reward is greater than 0.001')

	# test done condition
	def test_done_condition(self):
		print('check done condition')
		env = SimpleVehicle()
		state = env.reset()		
		state_size = env.state_size
		action_size = env.discrete_action_size
		agent = ReinforceAgent(state_size, action_size)
		
		action = agent.getAction(state)
		next_state, reward, done = env.discrete_step(action)		
		if (env._border_fcn(env.tmpstep[:, 0]) > env.tmpstep[:, 1]).any() \
                or (env.tmpstep[:, 0] < env.xlim[0]).any() \
                or (env.tmpstep[:, 0] > env.xlim[1]).any() \
                or (env.tmpstep[:, 1] < env.ylim[0]).any() \
				or (env.tmpstep[:, 1] > env.ylim[1]).any() \
				or ((env.tmpstep[:, 0] > min(env.ox1)).any() and (env.tmpstep[:, 0] < max(env.ox1)).any() and (env.tmpstep[:, 1] > min(env.oy1)).any() and (env.tmpstep[:, 1] < max(env.oy1)).any()) \
				or ((env.tmpstep[:, 0] > min(env.ox2)).any() and (env.tmpstep[:, 0] < max(env.ox2)).any() and (env.tmpstep[:, 1] > min(env.oy2)).any() and (env.tmpstep[:, 1] < max(env.oy2)).any()):		  				 					
			self.assertTrue(done, 'done is False')
		else:
			self.assertFalse(done, 'done is True')

	def check_discrete_action():
		print('check discrete action function')
		env = SimpleVehicle()
		action = discrete_action(3) # choose any number
		list_action_length = env.discrete_action_size
		# speed
		assertGreater(action[0], 0, 'speed is less than 1')
		assertLess(action[0], 4, 'speed is greater than 3')
		# heading
		assertGreater(action[1], -np.pi/4, 'heading is less than -pi/4')
		assertLess(action[1], np.pi/4, 'heading is greater than pi/4')								
		
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
