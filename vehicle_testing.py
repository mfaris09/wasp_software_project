import os
import numpy as np
import unittest
from simple_vehicle import SimpleVehicle
import copy

class testing_vehicle(unittest.TestCase):
	
	# 1. Initial position x
	def test_init_pos_x(self):
		print('initial x-axis position test')		
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertGreater(self.observation[0],-10) 
		self.assertLess(self.observation[0],10)

	# 2.Initial position y
	def  test_init_pos_y(self):		
		print('initial y-axis position test')			
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertGreater(self.observation[1],10) 
		self.assertLess(self.observation[1],20)		

	# 3. Initial heading (radian)	 
	def test_init_heading(self):
		print('initial heading orientation test')	
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertLess(self.observation[2],2*np.pi)		

	# 4. environment condition test
	def test_env(self):
		print('test vehicle environment')		
		env  = SimpleVehicle()
		
		self.assertIsNone(env.fig)
		self.assertIsNone(env.ax)			
		
		self.assertGreater(env.xlim[1], env.xlim[0])
		self.assertGreater(env.ylim[1], env.ylim[0])
		
		self.assertGreater(env.dt, 0.)
	
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
				
		self.assertNotEqual(init[0],next_state[0])	
		self.assertNotEqual(init[1],next_state[1])	
		self.assertEqual(init[2],next_state[2])							

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
				
		self.assertNotEqual(init[0],next_state[0])	
		self.assertNotEqual(init[1],next_state[1])					
		self.assertEqual(init[2],next_state[2])									

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
				
		self.assertNotEqual(init[0],next_state[0])	
		self.assertNotEqual(init[1],next_state[1])					
		self.assertLess(init[2],next_state[2])						
	
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
				
		self.assertNotEqual(init[0],next_state[0])	
		self.assertNotEqual(init[1],next_state[1])					
		self.assertLess(init[2],next_state[2])	

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
				
		self.assertNotEqual(init[0],next_state[0])	
		self.assertNotEqual(init[1],next_state[1])					
		self.assertGreater(init[2],next_state[2])							

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
				
		self.assertNotEqual(init[0],next_state[0])	
		self.assertNotEqual(init[1],next_state[1])					
		self.assertGreater(init[2],next_state[2])	

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
		
		self.assertGreater(action,-1)
		self.assertLess(action,6)												
		
		
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
