import os
import numpy as np
import unittest
from simple_vehicle import SimpleVehicle


class testing_vehicle(unittest.TestCase):
	
	# 1.
	def test_init_pos_x(self):
		print('initial x-axis position test')		
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertGreater(self.observation[0],-10) 
		self.assertLess(self.observation[0],10)

	# 2.
	def  test_init_pos_y(self):		
		print('initial y-axis position test')			
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertGreater(self.observation[1],10) 
		self.assertLess(self.observation[1],20)		

	# 3.	 
	def test_init_heading(self):
		print('initial heading orientation test')	
		env = SimpleVehicle()		
		self.observation = env.reset()
		
		self.assertLess(self.observation[2],2*np.pi)		

	# environment condition test
	def test_env(self):
		print('test vehicle environment')		
		env  = SimpleVehicle()
		
		self.assertIsNone(env.fig)
		self.assertIsNone(env.ax)			
		
		self.assertGreater(env.xlim[1], env.xlim[0])
		self.assertGreater(env.ylim[1], env.ylim[0])
		
		self.assertGreater(env.dt, 0.)

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
