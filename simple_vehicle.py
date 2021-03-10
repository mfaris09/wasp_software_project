import numpy as np
import matplotlib.pyplot as plt

class SimpleVehicle():
	# define constructor
	# using self enables all access to all instances defined within a class, including its methods and attributes
	def __init__(self):
		#solver timestep
		self.dt = .05
		
		self.fig = None #
		self.ax = None # 
				
		self.xlim = [-15, 15] # bound on x 
		self.ylim = [-2.5, 27.5] # bound on y
		
		# physical parameters
		self.vehicle_mid_length = 2.815 
		self.vehicle_front_length = 1.054
		self.vehicle_rear_length = 0.910 
		self.vehicle_total_length = self.vehicle_mid_length + self.vehicle_front_length + self.vehicle_rear_length 
		self.front_wheel_bar = 1.586
		self.vehicle_width = 2.096 
		self.rear_wheel_bar = 1.557
		self.wheel_diameter = 0.5		

		# general scales
		self.scalex = 2.
		self.scaley = 1.5
		
		# def list of actions (discrete) 
		self.action_list = []
		
		forward_lin_speed = 1
		forward_lin_speed_fast = 3
				
		straight = 0
		turn_left  = np.pi/4
		turn_right = -np.pi/4
		
		self.action_list.append(np.array([forward_lin_speed, turn_left]))	# 0		
		self.action_list.append(np.array([forward_lin_speed_fast, turn_left]))	# 1		
		self.action_list.append(np.array([forward_lin_speed, straight])) # 2			
		self.action_list.append(np.array([forward_lin_speed_fast, straight])) # 3			
		self.action_list.append(np.array([forward_lin_speed, turn_right])) # 4			
		self.action_list.append(np.array([forward_lin_speed_fast, turn_right])) # 5			
				
		# define number of states and actions		
		self.state_size = 3
		self.action_size = len(self.action_list)		

	# for giving initial condition
	# chosen randomly
	def reset(self):	
		self.action = np.array([0., np.pi / 4.])

		# initial states: pos x, pos y, direct
		# chosen to be andom within bounds
		self.state = np.array([
			# rear axle center x-position
			20. * np.random.rand() - 10., # plus minus 10
			# rear axle center y-position 
			10. * np.random.rand() + 10., # positive 10 - 20
			# direction of the car
			2. * np.pi * np.random.rand() # within 360 degree
		])
		
		observation = self.state # get initial values to "sensor"
				
		return observation
