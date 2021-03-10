import numpy as np
import matplotlib.pyplot as plt
from utils import rotation_matrix

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
		self.discrete_action_size = len(self.action_list)		

	# for giving initial condition
	# chosen randomly
	def reset(self):	
		self.action = np.array([0., np.pi / 4.])

		# initial states: pos x, pos y, direct
		# chosen to be random within bounds
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

	def set_vehicle_position(self, pos_x, pos_y, direction):	
		self.action = np.array([0., np.pi / 4.])
		
		assert self.xlim[0] < pos_x < self.xlim[1]
		assert 10. < pos_y < self.ylim[1]
		assert direction < 2*np.pi

		# initial states: pos x, pos y, direct
		# chosen to be andom within bounds
		self.state = np.array([
			# rear axle center x-position
			pos_x, # plus minus 10
			# rear axle center y-position 
			pos_y, # positive 10 - 20
			# direction of the car
			direction
		])
		
		observation = self.state # get initial values to "sensor"
				
		return observation		
	
	def step(self, action):
		speed, phi = action
		self.action = action 
		
		# kinematic model
		# states: position x, position y, and heading angle
		x, y, theta = self.state
		dxdt = speed * np.cos(theta)
		dydt = speed * np.sin(theta)
		dthetadt = speed * np.tan(phi) / self.vehicle_mid_length
		
		dstatedt = np.array([dxdt, dydt, dthetadt])
		self.state += dstatedt * self.dt
		observation = self.state
				
		# tmp: contain xy coordinate of 4 points of the vehicle rectangle 
		# array 0: x axis, array 1: y axis
		tmp = self.vehicle_body()
		x, y, theta = self.state
		# rotate
		tmp = np.dot(tmp, rotation_matrix(theta))
		# translate
		tmp += np.array([[x, y]])
		self.tmpstep = tmp 	
		
		# calculate reward
		reward = 1. / (np.sqrt(self.state[0] ** 2 + self.state[1] ** 2) + np.sqrt(self.dt*dxdt ** 2 + self.dt*dydt ** 2)) 

		# done condition: pass within parking line or hit the boundaries, wall, obstacles 
		done = (self._border_fcn(tmp[:, 0]) > tmp[:, 1]).any() \
                or (tmp[:, 0] < self.xlim[0]).any() \
                or (tmp[:, 0] > self.xlim[1]).any() \
                or (tmp[:, 1] < self.ylim[0]).any() \
				or (tmp[:, 1] > self.ylim[1]).any()
		
		return observation, reward, done

	def discrete_action(self,action_number):
		assert (action_number >= 0) and action_number < len(self.action_list)
		action = self.action_list[action_number]
		# print('discrete action :', action)
		return action

	def discrete_step(self,action_number):
		action = self.discrete_action(action_number)
		discrete_step = self.step(action)
		# print('discrete step :', discrete_step)
		return discrete_step

	# define vehicle size
	def vehicle_body(self):
		bbox = np.array([
			[self.vehicle_mid_length + self.vehicle_front_length,  .5 * self.vehicle_width],
			[self.vehicle_mid_length + self.vehicle_front_length, -.5 * self.vehicle_width],
			[        -self.vehicle_rear_length, -.5 * self.vehicle_width],
			[        -self.vehicle_rear_length,  .5 * self.vehicle_width]
		])
		return bbox

	# rear wheel function
	def _rear_wheel(self):
		wheel = np.array([
			[-self.wheel_diameter, 0.], 
			[ self.wheel_diameter, 0.]
		])
		return wheel
	
	# front wheel function
	def _front_wheel(self):
		wheel = self._rear_wheel() 
		s, phi = self.action
		wheel = np.dot(wheel, rotation_matrix(phi))
		return wheel		
		
	# for describing parking border
	def _border_fcn(self, x):
		ParkWidthScale = .7
		ParkLengthScale = 2.
		ParkOutWallPos = 5.	
		parking_scale = -self.scaley * self.vehicle_total_length * (np.sign(x + self.scalex * ParkWidthScale * self.vehicle_width) - 
		np.sign(x - self.scalex * ParkWidthScale * self.vehicle_width)) / ParkLengthScale + ParkOutWallPos
		
		return parking_scale
		
	def render(self):
		if self.fig is None:
			assert self.ax is None
			self.fig, self.ax = plt.subplots()
		
		# plotting border and dot on vehicle and walls	
		if not self.ax.lines:
			self.ax.plot([], [], "C0", linewidth=3)
			for _ in range(4):
				self.ax.plot([], [], "C1", linewidth=3)
			self.ax.plot([], [], "C2o", markersize=6)
		
			x = np.linspace(-15, 15, 1000)
			y = self._border_fcn(x)
			self.ax.plot(x, y, "C3", linewidth=3) # plot red wall
			self.ax.plot([0], [0], "C3o", markersize=6) # plot red dot on parking
	
			self.ax.grid()
			self.ax.set_xlim(self.xlim)
			self.ax.set_aspect("equal")
			self.ax.set_ylim(self.ylim)
					
		bbox, lfw, rfw, lrw, rrw, center = self.ax.lines[:6]
						
		# plotting vehicle's parts
		tmp = self.vehicle_body()
		x, y, theta = self.state
		# rotate
		tmp = np.dot(tmp, rotation_matrix(theta))
		# translate
		tmp += np.array([[x, y]])
		# repeat to close the drawed object
		tmp = np.concatenate([tmp, tmp[[0]]])
		bbox.set_data(tmp.T)

		# left front wheel	
		tmp = self._front_wheel()
		tmp += np.array([[self.vehicle_mid_length,  self.front_wheel_bar / 2.]]) 
		# rotate
		tmp = np.dot(tmp, rotation_matrix(theta))
		# translate
		tmp += np.array([[x, y]])
		lfw.set_data(tmp.T) 

		# right front wheel		
		tmp = self._front_wheel()
		tmp += np.array([[self.vehicle_mid_length, -self.front_wheel_bar / 2.]])
		# rotate
		tmp = np.dot(tmp, rotation_matrix(theta))
		# translate
		tmp += np.array([[x, y]])
		rfw.set_data(tmp.T) 
	
		# left rear wheel	
		tmp = self._rear_wheel()
		tmp += np.array([[0.,  self.rear_wheel_bar / 2.]])
		# rotate
		tmp = np.dot(tmp, rotation_matrix(theta))
		# translate
		tmp += np.array([[x, y]])
		lrw.set_data(tmp.T) 
		
		# right rear wheel	
		tmp = self._rear_wheel()
		tmp += np.array([[0., -self.rear_wheel_bar / 2.]])
		# rotate
		tmp = np.dot(tmp, rotation_matrix(theta))
		# translate
		tmp += np.array([[x, y]])
		rrw.set_data(tmp.T) 
	
		center.set_data([x], [y]) # 
	
		self.ax.relim() # 
		self.ax.autoscale_view()
		plt.draw()
		
		# process delay
		plt.pause(1e-07)			
