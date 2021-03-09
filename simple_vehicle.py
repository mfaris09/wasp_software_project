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


