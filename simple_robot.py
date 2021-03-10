import numpy as np
from shapely.geometry import Point, box, Polygon
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt

class SimpleRobot():
    def __init__(self, pos_x, pos_y, phi):
        self.set_robot_position(pos_x, pos_y, phi)
        self.wheel_distance = 2.0 #distance between 2 wheels 
        self.wheel_radius = 0.8 
        self.init_robot_body = []
        self.init_robot_body.append(Point(0,0).buffer(1.5)) #outer circle body
        self.init_robot_body.append(box(-1.1, -self.wheel_radius, -0.9, self.wheel_radius)) #left wheel
        self.init_robot_body.append(box( 0.9, -self.wheel_radius,  1.1, self.wheel_radius)) #right wheel
        self.init_robot_body.append(Polygon([(-0.6, 1),(0, 1.3),(0.6, 1)])) #triangle as the front marker
        
        # Robot sensor
        self.camera_near_clipping = 1.5 #in meters
        self.camera_far_clipping  = 3.5 #in meters
        self.sensing_range        = self.camera_far_clipping - self.camera_near_clipping
        self.camera_fov_angle     = 90.0 #degree
        self.n_direction          = 5 # number of simulated sensor. It can also be seen as how many zones that camera's Field of View (FoV) will be divided
        self.direction_list       = np.linspace(-self.camera_fov_angle, self.camera_fov_angle, self.n_direction+1)
        self.obstacle_map = []
        self.obstacle_distances = np.ones((self.n_direction))*self.camera_far_clipping
        for i in range(self.n_direction):
            self.obstacle_map.append(Polygon([
                                    [self.camera_near_clipping*np.sin(np.radians(self.direction_list[i])),  self.camera_near_clipping*np.cos(np.radians(self.direction_list[i]))],
                                    [self.camera_near_clipping*np.sin(np.radians(self.direction_list[i+1])),self.camera_near_clipping*np.cos(np.radians(self.direction_list[i+1]))],
                                    [self.camera_far_clipping*np.sin(np.radians(self.direction_list[i+1])),self.camera_far_clipping*np.cos(np.radians(self.direction_list[i+1]))],
                                    [self.camera_far_clipping*np.sin(np.radians(self.direction_list[i])),  self.camera_far_clipping*np.cos(np.radians(self.direction_list[i]))]]))
        
        # Rotating robot's components to rectify the robot drawing 
        angle_correction = -np.pi/2
        for i in range(len(self.init_robot_body)):
            self.init_robot_body[i] = rotate(self.init_robot_body[i], angle_correction, use_radians=True, origin=Point(0, 0))
        for i in range(len(self.obstacle_map)):
            self.obstacle_map[i] = rotate(self.obstacle_map[i], angle_correction, use_radians=True, origin=Point(0, 0))
        # Set self.robot_body and self.robot_sensors variables
        self.get_robot_body()
        self.get_robot_sensors()
        
    def set_robot_position(self, pos_x, pos_y, phi):
        # Set robot position manually by giving the new position
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.phi = phi
        
    def move(self, linear_speed, angular_speed, timestep):
        # Calculate robot position from the given speed and timestep which is based on differential drive kinematics
        self.pos_x += linear_speed*np.cos(self.phi)*timestep
        self.pos_y += linear_speed*np.sin(self.phi)*timestep
        self.phi   += angular_speed*timestep
        return self.pos_x, self.pos_y, self.phi
    
    def get_parts(self, part_list):
        # collect all relevant components given in the part_list
        moved_parts = []
        for part in part_list:
            rotated_part =  rotate(part, self.phi, use_radians=True, origin=Point(0, 0))
            translated_part = translate(rotated_part, self.pos_x, self.pos_y)
            moved_parts.append(translated_part)
        return moved_parts
    
    def get_robot_body(self):
        self.robot_body = self.get_parts(self.init_robot_body)
        return self.robot_body
    
    def get_robot_sensors(self):
        self.robot_sensors = self.get_parts(self.obstacle_map)
        return self.robot_sensors
    
class SimpleRobotEnv():
    def __init__(self):
        self.dt = .1 # timestep

        self.fig = None
        self.ax = None

        self.xlim = [-15, 15]
        self.ylim = [-2.5, 27.5]
        
        self.obstacles = []
        outer_wall_thick = 0.001
        self.obstacles.append(box(self.xlim[0], self.ylim[0], self.xlim[1]+outer_wall_thick, self.ylim[0]+outer_wall_thick)) #outer wall
        self.obstacles.append(box(self.xlim[0], self.ylim[0], self.xlim[0]+outer_wall_thick, self.ylim[1]+outer_wall_thick)) #outer wall
        self.obstacles.append(box(self.xlim[0], self.ylim[1], self.xlim[1]+outer_wall_thick, self.ylim[1]+outer_wall_thick)) #outer wall
        self.obstacles.append(box(self.xlim[1], self.ylim[0], self.xlim[1]+outer_wall_thick, self.ylim[1]+outer_wall_thick)) #outer wall
        
        self.obstacles.append(Polygon([(0,2.5),(-5,5),(5,5)]))
        self.obstacles.append(box(-12,13,-5,14))
        self.obstacles.append(box(  0,20,10,21))
        self.obstacles.append(Point(3,12.5).buffer(2))
        
        self.target_list = [[-10,2.5], [10,7.5], [-10, 20], [10,25]]
        pos_x, pos_y, phi = self.get_random_position()
        self.robot = SimpleRobot(pos_x, pos_y, phi)
        
        #List discrete actions for RL
        self.discrete_action_list = []
        slow_speed   = 0.3
        medium_speed = 0.6
        fast_speed   = 0.9
        turn_left  = -np.pi/8
        straight   = 0.
        turn_right = np.pi/8
        self.discrete_action_list.append(np.array([slow_speed, turn_left]))
        self.discrete_action_list.append(np.array([slow_speed, straight]))
        self.discrete_action_list.append(np.array([slow_speed, turn_right]))
        self.discrete_action_list.append(np.array([medium_speed, turn_left]))
        self.discrete_action_list.append(np.array([medium_speed, straight]))
        self.discrete_action_list.append(np.array([medium_speed, turn_right]))
        self.discrete_action_list.append(np.array([fast_speed, turn_left]))
        self.discrete_action_list.append(np.array([fast_speed, straight]))
        self.discrete_action_list.append(np.array([fast_speed, turn_right]))
        self.discrete_action_size = len(self.discrete_action_list)
        
        self.reset()
        self.state_size = len(self.get_state())
        
    def render(self, hold=False):
        # Draw the plot so we can see the visualization
        def matching_plot(plot_data, updated_data):
            # matching the plot with robot component
            # this helper function is used for animating the robot movement
            assert len(plot_data) == len(updated_data)
            for i in range(len(plot_data)):
                plot_data[i].set_data(updated_data[i].exterior.xy)
        
        if self.fig is None:
            assert self.ax is None
            self.fig, self.ax = plt.subplots()
        if not self.ax.lines:
            self.ax.plot([], [], "C0", linewidth=3)
            for _ in range(4):
                self.ax.plot([], [], "C1", linewidth=3)
            self.ax.plot([], [], "C2o", markersize=6)

            self.ax.grid()
            self.ax.set_xlim(self.xlim)
            self.ax.set_aspect("equal")
            self.ax.set_ylim(self.ylim)
        
            for obstacle in self.obstacles:
                x,y = obstacle.exterior.xy
                self.ax.plot(x, y)
                
            for body in self.robot.get_robot_body():
                x,y = body.exterior.xy
                self.ax.plot(x, y, 'k')
            
            for sensor in self.robot.get_robot_sensors():
                x,y = sensor.exterior.xy
                self.ax.plot(x, y, 'silver')
                
        # Retrieve robot's plot so it moves after initiated
        # This is to avoid redrawing the plot every cycle
        obstacle_idx = 6 + len(self.obstacles)
        robot_idx    = obstacle_idx + len(self.robot.get_robot_body())
        sensor_idx   = robot_idx + len(self.robot.get_robot_sensors())
        robot_body   = self.ax.lines[obstacle_idx:robot_idx]
        robot_sensor = self.ax.lines[robot_idx:sensor_idx]
        matching_plot(robot_body, self.robot.get_robot_body())
        matching_plot(robot_sensor, self.robot.get_robot_sensors())
        
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(1e-07)
        if hold:
            plt.show()


    def get_random_position(self):
        # Generate a pre-defined target position where the robot can appear without colliding to any obstacle
        target_n = np.random.randint(len(self.target_list))
        pos_x = self.target_list[target_n][0]
        pos_y = self.target_list[target_n][1]
        phi   = np.random.uniform(-np.pi, np.pi)
        return pos_x, pos_y, phi
    
    def reset(self):
        # Reset robot to to pre-defined position 
        self.action = self.discrete_action_list[1]
        pos_x, pos_y, phi = self.get_random_position()
        self.robot.set_robot_position(pos_x, pos_y, phi)
        return self.get_state()
    
    def get_state(self):
        # Get the state of the robot that contains:
        # 1. list of Sensor readings with the size equals to robot.n_direction
        # 2. the action taken by the robot (stored as the last two of the list member)
        self.render()
        obstacle_distances = np.ones((self.robot.n_direction))*self.robot.camera_far_clipping
        robot_center       = Point((self.robot.pos_x, self.robot.pos_y))
        for obstacle in self.obstacles:
            for i in range(self.robot.n_direction):
                if obstacle.intersects(self.robot.robot_sensors[i]):
                    intersection_poylgon = obstacle.intersection(self.robot.robot_sensors[i])
                    obstacle_distances[i] = min(obstacle_distances[i], robot_center.distance(intersection_poylgon))
        return np.concatenate((obstacle_distances, self.action))
    
    def check_collision(self):
        robot_center = Point((self.robot.pos_x, self.robot.pos_y))
        for obstacle in self.obstacles:
            for i in range(len(self.robot.robot_body)):
                if obstacle.intersects(self.robot.robot_body[i]):
                    return True
        return False
    
    def get_reward(self, state):
        # Calculate reward value from the given state
        # It consider the closest distance of the obstacle
        # and multiply by the speed so risky dangerous situation is penalized more with faster speed 
        # and safe situation is rewarded more with faster speed
        closest_distance = np.min(state[:self.robot.n_direction]) #get the closest distance from all sensors
        robot_linear_speed = state[self.robot.n_direction]
        if self.check_collision():
            reward = -1.0 * robot_linear_speed
        elif closest_distance >= self.robot.camera_far_clipping:
            reward = -0.1 * robot_linear_speed
        elif closest_distance >= (2/3*self.robot.sensing_range + self.robot.camera_near_clipping):
            reward = 1 * robot_linear_speed
        elif closest_distance >= (1/3*self.robot.sensing_range + self.robot.camera_near_clipping):
            reward = -0.3 * robot_linear_speed
        else: 
            reward = -0.5 * robot_linear_speed
        return reward    
    
    def step(self, action):
        # step function for RL
        # The input is an action and returning state, reward and done status
        # done is set true when the robot collides with obstacle
        linear_speed, angular_speed = action
        self.action = action 
        
        self.robot.move(linear_speed, angular_speed, self.dt)
        state = self.get_state()

        reward = self.get_reward(state)
        done = self.check_collision()
        return state, reward, done
    
    def discrete_action(self,action_number):
        # Select pre-defined discrete action for RL
        assert (action_number >= 0) and action_number < self.discrete_action_size
        return self.discrete_action_list[action_number]
    
    def discrete_step(self,action_number):
        # Run step function using discrete action
        action = self.discrete_action(action_number)
        return self.step(action)
    
    