import numpy as np
from shapely.geometry import Point, box, Polygon
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt
import numbers

class SimpleRobot():
    def __init__(self, pos_x, pos_y, phi):
        assert isinstance(pos_x, numbers.Number)
        assert isinstance(pos_y, numbers.Number)
        assert isinstance(phi,   numbers.Number)
        self.set_robot_position(pos_x, pos_y, phi)
        self.wheel_distance   = 2.0 #distance between 2 wheels 
        self.wheel_radius     = 0.8 
        self.init_robot_body  = []
        robot_center_point    = (0,0)
        robot_radius          = 1.5
        robot_wheel_inner_pos = 0.9
        robot_wheel_outer_pos = 1.1
        self.init_robot_body.append(Point(robot_center_point).buffer(robot_radius)) #outer circle body
        self.init_robot_body.append(box(-robot_wheel_outer_pos, -self.wheel_radius, -robot_wheel_inner_pos, self.wheel_radius)) #left wheel
        self.init_robot_body.append(box( robot_wheel_inner_pos, -self.wheel_radius,  robot_wheel_outer_pos, self.wheel_radius)) #right wheel
        # adding triangle as the front marker
        triangle_left_corner  = (-0.6, 1)
        triangle_top_corner   = (0, 1.3)
        triangle_right_corner = (0.6, 1)
        
        self.init_robot_body.append(Polygon([triangle_left_corner, triangle_top_corner, triangle_right_corner])) 
        
        # Robot sensor
        self.camera_near_clipping = 1.5 #in meters
        self.camera_far_clipping  = 3.5 #in meters
        self.sensing_range        = self.camera_far_clipping - self.camera_near_clipping
        self.camera_fov_angle     = 90.0 #degree
        self.n_fov_zones          = 5 # number of simulated sensor. It can also be seen as how many zones that camera's Field of View (FoV) will be divided
        self.fov_zones_list       = np.linspace(-self.camera_fov_angle, self.camera_fov_angle, self.n_fov_zones+1)
        self.obstacle_map = []
        self.obstacle_distances = np.ones((self.n_fov_zones))*self.camera_far_clipping
        for i in range(self.n_fov_zones):
            self.obstacle_map.append(Polygon([
                                    [self.camera_near_clipping*np.sin(np.radians(self.fov_zones_list[i])),  self.camera_near_clipping*np.cos(np.radians(self.fov_zones_list[i]))],
                                    [self.camera_near_clipping*np.sin(np.radians(self.fov_zones_list[i+1])),self.camera_near_clipping*np.cos(np.radians(self.fov_zones_list[i+1]))],
                                    [self.camera_far_clipping*np.sin(np.radians(self.fov_zones_list[i+1])),self.camera_far_clipping*np.cos(np.radians(self.fov_zones_list[i+1]))],
                                    [self.camera_far_clipping*np.sin(np.radians(self.fov_zones_list[i])),  self.camera_far_clipping*np.cos(np.radians(self.fov_zones_list[i]))]]))
        
        # Rotating robot's components to rectify the robot drawing 
        self.robot_rotation_point = Point(0, 0)
        angle_correction = -np.pi/2
        for i in range(len(self.init_robot_body)):
            self.init_robot_body[i] = rotate(self.init_robot_body[i], angle_correction, use_radians=True, origin=self.robot_rotation_point)
        for i in range(len(self.obstacle_map)):
            self.obstacle_map[i] = rotate(self.obstacle_map[i], angle_correction, use_radians=True, origin=self.robot_rotation_point)
        # Set self.robot_body and self.robot_sensors variables
        self.get_robot_body()
        self.get_robot_sensors()
        
    def set_robot_position(self, pos_x, pos_y, phi):
        # Set robot position manually by giving the new position
        assert isinstance(pos_x, numbers.Number)
        assert isinstance(pos_y, numbers.Number)
        assert isinstance(phi,   numbers.Number)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.phi = phi
        
    def move(self, linear_speed, angular_speed, timestep):
        # Calculate robot position from the given speed and timestep which is based on differential drive kinematics
        assert isinstance(linear_speed,  numbers.Number)
        assert isinstance(angular_speed, numbers.Number)
        assert isinstance(timestep,      numbers.Number)
        self.pos_x += linear_speed*np.cos(self.phi)*timestep
        self.pos_y += linear_speed*np.sin(self.phi)*timestep
        self.phi   += angular_speed*timestep
        return self.pos_x, self.pos_y, self.phi
    
    def get_parts(self, part_list):
        # collect all relevant components given in the part_list
        assert len(part_list) > 0
        
        moved_parts = []
        for part in part_list:
            rotated_part =  rotate(part, self.phi, use_radians=True, origin=self.robot_rotation_point)
            translated_part = translate(rotated_part, self.pos_x, self.pos_y)
            moved_parts.append(translated_part)
        return moved_parts
    
    def get_robot_body(self):
        self.robot_body = self.get_parts(self.init_robot_body)
        return self.robot_body
    
    def get_robot_sensors(self):
        self.robot_sensors = self.get_parts(self.obstacle_map)
        return self.robot_sensors

class TranslationSpeed():
    def __init__(self, slow, medium, fast):
        assert isinstance(slow, numbers.Number)
        assert isinstance(medium, numbers.Number)
        assert isinstance(fast, numbers.Number)
        #translation speed
        self.__slow   = slow
        self.__medium = medium
        self.__fast   = fast
        
    def set_slow(self, slow):
        assert isinstance(slow, numbers.Number)
        self.__slow   = slow
    def get_slow(self):
        return self.__slow
            
    def set_medium(self, medium):
        assert isinstance(medium, numbers.Number)
        self.__medium   = medium
    def get_medium(self):
        return self.__medium
        
    def set_fast(self, fast):
        assert isinstance(fast, numbers.Number)
        self.__fast   = fast
    def get_fast(self):
        return self.__fast
    
        
class TurningSpeed():
    def __init__(self, turning_speed):
        assert isinstance(turning_speed, numbers.Number)
        self.__left     = -turning_speed
        self.__straight =  0.
        self.__right    =  turning_speed
        
    def set_speed(self, turning_speed):
        self.__left     = -turning_speed
        self.__right    =  turning_speed
        
    def get_left(self):
        return self.__left
    def get_straight(self):
        return self.__straight
    def get_right(self):
        return self.__right
        
    
class DiscreteAction():
    def __init__(self, slow_speed, medium_speed, fast_speed, turning_speed):
        assert isinstance(slow_speed, numbers.Number)
        assert isinstance(medium_speed, numbers.Number)
        assert isinstance(fast_speed, numbers.Number)
        assert isinstance(turning_speed, numbers.Number)
        self.__translation_speed = TranslationSpeed(slow_speed, medium_speed, fast_speed)
        self.__turning_speed     = TurningSpeed(turning_speed)
        
        self.__action_list = []
        self.__action_list.append(np.array([self.__translation_speed.get_slow(),   self.__turning_speed.get_left()]))
        self.__action_list.append(np.array([self.__translation_speed.get_slow(),   self.__turning_speed.get_straight()]))
        self.__action_list.append(np.array([self.__translation_speed.get_slow(),   self.__turning_speed.get_right()]))
        self.__action_list.append(np.array([self.__translation_speed.get_medium(), self.__turning_speed.get_left()]))
        self.__action_list.append(np.array([self.__translation_speed.get_medium(), self.__turning_speed.get_straight()]))
        self.__action_list.append(np.array([self.__translation_speed.get_medium(), self.__turning_speed.get_right()]))
        self.__action_list.append(np.array([self.__translation_speed.get_fast(),   self.__turning_speed.get_left()]))
        self.__action_list.append(np.array([self.__translation_speed.get_fast(),   self.__turning_speed.get_straight()]))
        self.__action_list.append(np.array([self.__translation_speed.get_fast(),   self.__turning_speed.get_right()]))
        
        self.action_length = len(self.__action_list[0])
        self.number_of_available_action = len(self.__action_list)
        
    def get_action(self, action_number):
        assert (action_number >= 0) and (action_number < self.number_of_available_action) 
        return self.__action_list[action_number]
    
        
class SimpleRobotEnv():
    def __init__(self):
        self.dt = .1 # timestep

        self.figure = None
        self.axes   = None

        self.limit_x_axis = [-15, 15]
        self.limit_y_axis = [-2.5, 27.5]
        
        self.obstacles = []
        outer_wall_thick = 0.001
        self.obstacles.append(box(self.limit_x_axis[0], self.limit_y_axis[0], self.limit_x_axis[1]+outer_wall_thick, self.limit_y_axis[0]+outer_wall_thick)) #outer wall
        self.obstacles.append(box(self.limit_x_axis[0], self.limit_y_axis[0], self.limit_x_axis[0]+outer_wall_thick, self.limit_y_axis[1]+outer_wall_thick)) #outer wall
        self.obstacles.append(box(self.limit_x_axis[0], self.limit_y_axis[1], self.limit_x_axis[1]+outer_wall_thick, self.limit_y_axis[1]+outer_wall_thick)) #outer wall
        self.obstacles.append(box(self.limit_x_axis[1], self.limit_y_axis[0], self.limit_x_axis[1]+outer_wall_thick, self.limit_y_axis[1]+outer_wall_thick)) #outer wall
        
        self.obstacles.append(Polygon([(0,2.5),(-5,5),(5,5)]))
        self.obstacles.append(box(-12,13,-5,14))
        self.obstacles.append(box(  0,20,10,21))
        self.obstacles.append(Point(3,12.5).buffer(2))
        
        bottom_left_pos  = [-10, 2.5]
        bottom_right_pos = [ 10, 7.5]
        top_left_pos     = [-10, 20]
        top_right_pos    = [ 10, 25]
        self.target_list = [bottom_left_pos, bottom_right_pos, top_left_pos, top_right_pos]
        pos_x, pos_y, phi = self.get_random_position()
        self.robot = SimpleRobot(pos_x, pos_y, phi)
        
        #List discrete actions for RL
        self.rl_actions = DiscreteAction(0.3, 0.6, 0.9, np.pi/8)
        
        self.reset()
        self.state_size = len(self.get_state())
        
    def init_figure(self):
        assert self.axes is None
        self.figure, self.axes = plt.subplots()
        
    def plot_object(self, objects_list, color=None):
        for object in objects_list:
            x, y = object.exterior.xy
            if color == None:
                self.axes.plot(x, y)
            else:
                self.axes.plot(x, y, color)            
        
    def init_axes(self):
        self.axes.plot([], [], "C0", linewidth=3)
        for _ in range(4):
            self.axes.plot([], [], "C1", linewidth=3)
        self.axes.plot([], [], "C2o", markersize=6)

        self.axes.grid()
        self.axes.set_xlim(self.limit_x_axis)
        self.axes.set_aspect("equal")
        self.axes.set_ylim(self.limit_y_axis)
        
        self.plot_object(self.obstacles)
        self.plot_object(self.robot.get_robot_body(), 'k')
        self.plot_object(self.robot.get_robot_sensors(), 'silver')
            
    def render(self, hold=False):
        # Draw the plot so we can see the visualization
        assert type(hold) == bool
        def matching_plot(plot_data, updated_data):
            # matching the plot with robot component
            # this helper function is used for animating the robot movement
            assert len(plot_data) == len(updated_data)
            for i in range(len(plot_data)):
                plot_data[i].set_data(updated_data[i].exterior.xy)
        
        if self.figure is None:
            self.init_figure()
            
        if not self.axes.lines:
            self.init_axes()
                
        # Retrieve robot's plot so it moves after initiated
        # This is to avoid redrawing the plot every cycle
        initiated_object_plot = 6
        obstacle_idx = initiated_object_plot + len(self.obstacles)
        robot_idx    = obstacle_idx + len(self.robot.get_robot_body())
        sensor_idx   = robot_idx + len(self.robot.get_robot_sensors())
        robot_body   = self.axes.lines[obstacle_idx:robot_idx]
        robot_sensor = self.axes.lines[robot_idx:sensor_idx]
        matching_plot(robot_body, self.robot.get_robot_body())
        matching_plot(robot_sensor, self.robot.get_robot_sensors())
        
        self.axes.relim()
        self.axes.autoscale_view()
        pause_time_for_plotting = 1e-07 # in seconds
        plt.draw()
        plt.pause(pause_time_for_plotting)
        if hold:
            plt.show()


    def get_random_position(self):
        # Generate a pre-defined target position where the robot can appear without colliding to any obstacle
        selected_target = np.random.randint(len(self.target_list))
        pos_x = self.target_list[selected_target][0]
        pos_y = self.target_list[selected_target][1]
        phi   = np.random.uniform(-np.pi, np.pi)
        return pos_x, pos_y, phi
    
    def reset(self):
        # Reset robot to to pre-defined position 
        self.action = self.rl_actions.get_action(1)
        pos_x, pos_y, phi = self.get_random_position()
        self.robot.set_robot_position(pos_x, pos_y, phi)
        return self.get_state()
    
    def get_closest_obstacle_distance(self, obstacle_object, obstacle_distance, robot_center, sensor_zone):
        intersection_poylgon  = obstacle_object.intersection(sensor_zone)
        return min(obstacle_distance, robot_center.distance(intersection_poylgon))
    
    def get_state(self):
        # Get the state of the robot that contains:
        # 1. list of Sensor readings with the size equals to robot.n_fov_zones
        # 2. the action taken by the robot (stored as the last two of the list member)
        self.render()
        obstacle_distances = np.ones((self.robot.n_fov_zones))*self.robot.camera_far_clipping
        robot_center       = Point((self.robot.pos_x, self.robot.pos_y))
        for obstacle in self.obstacles:
            for i in range(self.robot.n_fov_zones):
                if obstacle.intersects(self.robot.robot_sensors[i]):
                    obstacle_distances[i] = self.get_closest_obstacle_distance(obstacle, obstacle_distances[i], robot_center, self.robot.robot_sensors[i])
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
        assert len(state) == self.robot.n_fov_zones + self.rl_actions.action_length
        
        closest_distance = np.min(state[:self.robot.n_fov_zones]) #get the closest distance from all sensors
        robot_linear_speed = state[self.robot.n_fov_zones]
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
        assert len(action) == 2
        
        linear_speed, angular_speed = action
        self.action = action 
        
        self.robot.move(linear_speed, angular_speed, self.dt)
        state = self.get_state()

        reward = self.get_reward(state)
        done = self.check_collision()
        return state, reward, done
    
    def discrete_action(self,action_number):
        # Select pre-defined discrete action for RL
        assert (action_number >= 0) and action_number < self.rl_actions.number_of_available_action
        return self.rl_actions.get_action(action_number)
    
    def discrete_step(self,action_number):
        # Run step function using discrete action
        assert (action_number >= 0) and action_number < self.rl_actions.number_of_available_action
        action = self.discrete_action(action_number)
        return self.step(action)
    
    