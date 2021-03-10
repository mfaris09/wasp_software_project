import unittest
import numpy as np
from simple_robot import SimpleRobot, SimpleRobotEnv

class testing_robot(unittest.TestCase):
    robot = SimpleRobot(0., 0., 0.,)
    def test_set_robot_position(self):
        #testing if the robot position is updated properly
        for i in range(-5, 5, 2):
            self.robot.set_robot_position(i, i, i*np.pi)
            self.assertEquals(self.robot.pos_x, i)
            self.assertEquals(self.robot.pos_y, i)
            self.assertEquals(self.robot.phi, i*np.pi)
    
    def test_init(self):
        #testing if the rotation does not add more components
        self.assertEquals(len(self.robot.init_robot_body), 4)
        self.assertEquals(len(self.robot.obstacle_map), self.robot.n_direction)
        
    def test_move(self):
        #testing if the robot moves in x-axis
        self.robot.set_robot_position(0., 0., 0.)
        for i in range(5):
            self.robot.move(1.0, 0., 0.1)
            self.assertGreater(self.robot.pos_x, 0., 'robot does not move positively in x-axis')
        self.robot.set_robot_position(0., 0., 0.)
        for i in range(5):
            self.robot.move(-1.0, 0., 0.1)
            self.assertLess(self.robot.pos_x, 0., 'robot does not move negatively in x-axis')
        #testing if the robot moves in y-axis
        self.robot.set_robot_position(0., 0., np.pi/2)
        for i in range(5):
            self.robot.move(1.0, 0., 0.1)
            self.assertGreater(self.robot.pos_y, 0., 'robot does not move positively in y-axis')
        self.robot.set_robot_position(0., 0., np.pi/2)
        for i in range(5):
            self.robot.move(-1.0, 0., 0.1)
            self.assertLess(self.robot.pos_y, 0., 'robot does not move negatively in y-axis')
        #testing if the robot keeps its position during rotataion
        self.robot.set_robot_position(0., 0., 0.)
        for i in range(-12,12):
            self.robot.move(0.0, i*np.pi/6, 0.1)
            self.assertEqual(self.robot.pos_x, 0., 'robot moves in x axis')
            self.assertEqual(self.robot.pos_y, 0., 'robot moves in y axis')
            
    def test_get_parts(self):
        #test if get_parts return the correct number of robot's part
        robot_body    = self.robot.get_robot_body()
        robot_sensors = self.robot.get_robot_sensors()
        self.assertEqual(len(robot_body), 4, 'quantity of robot body incorrect')
        self.assertEqual(len(robot_sensors), self.robot.n_direction, 'quantity of robot sensors incorrect')
        
class testing_environment(unittest.TestCase):
    env = SimpleRobotEnv()
    
    def test_init(self):
        test_env = SimpleRobotEnv()
        self.assertIsNone(test_env.fig, 'fig is not None')
        self.assertIsNone(test_env.ax,  'ax is not None')
        self.assertGreater(test_env.xlim[1], test_env.xlim[0], 'x-axis is not define properly, xmax should > xmin')
        self.assertGreater(test_env.ylim[1], test_env.ylim[0], 'y-axis is not define properly, ymax should > ymin')
        self.assertGreater(test_env.dt, 0., 'timestep must be greater than zero')
        self.assertGreater(len(test_env.obstacles), 0, 'there must be obstacles created to be avoided by the robot')
        
    def test_render(self):
        self.env.render(hold=False)
        self.assertIsNotNone(self.env.fig, 'fig is None')
        self.assertIsNotNone(self.env.ax,  'ax is None')
        
    def test_get_random_position(self):
        #test if the random position is correct
        self.assertGreater(len(self.env.target_list), 0, 'no target position is available')
        for i in range(100):
            pos_x, pos_y, phi = self.env.get_random_position()
            self.assertIn([pos_x, pos_y], self.env.target_list, 'position is not in target list')
            
    def test_reset(self):
        #test if reset is correct
        self.env.reset()
        self.assertEqual(self.env.action[0], self.env.discrete_action_list[1][0], 'linear speed in reset action is not correct')
        self.assertEqual(self.env.action[1], self.env.discrete_action_list[1][1], 'angular speed in reset action is not correct')
        
    def test_get_state(self):
        #test if the state size is correct
        self.env.reset()
        self.assertEqual(len(self.env.get_state()), self.env.robot.n_direction + 2, 'state size incorrect')
        for i in range(100):
            pos_x, pos_y, phi = self.env.get_random_position()
            self.env.robot.set_robot_position(pos_x, pos_y, phi)
            state = self.env.get_state()
            max_distance = np.max(state[:self.env.robot.n_direction])
            self.assertEqual(max_distance, self.env.robot.camera_far_clipping, 'sensor reading has problem, there must be a sensor that has max value (3.5)')
        #set the robot far from a wall, the sensor must have readings < camera_far_clipping
        self.env.robot.set_robot_position(10., 10., 0.)
        state = self.env.get_state()
        min_distance = np.min(state[:self.env.robot.n_direction])
        self.assertEqual(min_distance, self.env.robot.camera_far_clipping, 'sensor reading incorrect when no obstacle')
        #set the robot near a wall, the sensor must have readings < camera_far_clipping
        self.env.robot.set_robot_position(12., 10., 0.)
        state = self.env.get_state()
        min_distance = np.min(state[:self.env.robot.n_direction])
        self.assertLess(min_distance, self.env.robot.camera_far_clipping, 'sensor reading incorrect with an obstacle')
    
    def test_check_collision(self):
        # Test if the collision is detected properly
        self.env.reset()
        self.env.render()
        collision_bool = self.env.check_collision()
        self.assertFalse(collision_bool, 'robot in reset position must not collide, boolean must be false')
        #set the robot to collide with the wall
        self.env.robot.set_robot_position(14., 10., 0.) 
        self.env.render()
        collision_bool = self.env.check_collision()
        self.assertTrue(collision_bool, 'robot in collision, boolean must be true')
        
    def test_get_reward(self):
        # Test the correctness of the reward given the states
        self.env.robot.set_robot_position(10., 10., 0.)
        state = self.env.get_state()
        reward = self.env.get_reward(state)
        self.assertEqual(reward, -0.1*self.env.action[0], 'No obstacle detected, reward must be -0.1')
        #set the robot to collide with the wall
        self.env.robot.set_robot_position(14., 10., 0.) 
        state = self.env.get_state()
        reward = self.env.get_reward(state)
        self.assertEqual(reward, -1.0*self.env.action[0], 'Collision, reward must be minimum')
        #set the robot in safe situation
        self.env.robot.set_robot_position(12., 10., 0.)
        state = self.env.get_state()
        reward = self.env.get_reward(state)
        self.assertGreater(reward, 0., 'safe situation, reward must be positive')
        #set the robot in warning situation
        self.env.robot.set_robot_position(12.5, 10., 0.)
        state = self.env.get_state()
        reward = self.env.get_reward(state)
        self.assertLess(reward, 0., 'obstacle is near, reward must be negative')
        
    def test_step(self):
        # Set the robot in warning situation
        # Put a robot towards a wall and detecting obstacle
        self.env.robot.set_robot_position(12.5, 10., 0.)
        linear_speed = 1.0
        angular_speed = 0.0
        action = [linear_speed, angular_speed]
        # move forward by 0.1 m, there must be no collision and obstacle is detected
        state, reward, done = self.env.step(action)
        self.assertLess(reward, 0., 'obstacle is near, reward must be negative')
        self.assertFalse(done, 'robot not in collision, boolean must be false')
        min_distance = np.min(state[:self.env.robot.n_direction])
        self.assertLess(min_distance, self.env.robot.camera_far_clipping, 'sensor reading incorrect with an obstacle')
        # Move the robot forward until it collides to the wall
        for i in range(15):
            state, reward, done = self.env.step(action)
        self.assertEqual(reward, -1.0*action[0], 'Collision, reward must be minimum')
        self.assertTrue(done, 'robot in collision, boolean must be true')
        
        